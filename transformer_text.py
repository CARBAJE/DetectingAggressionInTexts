import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def numericalize(text, vocab, max_len=100):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_len]]
    return torch.tensor(ids, dtype=torch.long)


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels_bin, labels_multi, vocab, max_len=100):
        self.texts = texts
        self.labels_bin = labels_bin
        self.labels_multi = labels_multi
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = numericalize(self.texts[idx], self.vocab, self.max_len)
        return x, self.labels_bin[idx], self.labels_multi[idx]


def collate_fn(batch):
    texts, labels_bin, labels_multi = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded, torch.tensor(labels_bin), torch.tensor(labels_multi)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(
            x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, dropout, forward_expansion, num_classes, device):
        super().__init__()
        self.device = device
        self.encoder = Encoder(
            vocab_size, embed_size, num_layers, heads, device,
            forward_expansion, dropout, max_length=100
        )
        self.fc_binary = nn.Linear(embed_size, 1)
        self.fc_multi = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        enc = self.encoder(x, mask)
        pooled = enc.mean(dim=1)
        binary_out = torch.sigmoid(self.fc_binary(pooled)).squeeze()
        multi_out = self.fc_multi(pooled)
        return binary_out, multi_out


def train(model, dataloader, optimizer, criterion_bin, criterion_multi):
    model.train()
    total_loss = 0
    for x, y_bin, y_multi in dataloader:
        x, y_bin, y_multi = x.to(model.device), y_bin.to(
            model.device).float(), y_multi.to(model.device)
        optimizer.zero_grad()
        pred_bin, pred_multi = model(x)
        loss_bin = criterion_bin(pred_bin, y_bin)
        loss_multi = criterion_multi(pred_multi, y_multi)
        loss = loss_bin + loss_multi
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader):
    model.eval()
    all_preds_bin, all_true_bin = [], []
    all_preds_multi, all_true_multi = [], []

    with torch.no_grad():
        for x, y_bin, y_multi in dataloader:
            x = x.to(model.device)
            pred_bin, pred_multi = model(x)
            preds_bin = (pred_bin > 0.5).int().cpu().tolist()
            preds_multi = pred_multi.argmax(dim=1).cpu().tolist()
            all_preds_bin += preds_bin
            all_preds_multi += preds_multi
            all_true_bin += y_bin.tolist()
            all_true_multi += y_multi.tolist()

    print("\n=== Clasificación Binaria ===")
    print(classification_report(all_true_bin, all_preds_bin, zero_division=0))
    print("\n=== Clasificación Multiclase ===")
    print(classification_report(all_true_multi, all_preds_multi, zero_division=0))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dataset = load_dataset(
        "thefrankhsu/hate_speech_twitter", split="train")

    df_t = raw_dataset.to_pandas()
    df_t = df_t[["tweet", "label", "categories"]].dropna()
    df_t["label"] = df_t["label"].astype(int)

    le = LabelEncoder()
    df_t["categories_enc"] = le.fit_transform(df_t["categories"])

    train_texts_t, val_texts_t, train_labels_t, val_labels_t, train_cats_t, val_cats_t = train_test_split(
        df_t["tweet"].tolist(),
        df_t["label"].tolist(),
        df_t["categories_enc"].tolist(),
        test_size=0.2,
        random_state=42
    )

    vocab = build_vocab(train_texts_t)

    train_dataset_t = HateSpeechDataset(
        train_texts_t, train_labels_t, train_cats_t, vocab)
    val_dataset_t = HateSpeechDataset(
        val_texts_t, val_labels_t, val_cats_t, vocab)

    train_loader_t = DataLoader(
        train_dataset_t, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader_t = DataLoader(val_dataset_t, batch_size=32,
                              shuffle=False, collate_fn=collate_fn)

    model_t = TransformerClassifier(
        vocab_size=len(vocab),
        embed_size=128,
        num_layers=2,
        heads=4,
        dropout=0.1,
        forward_expansion=4,
        num_classes=len(le.classes_),
        device=device
    ).to(device)

    optimizer_t = torch.optim.Adam(model_t.parameters(), lr=3e-4)
    criterion_bin_t = nn.BCELoss()
    criterion_multi_t = nn.CrossEntropyLoss()

    for epoch in range(200):
        loss = train(model_t, train_loader_t, optimizer_t,
                     criterion_bin_t, criterion_multi_t)
        print(f"\rEpoch {epoch + 1}: loss = {loss:.4f}", end="")

    evaluate(model_t, val_loader_t)


if __name__ == "__main__":
    main()
