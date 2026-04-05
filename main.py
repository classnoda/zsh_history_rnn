import torch
from nltk import RegexpTokenizer
from torch import nn
import json

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42

with open("metadata/model_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

vocab = metadata["vocab"]
context_len = metadata["context_len"]
inv_vocab = {int(v): k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)
print(f"Словарь загружен. Размер: {VOCAB_SIZE}")


tokenizer = RegexpTokenizer(r"\w+|[^\w\s]")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CommandsPrediction(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        logits = self.fc(out[:, -1, :])
        return logits

model = CommandsPrediction(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM).to(device)


weights = torch.load('metadata/model_weights.pth')
model.load_state_dict(weights)

model.eval()

def predict_next(text, top_k = 3):
    model.eval()
    with torch.no_grad():
        encoded = [vocab.get(t, vocab["<UNK>"]) for t in tokenizer.tokenize(text.lower())]

        encoded = encoded[-context_len:]
        padded = [0] * (context_len - len(encoded)) + encoded
        input_tensor = torch.tensor([padded], dtype=torch.long, device=device)
        probs = torch.softmax(model(input_tensor), dim=1)
        top_probs, top_ids = torch.topk(probs, top_k)
        return [
            (inv_vocab.get(top_ids[0][i].item(), "<UNK>"), top_probs[0][i].item())
            for i in range(top_k)
        ]


text = input("Введите начало команды: ").strip()

predictions = predict_next(text)

print(f"\nВарианты продолжения «{text}»:")
for i, (word, prob) in enumerate(predictions, 1):
    print(f"  {i}. {word}  (уверенность: {prob:.2%})")
print("─" * 30)

