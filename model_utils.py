import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL =================

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, questions):
        embedded = self.embedding(questions)
        _, (hidden, _) = self.lstm(embedded)
        return hidden[-1]


class Attention(nn.Module):
    def __init__(self, image_dim, question_dim, hidden_dim=512):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        self.attention_proj = nn.Linear(hidden_dim, 1)

    def forward(self, image_features, question_features):
        img_proj = self.image_proj(image_features)
        ques_proj = self.question_proj(question_features).unsqueeze(1)
        combined = torch.tanh(img_proj + ques_proj)
        scores = self.attention_proj(combined).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        attended = (image_features * weights.unsqueeze(-1)).sum(dim=1)
        return attended


class BUTD_VQA(nn.Module):
    def __init__(self, vocab_size, num_answers, image_dim=256):
        super().__init__()
        self.question_encoder = QuestionEncoder(vocab_size)
        self.attention = Attention(image_dim, 512)
        self.classifier = nn.Sequential(
    		nn.Linear(image_dim + 512, 1024),
    		nn.ReLU(),
    		nn.Dropout(0.5),
    		nn.Linear(1024, num_answers)
	)

    def forward(self, image_features, questions):
        question_features = self.question_encoder(questions)
        attended = self.attention(image_features, question_features)
        combined = torch.cat([attended, question_features], dim=1)
        logits = self.classifier(combined)
        return logits


# ================= LOAD MODEL ONCE =================

with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

with open("idx2answer.pkl", "rb") as f:
    idx2answer = pickle.load(f)

model = BUTD_VQA(len(word2idx), len(idx2answer)).to(device)
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

detector = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
detector.eval()
backbone = detector.backbone


def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = TF.to_tensor(image).to(device)

    with torch.no_grad():
        features_dict = backbone(image_tensor.unsqueeze(0))

    if isinstance(features_dict, dict):
        features_map = list(features_dict.values())[-1]
    else:
        features_map = features_dict

    pooled = torch.nn.functional.adaptive_avg_pool2d(features_map, (1,1))
    pooled = pooled.squeeze(-1).squeeze(-1)

    return pooled.unsqueeze(0)


def encode_question(question, max_len=14):
    question = question.lower()
    question = re.sub(r'[^\w\s\?]', '', question)
    tokens = question.split()

    encoded = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]

    if len(encoded) < max_len:
        encoded += [word2idx["<PAD>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    return torch.tensor(encoded).unsqueeze(0).to(device)


def predict(image_path, question):
    features = extract_features(image_path)
    question_tensor = encode_question(question)

    with torch.no_grad():
        logits = model(features, question_tensor)
        pred_idx = torch.argmax(logits, dim=1).item()

    return idx2answer[pred_idx]