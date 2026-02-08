import gradio as gr
import torch
import torch.nn as nn
import pickle
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

device = torch.device("cpu")

# ----------------------
# Load vocabulary
# ----------------------

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("rev_vocab.pkl", "rb") as f:
    rev_vocab = pickle.load(f)

vocab_size = len(vocab)

# ----------------------
# Model Architecture
# ----------------------

class Encoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512):
        super().__init__()
        self.linear = nn.Linear(feature_dim, hidden_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.linear(x))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.fc(out)
        return out, hidden

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size)

model = ImageCaptioningModel(vocab_size)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ----------------------
# Load ResNet
# ----------------------

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

# ----------------------
# Caption Generation
# ----------------------

def generate_caption(feature, max_length=40):
    feature = feature.unsqueeze(0)
    
    hidden_state = model.encoder(feature)
    hidden = (
        hidden_state.unsqueeze(0),
        torch.zeros_like(hidden_state.unsqueeze(0))
    )
    
    input_word = torch.tensor([[vocab["<start>"]]])
    generated = []
    
    for _ in range(max_length):
        embeddings = model.decoder.embedding(input_word)
        output, hidden = model.decoder.lstm(embeddings, hidden)
        output = model.decoder.fc(output.squeeze(1))
        
        predicted = output.argmax(1)
        word = rev_vocab[predicted.item()]
        
        if word == "<end>":
            break
            
        generated.append(word)
        input_word = predicted.unsqueeze(0)
    
    return " ".join(generated)

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        feature = resnet(image).view(1, -1)
        caption = generate_caption(feature.squeeze(0))
    return caption

# ----------------------
# Gradio UI
# ----------------------

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Neural Storyteller - Image Captioning",
    description="Upload an image and the model will generate a caption."
)

demo.launch()