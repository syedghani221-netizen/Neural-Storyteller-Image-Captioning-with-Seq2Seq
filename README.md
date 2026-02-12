## 🖼️ Neural Storyteller - Image Captioning with Deep Learning
An end-to-end deep learning system that generates natural language descriptions for images using CNN-LSTM architecture trained on the Flickr30k dataset.

Kaggle Notebook: https://www.kaggle.com/code/ghanishah/neural-storyteller-image-captioning-with-seq2seq


HuggingFace: https://huggingface.co/spaces/syedghani/neural-storyteller

An end-to-end Generative AI pipeline that "sees" an image and "tells" its story. This project implements a **CNN-RNN Encoder-Decoder architecture** trained on the **Flickr30k** dataset to automate image description.

---
## What is Neural Storyteller?
Neural Storyteller is a Generative AI pipeline that bridges the gap between computer vision and natural language processing. Upload an image, and the AI "sees" its content and "tells" you a descriptive story in natural language.
Built on a CNN-RNN Encoder-Decoder architecture and trained on the Flickr30k dataset (31,783 images with human-written captions), this project demonstrates state-of-the-art image captioning techniques.
✨ Key Highlights

🎨 Automatic Image Understanding — Extracts semantic features from any image
📝 Natural Language Generation — Produces human-like descriptive captions
🚀 Production-Ready Deployment — Live web interface on HuggingFace Spaces
📊 Comprehensive Evaluation — Validated with BLEU metrics and visual analysis
⚡ GPU-Optimized Training — Efficient feature extraction and model training


## 🏗️ Architecture Overview

The model bridges Computer Vision and Natural Language Processing using a two-stage pipeline:

1.  **Encoder (Vision):** A pre-trained **ResNet50** CNN (ImageNet weights) extracts spatial features. The final pooling layer outputs a 2048-dimensional vector representing the image's "essence."
2.  **Decoder (Language):** An **LSTM-based** Recurrent Neural Network. It takes the image features as its initial state and predicts the next word in the sequence using word embeddings and greedy search.

---
## 📈 Performance & Deliverables

### 1. Training Convergence (Loss Curve)
The model was optimized using **Cross-Entropy Loss** and the **Adam Optimizer** over 5 epochs. 

| Epoch | Loss |
| :--- | :--- |
| 1 | 3.9365 |
| 2 | 3.1909 |
| 3 | 2.9069 |
| 4 | 2.7046 |
| 5 | 2.5412 |

> **Note:** See the full loss plot in the `./outputs` folder for training vs. validation trends.

### 2. Quantitative Evaluation
| Metric | Result |
| :--- | :--- |
| **Dataset Size** | 31,783 Images |
| **Vocabulary Size** | 20,013 Tokens |
| **Max Seq Length** | 80 Tokens |
| **Evaluation** | BLEU-4 Score (NLTK) |

---

## 🖼️ Neural Storytelling Samples
*Real output from the model's test phase:*

| Image | Ground Truth | Model Prediction |
| :---: | :--- | :--- |
| Sample 1 | Two young guys look at their phones. | `<start>` two young men are standing outside `<end>` |
| Sample 2 | A dog running in the grass. | `<start>` a dog runs through the green grass `<end>` |

---

## 🚀 Deployment & Usage

### Running the App
The model is deployed via a **Gradio** web interface. To launch it locally:

1. **Install Dependencies:**
   ```bash
   pip install torch torchvision gradio pillow nltk pandas
2. **Execute:**

   Bash
   python app.py
   Project Structure
   Neural Storyteller.ipynb: The complete training and evaluation pipeline.
   
   app.py: Gradio interface for real-time inference.
   
   flickr30k_features.pkl: Pre-extracted image features (Global average pooling).

### Author: Syed Ghani
