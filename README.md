# Named Entity Recognition (NER) with CNN + BiLSTM

## Project Overview

This project implements a **Named Entity Recognition (NER) model** using a combination of **Convolutional Neural Networks (CNNs) and Bidirectional Long Short-Term Memory Networks (BiLSTMs)**. The model is trained on the **CoNLL-2003 dataset** and leverages **pre-trained GloVe word embeddings** for enhanced text representation. The goal is to classify words into entity categories such as **Person (PER), Location (LOC), Organization (ORG), and Miscellaneous (MISC)**.

## Dataset

I use the **CoNLL-2003 dataset**, a widely used benchmark dataset for NER. It consists of labeled sentences with four named entity categories.

- Dataset link: [CoNLL-2003 on Hugging Face](https://huggingface.co/datasets/conll2003)
- Features:
  - **Tokens**: Words in a sentence
  - **NER Tags**: Named entity labels
  - **POS Tags**: Part-of-speech tags
  - **Chunk Tags**: Syntactic chunking labels

## Pre-trained Word Embeddings

I use **GloVe (Global Vectors for Word Representation)** embeddings to convert words into dense vector representations.

- Download GloVe embeddings: [GloVe 6B Dataset](https://nlp.stanford.edu/projects/glove/)
- I use the **300-dimensional embeddings (glove.6B.300d.txt)**.

## Model Architecture

The model consists of three main components:

1. **Embedding Layer**
   - Loads pre-trained GloVe embeddings
   - Converts words into fixed-size dense vectors
2. **CNN Layer**
   - Extracts local features (e.g., prefixes/suffixes) from word embeddings
3. **BiLSTM Layer**
   - Captures long-term dependencies from both forward and backward contexts
4. **Fully Connected Layer**
   - Classifies words into NER categories

## Training & Evaluation

- **Loss function:** Cross-Entropy Loss
- **Optimizer:** Adam Optimizer
- **Batch size:** 32
- **Epochs:** 5
- **Evaluation Metrics:** Precision, Recall, F1-score (using `seqeval` library)

### Model Performance

| Entity Type | Precision | Recall   | F1-score | Support |
| ----------- | --------- | -------- | -------- | ------- |
| LOC         | 0.85      | 0.89     | 0.87     | 1668    |
| MISC        | 0.67      | 0.70     | 0.68     | 702     |
| ORG         | 0.73      | 0.78     | 0.76     | 1661    |
| PER         | 0.91      | 0.86     | 0.88     | 1617    |
| **Overall** | **0.81**  | **0.83** | **0.82** | 5648    |

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ner-cnn-bilstm.git
cd ner-cnn-bilstm
```

### 2. Install Dependencies
Ensure you have Python 3.8+ and the following libraries installed:
```bash
pip install torch torchvision torchaudio
pip install datasets transformers
pip install numpy pandas tqdm
pip install seqeval
pip install matplotlib
```

### 3. Download Pre-trained GloVe Embeddings

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove/
```

### 4. Run Training Script

```bash
python train.py
```

### 5. Evaluate the Model

```bash
python evaluate.py
```
