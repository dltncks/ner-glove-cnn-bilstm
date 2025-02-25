# Named Entity Recognition using CNN + BiLSTM and GloVe Embeddings

## 1. Introduction
Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that involves identifying and classifying named entities such as persons, organizations, locations, and miscellaneous entities in a given text. In this project, I implemented a CNN + BiLSTM-based model for NER, leveraging pre-trained GloVe embeddings to enhance word representation.

## 2. Dataset Curation
### 2.1 CoNLL-2003 Dataset
The dataset used for this project is the **CoNLL-2003** dataset, which is widely recognized for NER tasks. The dataset consists of text sequences labeled with four entity types:
- **PER**: Person names
- **ORG**: Organizations
- **LOC**: Locations
- **MISC**: Miscellaneous entities (e.g., nationalities, events)

#### Dataset Statistics:
| Split       | Number of Sentences |
|------------|------------------|
| Training   | 14,041           |
| Validation | 3,250            |
| Test       | 3,453            |

The dataset is structured with tokens (`tokens`), POS tags (`pos_tags`), chunking tags (`chunk_tags`), and NER labels (`ner_tags`). However, I only used `tokens` and `ner_tags` for this project.

**Dataset Source:** [Hugging Face - CoNLL-2003 Dataset](https://huggingface.co/datasets/conll2003)

## 3. Word Embeddings and Model Architecture
### 3.1 Word Embeddings
To improve model performance, I utilized **pre-trained GloVe embeddings (300D)** from Stanford NLP.
- Source: `glove.6B.300d.txt`
- Vocabulary Size: 400,000
- Embedding Dimension: 300
- **Reference:** [Stanford NLP - GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)

Each word in the dataset is mapped to a corresponding vector representation from the GloVe embeddings. Additionally, I introduced a **padding token (`<PAD>`)** with an all-zero vector to handle variable-length sequences.

### 3.2 Model Architecture: CNN + BiLSTM
I designed a **hybrid CNN + BiLSTM model** that effectively captures both local and long-range dependencies in text sequences. The model consists of the following layers:

1. **Embedding Layer**: Maps word indices to 300D GloVe embeddings.
2. **1D CNN Layer**: Extracts local features from word embeddings.
3. **Bidirectional LSTM (BiLSTM)**: Captures context dependencies from both past and future words.
4. **Fully Connected Layer**: Maps LSTM outputs to named entity labels.
5. **Dropout (0.5)**: Prevents overfitting.
6. **Softmax Activation**: Outputs probability distribution for each word’s NER label.

### 3.3 Training Process
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam with learning rate = 0.001
- **Batch Size**: 32
- **Padding Strategy**: Dynamic padding using `pad_sequence()`
- **Training Duration**: 5 epochs

## 4. Results and Performance Evaluation
The model was trained for **5 epochs**, achieving a consistently decreasing loss:

#### Training Loss per Epoch:
| Epoch | Loss |
|-------|------|
| 1     | 0.1406 |
| 2     | 0.0399 |
| 3     | 0.0284 |
| 4     | 0.0199 |
| 5     | 0.0152 |

### **Evaluation on Test Set**
I evaluated the model using **precision, recall, and F1-score**, computed with `seqeval.metrics.classification_report`.

#### **Classification Report:**
| Entity | Precision | Recall | F1-Score | Support |
|--------|----------|--------|----------|---------|
| LOC    | 0.85     | 0.89   | 0.87     | 1668    |
| MISC   | 0.67     | 0.70   | 0.68     | 702     |
| ORG    | 0.73     | 0.78   | 0.76     | 1661    |
| PER    | 0.91     | 0.86   | 0.88     | 1617    |
| **Overall** | **0.81** | **0.83** | **0.82** | **5648** |

## 5. Analysis and Experiments
### 5.1 Impact of CNN Layer
Adding a **CNN layer** before BiLSTM improved the model’s ability to recognize local patterns within words, enhancing its accuracy.

### 5.2 Impact of Pre-trained GloVe Embeddings
Using **pre-trained GloVe embeddings** significantly boosted model performance compared to random initialization. Without GloVe, F1-scores were **~5% lower**.

### 5.3 Hyperparameter Tuning
I experimented with:
- **Batch sizes (16, 32, 64)** → Best performance with **32**.
- **Learning rates (0.0001, 0.001, 0.01)** → **0.001** worked best.
- **LSTM Hidden Units (128, 256, 512)** → Best result with **256 units**.

## 6. Lessons Learned & Future Work
### 6.1 Lessons Learned
- **Combining CNN + BiLSTM is effective** for NER tasks, as CNN extracts local features while BiLSTM captures context dependencies.
- **Pre-trained embeddings like GloVe significantly enhance performance.**
- **Dynamic padding ensures efficient batch processing.**

### 6.2 Future Improvements
- Implement **CRF (Conditional Random Fields)** for better label dependency handling.
- Train with **larger datasets** (e.g., OntoNotes 5.0) to improve generalization.
- Experiment with **character-level CNNs** to capture subword-level information.

## 7. Conclusion
This project successfully implemented a **CNN + BiLSTM model** for Named Entity Recognition using the **CoNLL-2003 dataset** and **GloVe word embeddings**. The model achieved an overall **F1-score of 82%**, demonstrating the effectiveness of combining convolutional and recurrent layers in NER tasks. Further improvements could be made by integrating CRFs and training on larger datasets.
