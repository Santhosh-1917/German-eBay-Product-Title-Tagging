


# 🧠 German eBay Product Title Tagging (NER using BiLSTM-CRF)

This project performs **Named Entity Recognition (NER)** on German eBay product titles.  
Each token in the product title is tagged with an appropriate aspect (e.g., *Hersteller*, *Farbe*, *Produktart*).  
The model learns to automatically label unseen product titles for structured extraction.

---

## 📂 Project Overview

This repository implements a **BiLSTM-CRF** model for token-level sequence tagging across two product categories.  
It also supports optional integration of **FastText German word embeddings** for improved semantic understanding.

### 🔧 Key Steps
1. **Data Preprocessing**
   - Loads and cleans the training dataset (`Tagged_Titles_Train.tsv`)
   - Replaces missing tags with most frequent tag per token (or `"O"`)
   - Splits by category and groups by record number

2. **Vocabulary & Encoding**
   - Builds token and tag mappings (`word2idx`, `tag2idx`)
   - Pads sentences and creates attention masks

3. **Model Architecture**
   - Embedding Layer (random or FastText)
   - Bidirectional LSTM
   - Linear projection → tag space
   - Conditional Random Field (CRF) for structured decoding

4. **Training & Evaluation**
   - Early stopping, learning-rate scheduling, and F1 tracking
   - Macro F1-score, accuracy, and classification report per category

---

## 🧩 Model Architecture

Input → Embedding (100–300d) → BiLSTM → Linear Layer → CRF → Tag Sequence

- **Embedding**: Random or pretrained (FastText German)  
- **Hidden Dim**: 128–256  
- **Optimizer**: Adam (lr = 1e-3)  
- **Regularization**: Dropout 0.3  
- **Loss Function**: Negative log-likelihood from CRF  
- **Metrics**: Macro F1-score, Accuracy  

---

## 📊 Results

| Category | Model Type | Validation F1 | Accuracy |
|-----------|-------------|---------------|-----------|
| Category 1 | BiLSTM-CRF | **0.78** | 0.93 |
| Category 2 | BiLSTM-CRF | **0.55** | 0.90 |

*(Expected improvement to ~0.82 F1 with FastText embeddings)*

---

## 🛠️ Installation

```bash
git clone https://github.com/<your-username>/German-eBay-Product-Title-Tagging.git
cd German-eBay-Product-Title-Tagging
pip install -r requirements.txt

Dependencies

torch
torchcrf
pandas
numpy
matplotlib
scikit-learn
gensim


⸻

##🚀 Training

Option 1 — Baseline (Random embeddings)

model_cat1, best_f1_cat1 = train_one_category(...)
model_cat2, best_f1_cat2 = train_one_category(...)

Option 2 — With FastText German embeddings

Download FastText German vectors:

cc.de.300.vec.gz

Then run:

cat1_emb_matrix = build_embedding_matrix(cat1_word2idx, ft_model)
model_cat1, best_f1_cat1 = train_one_category(..., pretrained_matrix=cat1_emb_matrix)


⸻

📁 Dataset Description

Column	Description
Record Number	Unique ID per product title
Category	Product group (1 or 2)
Token	Word in the title
Tag	Labeled aspect (e.g. Hersteller, Farbe, Größe)


⸻

💾 Outputs
	•	bilstm_crf_cat1.pt — Trained model weights for Category 1
	•	bilstm_crf_cat2.pt — Trained model weights for Category 2
	•	submission.tsv — Predicted tags for unseen test titles

⸻

🔮 Future Work
	•	Integrate FastText embeddings for semantic boost
	•	Experiment with DistilBERT-German for contextual embeddings
	•	Fine-tune learning rate and dropout
	•	Add cross-category ensemble for higher robustness

⸻

🧰 Tech Stack


⸻

👨‍💻 Author

Santhosh Narayanan Baburaman
🎓 M.S. Analytics @ University of Southern California
📧 [your-email@example.com]
🔗 LinkedIn | GitHub

⸻


---

