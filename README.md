# Vietnamese Information Authentication Problem

## 1. Introduction
In today’s digital era, the rapid spread of online information — especially through social media and digital platforms — has created a pressing need for **automated fact-checking systems**.  
While English has numerous resources and datasets for fact verification (e.g., FEVER, LIAR, PolitiFact), Vietnamese remains a **low-resource language** in this field.

This project addresses the **Vietnamese fact verification task** through two main subtasks:
1. **Evidence Retrieval** – Identifying and extracting the most relevant evidence sentences from the provided context.
2. **Claim Classification** – Predicting whether a given claim is:
   - **Supports** – Evidence supports the claim
   - **Refutes** – Evidence contradicts the claim
   - **Not Enough Information (NEI)** – Evidence is insufficient to make a conclusion

We experiment with both **traditional machine learning models** and **modern pretrained language models (PLMs)** to establish a benchmark for this task.

---

## 2. Datasets

### **ViWikiFC**
- **Source:** Extracted from Vietnamese Wikipedia articles.
- **Composition:** Each sample contains a claim and an evidence sentence from the same article.
- **Labels:** `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`.
- **Size:** ~21,000 samples.
- **Characteristics:** Well-structured, balanced label distribution.

### **ViFactCheck**
- **Source:** Verified claims from the FactCheckVN organization, collected from social media, political speeches, and news media.
- **Composition:** Each claim is paired with an evidence passage from reliable news sources.
- **Labels:** `SUPPORTS` (0), `REFUTES` (1), `NOT ENOUGH INFO` (2).
- **Size:** ~7,000 samples.
- **Characteristics:** Imbalanced label distribution, noisier real-world data.

---

## 3. Methodology

### 3.1 Evidence Retrieval
We evaluate three retrieval strategies:
- **BM25** – Lexical matching using term frequency and inverse document frequency.
- **SBERT** – Semantic similarity using sentence embeddings and cosine similarity.
- **Hybrid (BM25 + SBERT)** – Weighted combination of BM25 (α = 0.7) and SBERT scores.

### 3.2 Claim Classification
We compare six classification models:
- **Traditional ML:** Support Vector Machine (SVM), Logistic Regression (LR) using TF-IDF features.
- **Deep Learning:** Bidirectional LSTM (BiLSTM).
- **Pretrained Language Models:**
  - **PhoBERT** – BERT-based model trained specifically for Vietnamese.
  - **XLM-RoBERTa** – Multilingual transformer model supporting Vietnamese.
  - **BARTPho** – Encoder-decoder model tailored for Vietnamese.

**Evaluation Metric:** Macro F1-score for each label and dataset.

---

## 4. Results

| Model          | ViWikiFC (Supports) | ViWikiFC (Refutes) | ViWikiFC (NEI) | ViFactCheck (Supports) | ViFactCheck (Refutes) | ViFactCheck (NEI) |
|----------------|---------------------|--------------------|----------------|------------------------|-----------------------|-------------------|
| **SVM**        | 0.37                | 0.45               | 0.44           | 0.41                   | 0.30                  | 0.37              |
| **LR**         | 0.38                | 0.45               | 0.45           | 0.51                   | 0.42                  | 0.63              |
| **BiLSTM**     | 0.45                | 0.44               | 0.55           | 0.58                   | 0.47                  | 0.65              |
| **BARTPho**    | 0.64                | 0.69               | 0.74           | 0.59                   | 0.55                  | 0.61              |
| **XLM-RoBERTa**| 0.69                | 0.67               | 0.76           | **0.72**                | 0.52                  | **0.70**          |
| **PhoBERT**    | **0.71**            | **0.71**           | **0.75**       | 0.59                   | **0.56**              | 0.59              |

**Key Insights:**
- **PhoBERT** achieves the best performance on the clean, structured **ViWikiFC** dataset.
- **XLM-RoBERTa** outperforms all models on the noisier **ViFactCheck** dataset.
- Traditional models (SVM, LR) lag significantly behind PLMs, especially on the Refutes class.

---

## 5. Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/Vietnamese-Information-Authentication-Problem.git
cd Vietnamese-Information-Authentication-Problem

# Install dependencies
pip install -r requirements.txt

# Run PhoBERT + BM25 pipeline
python src/ie_403_fi3.py
