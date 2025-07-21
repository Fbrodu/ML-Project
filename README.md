# Fake News Classification Project

This project aims to automatically classify news articles as *Fake* or *Real* using supervised machine learning techniques. 
It is based on the https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset available on Kaggle.

## Authors
- Francesca Brodu  
- Caterina Mereu  

## Date
July 2025

## Dataset
The dataset contains two CSV files:
- `Fake.csv` – articles labeled as fake
- `True.csv` – articles labeled as real

Each row includes:  
`title`, `text`, `subject`, `date`

We merged title and text for input features and created a binary target:
- 0 = Fake  
- 1 = Real

---

## Project Workflow

### 1. Preprocessing
- Lowercasing
- Removal of punctuation and numbers
- Concatenation of `title` and `text`

### 2. Feature Extraction
- TF-IDF vectorization (unigrams and bigrams)
- Feature scaling with MinMaxScaler

### 3. Dimensionality Reduction
- PCA for visualization and analysis
- t-SNE for 2D projection + GMM clustering

### 4. Classification Models
We used the following classifiers with **GridSearchCV** for hyperparameter tuning:
- **SVM** (linear kernel)
- **SVM** (RBF kernel)
- **k-NN**
- **MLP** (Multi-Layer Perceptron)

### 5. Evaluation
Models were evaluated using **ShuffleSplit cross-validation** (5 iterations). Metrics computed:
- Accuracy
- Precision (macro and weighted)
- Recall (macro and weighted)
- F1-score (macro and weighted)
- Confusion matrix

---

## Output Files
- `metrics_summary.csv` – best parameters and scores for each classifier
- `metrics_summary.txt` – the same file, but .txt
- Confusion matrix plots
- t-SNE visualization with GMM clustering

---
