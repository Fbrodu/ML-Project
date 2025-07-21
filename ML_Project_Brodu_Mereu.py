"""
Fake News Classification Project
--------------------------------
This script performs binary classification of news articles as either "Fake" or "Real" using various machine learning models.
It includes the following steps:

1. Data loading from the Kaggle "Fake and Real News Dataset"
2. Text preprocessing (cleaning and normalization)
3. TF-IDF vectorization with unigrams and bigrams
4. Feature scaling using MinMaxScaler
5. Dimensionality reduction using PCA
6. Model training and hyperparameter tuning via GridSearchCV
   - Models: SVM (linear, RBF), k-NN, MLP
7. Cross-validation with ShuffleSplit (5 splits)
8. Evaluation using Accuracy, Precision, Recall, F1-score (macro and weighted)
9. Confusion Matrix plotting
10. Final t-SNE visualization and clustering using Gaussian Mixture Model (GMM)

Results are saved to:
- 'metrics_summary.csv'
- 'metrics_summary.txt'

Author: Francesca Brodu, Caterina Mereu
Date: July 2025
"""

# Text processing and feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import string
import re

# Import data manipulation lib
import pandas as pd
import numpy as np

# Import classifiers
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Model selection, evaluation and validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

# Import dimensionality reduction and visualization techniques
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

# Import plotting and evaluation libs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Load data from Kaggle dataset
fake = pd.read_csv("C:/Users/franc/OneDrive/Desktop/DATI/Fake.csv")
true = pd.read_csv("C:/Users/franc/OneDrive/Desktop/DATI/True.csv")

# Create labels for the classifier and combine the two dataframes
fake['label'] = 0
true['label'] = 1
dataset = pd.concat([true, fake], ignore_index=True)

# Preprocessing: Clean the text data by removing noise and standardizing format.
# - Converts all characters to lowercase to avoid case sensitivity issues (e.g., "The" and "the" are treated the same)
# - Removes any text within square brackets (e.g., citations like [1] or [source])
# - Removes all punctuation using Python's string.punctuation set
# - Removes words that contain digits
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Apply the cleaning function to both the 'title' and 'text' columns in the dataset.
dataset['title'] = dataset['title'].apply(clean_text)
dataset['text'] = dataset['text'].apply(clean_text)

# Define the cross-validation strategy using ShuffleSplit.
# This will create 5 random train/test splits each time using:
# - 70% of the data for training
# - 30% for testing
splitter = ShuffleSplit(n_splits=5, random_state=42, train_size=0.7)

# Define classifiers and hyperparameters
clf_list = [
    GridSearchCV(estimator=svm.SVC(kernel="linear"), param_grid={
        'C': [0.01, 0.1, 1, 10, 100]}, cv=3),
    GridSearchCV(estimator=svm.SVC(kernel="rbf"), param_grid={
        'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}, cv=3),
    GridSearchCV(estimator=KNeighborsClassifier(), param_grid={
        'n_neighbors': [3, 5, 7, 9]}, cv=3),
    GridSearchCV(estimator=MLPClassifier(max_iter=500, early_stopping=True, random_state=42), param_grid={
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001]
    }, cv=3)
]

clf_names = ['SVM - linear', 'SVM - RBF', 'kNN', 'MLP']

# Initialize array (for store accuracy) with zeros
acc = np.zeros((len(clf_list), splitter.get_n_splits()))

# Initialize lists to store all true and predicted labels for each classifier
labels_y_true = [[] for _ in clf_list]
labels_y_pred = [[] for _ in clf_list]

# Store best hyperparams for each classifier
best_params_first_fold = [None] * len(clf_list)

# At the end of the cross-validation loop, calculate average metrics for each classifier
results_list = []

# Assuming 'text', 'title' and 'label' are the relevant columns for x and y
x = (dataset['title'] + " " + dataset['text']).values
y = dataset['label'].values

# Loop on the splits: Start cross-validation
for i, (tr_idx, ts_idx) in enumerate(splitter.split(x, y)):
    x_tr = x[tr_idx]
    y_tr = y[tr_idx]
    x_ts = x[ts_idx]
    y_ts = y[ts_idx]

    # Convert the raw text data into numerical feature vectors using TF-IDF.
    # - stop_words='english': removes common English stop words (e.g., "the", "and")
    # - max_df=0.7: ignores terms that appear in more than 70% of documents (too frequent, likely not informative)
    # - min_df=5: ignores terms that appear in fewer than 5 documents (too rare)
    # - ngram_range=(1, 2): uses both unigrams and bigrams to capture more context
    # - max_features=2000: limits the number of features to the 2000 most informative terms
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5, ngram_range=(1, 2), max_features=2000)
    x_tr_vectorized = vectorizer.fit_transform(x_tr)
    x_ts_vectorized = vectorizer.transform(x_ts)

    # Normalize the TF-IDF features to the [0, 1] range using MinMaxScaler
    scaler = MinMaxScaler()
    x_tr_scaled = scaler.fit_transform(x_tr_vectorized.toarray())
    x_ts_scaled = scaler.transform(x_ts_vectorized.toarray())

    # Apply PCA to reduce dimensionality to 50
    pca = PCA(n_components=50, random_state=42)
    x_tr_pca = pca.fit_transform(x_tr_scaled)
    x_ts_pca = pca.transform(x_ts_scaled)

    # Classifier loop
    for k, clf in enumerate(clf_list):
        clf.fit(x_tr_pca, y_tr)

        y_pred = clf.predict(x_ts_pca)

        # Store accuracy for each classifier
        acc[k, i] = (y_pred == y_ts).mean()

        # Store all true and predicted labels for each classifier
        labels_y_true[k].extend(y_ts)
        labels_y_pred[k].extend(y_pred)

        # Store best params for each classifier
        best_params = tuple(sorted(clf.best_params_.items()))
        if i == 0:
            best_params_first_fold[k] = clf.cv_results_


# Print best hyperparam, accuracy and classification report for each classifier
for k, name in enumerate(clf_names):
    print(f"\n {name} ")

    cv_results = best_params_first_fold[k]  # cv_results_ from the first outer fold for classifier k
    best_index = np.argmax(cv_results['mean_test_score'])  # Index of the best hyperparameter setting
    best_params = cv_results['params'][best_index]  # Corresponding best hyperparameters
    print(f"Best hyperparams: {dict(best_params)}")
    print("    - Grid scores on development set:")
    means = cv_results['mean_test_score']
    stds = cv_results['std_test_score']
    for mean, std, params in zip(means, stds, cv_results['params']):
        print(f"        {mean:.3f} (+/-{std * 2:.03f}) for {params}")

    print(f"Mean test accuracy = {acc[k].mean():.2%} +/- {acc[k].std():.2%}")

    # Compute evaluation metrics for each class separately:
    # - Precision: how many predicted positives are actually correct
    # - Recall: how many actual positives are correctly predicted
    # - F1-score: harmonic mean of precision and recall
    # - Support: number of true samples for each class
    # These are computed for both classes: 0 (Fake) and 1 (Real).
    # The parameter zero_division=0 avoids warnings when division by zero occurs.
    precision, recall, f1, support = precision_recall_fscore_support(labels_y_true[k], labels_y_pred[k], labels=[0, 1],
                                                                     zero_division=0)

    # Calculate macro-average precision, recall, and F1-score
    # Macro average is the unweighted mean of the scores for each class
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()

    # Calculate weighted-average precision, recall, and F1-score
    # Weighted average accounts for the support (number of true instances) for each class
    total_support = support.sum()
    precision_weighted = np.sum(precision * support) / total_support
    recall_weighted = np.sum(recall * support) / total_support
    f1_weighted = np.sum(f1 * support) / total_support

    # Append the metrics to the results list
    results_list.append({
        "Classifier": name,
        "Best hyperparams": dict(best_params),
        "Accuracy Mean": acc[k].mean(),
        "Accuracy Std": acc[k].std(),
        "Precision Fake": precision[0],
        "Recall Fake": recall[0],
        "F1 Fake": f1[0],
        "Support Fake": support[0],
        "Precision Real": precision[1],
        "Recall Real": recall[1],
        "F1 Real": f1[1],
        "Support Real": support[1],
        "Precision Macro": precision_macro,
        "Recall Macro": recall_macro,
        "F1 Macro": f1_macro,
        "Precision Weighted": precision_weighted,
        "Recall Weighted": recall_weighted,
        "F1 Weighted": f1_weighted
    })

    # Compute and plot the Confusion Matrix for each classifier.
    # The confusion matrix shows the counts of:
    # - True Positives, True Negatives, False Positives and False Negatives
    # It helps visually assess how well the classifier distinguishes between the two classes (Fake vs Real).
    cm = confusion_matrix(labels_y_true[k], labels_y_pred[k], labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

# Create a final DataFrame from the list of results
df_results = pd.DataFrame(results_list)

# Save the metrics summary DataFrame to a CSV file
df_results.to_csv("metrics_summary.csv", index=False)

# Save the same summary to a text file
with open("metrics_summary.txt", "w", encoding="utf-8") as f:
    f.write(df_results.to_string(index=False))

# Vectorize + Scaler + PCA on the entire dataset
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5, ngram_range=(1, 2), max_features=2000)
x_vectorized = vectorizer.fit_transform(x)
x_scaled = MinMaxScaler().fit_transform(x_vectorized.toarray())
pca = PCA(n_components=50, random_state=42)
x_pca = pca.fit_transform(x_scaled)

# Apply t-SNE to reduce dimensionality to 2 components
tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(x_pca)

# Perform Gaussian Mixture clustering on the t-SNE output
clustering = GaussianMixture(n_components=2, random_state=42).fit(x_tsne)
labels = clustering.predict(x_tsne)

# Plot 2D
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

# This function creates a 2D t-SNE scatter plot
# Inputs:
# - ax: the subplot axis where the data will be drawn
# - x_tsne: the 2D t-SNE transformed data
# - labels: the class or cluster labels (used to group and color the points)
# - title: the title of the subplot
# - class_names: list of label names to show in the legend
# - colors: list of colors corresponding to each label
def plot_tsne(ax, x_tsne, labels, title, class_names, colors):
    for i, name in enumerate(class_names):
        ax.scatter(
            x_tsne[labels == i, 0],
            x_tsne[labels == i, 1],
            c=colors[i],
            label=name,
            s=5,
            alpha=0.5
        )

    ax.set_title(title, fontsize=16)

    custom_legend = [
        Line2D([0], [0], marker='o', color='w', label=class_names[i],
               markerfacecolor=colors[i], markersize=8, alpha=0.5)
        for i in range(len(class_names))
    ]
    ax.legend(handles=custom_legend, fontsize=12, loc='upper right', frameon=False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_facecolor('white')


# Subplot for true class labels
plot_tsne(
    ax=axes[0],
    x_tsne=x_tsne,
    labels=y,
    title="True Labels",
    class_names=["Fake", "Real"],
    colors=["#8064A2", "#4BACC6"]
)

# Subplot for GMM clusters
plot_tsne(
    ax=axes[1],
    x_tsne=x_tsne,
    labels=labels,
    title="GMM Clusters",
    class_names=["Cluster 0", "Cluster 1"],
    colors=["#4BACC6", "#8064A2"]
)

plt.tight_layout()
plt.show()




