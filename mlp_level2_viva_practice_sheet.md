# 🎯 MLP Level 2 Viva — ULTIMATE Practice Sheet

> [!IMPORTANT]
> This is your **complete, line-by-line** preparation guide. The viva is **30-60 minutes** long. Your goal: explain your notebook like a story (20-30 min), then handle theory + coding questions (10-20 min). **Know every single word, parameter, and function** in your code.

---

## 📋 Table of Contents

1. [Problem Statement & Dataset](#1-problem-statement--dataset)
2. [Notebook Walkthrough (Cell-by-Cell)](#2-notebook-walkthrough-cell-by-cell)
3. [Theory Questions Bank (Sorted by Frequency)](#3-theory-questions-bank)
4. [Live Coding Questions Bank](#4-live-coding-questions-bank)
5. [Proctor Tips & Strategies](#5-proctor-tips--strategies)

---

## 1. Problem Statement & Dataset

### What is the problem?
- **Comment Category Prediction Challenge** — a **multi-class classification** problem
- **Goal**: Predict the `label` (0, 1, 2, or 3) of a comment
- **Metric**: **F1-macro** — treats all classes equally, critical for imbalanced datasets
- It's a 4-class classification problem on a social media comment dataset

### Why F1-macro and NOT accuracy?
- The dataset is **heavily imbalanced**:
  - Label 0: 114,173 (57.7%)
  - Label 1: 15,918 (8.0%)
  - Label 2: 62,440 (31.5%)
  - Label 3: 5,469 (2.8%)
- Accuracy would be misleading — a model that always predicts "0" would get ~57.7% accuracy but score terribly on F1-macro
- F1-macro computes F1 per class and takes the **unweighted mean**, so all classes matter equally
- **F1 = 2 × (Precision × Recall) / (Precision + Recall)**
- **Precision** = TP / (TP + FP) — "of things I predicted as X, how many actually are X?"
- **Recall** = TP / (TP + FN) — "of actual X's, how many did I catch?"

### Dataset Overview
| Feature | Type | Description |
|---------|------|-------------|
| `created_date` | datetime | When the comment was posted |
| `post_id` | int (52 unique) | Which post this comment belongs to |
| `emoticon_1/2/3` | int | Emoticon indicators |
| `upvote` | int | Number of upvotes |
| `downvote` | int | Number of downvotes |
| `if_1` | int (57 unique) | Interaction feature 1 |
| `if_2` | int (81 unique) | Interaction feature 2 |
| `race` | categorical | white, black, asian, latino, other, none (73% missing → NaN) |
| `religion` | categorical | christian, muslim, jewish, etc. (73% missing → NaN) |
| `gender` | categorical | female, male, transgender, other, none (73% missing → NaN) |
| `disability` | bool | True/False |
| `comment` | text | The actual comment text |
| `label` | int (0-3) | **TARGET** variable |

### Key Observations from EDA
- **~73% of race/religion/gender are NaN** — these are NOT randomly missing, they indicate the comment does NOT mention those attributes
- **NaN is filled with 'missing'** — treating it as its own category
- **Label 3 (most severe) is the rarest** — only 2.8% of data
- **Comment length varies by label**: Label 3 comments tend to be shorter (mean ~194 chars vs ~296-336 for others)
- **Cross-tabs show strong correlations**: e.g., comments mentioning specific races/religions heavily skew toward certain labels

---

## 2. Notebook Walkthrough (Cell-by-Cell)

### Cell 1: Imports
```python
import os, pandas as pd, numpy as np, re, time, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import lightgbm as lgb
import xgboost as xgb
from scipy.sparse import issparse
```

**Why these libraries?**
- **sklearn**: Provides all ML models, preprocessing, pipelines, metrics
- **lightgbm / xgboost**: Gradient boosting frameworks — faster and more powerful than sklearn's GradientBoosting
- **TfidfVectorizer**: Converts text to numerical features
- **ColumnTransformer**: Applies different transformations to different column types
- **Pipeline**: Chains preprocessing + model into a single workflow (avoids data leakage)
- **StratifiedKFold**: Ensures each fold has the same class distribution as the full dataset

> [!TIP]
> **What is `StratifiedKFold`?** — It splits data into K folds while maintaining the class distribution. Critical for imbalanced datasets. Without stratification, a fold might have very few samples of the minority class.

---

### Cell 2: Data Loading + EDA

```python
DATA_DIR = "/kaggle/input/comment-category-prediction-challenge"
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_raw  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "Sample.csv"))
```

**Key outputs:**
- Train: (198,000 rows × 15 cols), Test: (102,000 rows × 14 cols)
- Missing values: `race`, `religion`, `gender` all have ~145,423 NaN in train
- `comment` has 1 NaN in train
- **Train has `label` column, test does NOT** (we need to predict it)

**EDA Questions You Must Answer:**
- Q: "What did you learn from EDA?"
  - The dataset is imbalanced (label 0 dominant, label 3 very rare)
  - Comments that mention race/religion/gender tend to be labeled differently than those that don't
  - Comment length varies by class — shorter comments tend to be more toxic (label 3)
  - Certain post_ids have very different label distributions

---

### Cell 3: Visualization
- Bar chart of label distribution
- Histogram of comment character lengths
- Post ID vs Label stacked bar
- Race, Religion distributions (showing NaN as "missing")
- Disability vs Label crosstab

**Why these plots?**
- To understand **class imbalance** → drives choice of F1-macro and class_weight='balanced'
- To understand if **text features** carry signal → yes, comment length differs by class
- To understand if **structured features** carry signal → yes, certain post_ids and demographic features correlate with labels

---

### Cell 4: Feature Engineering

```python
def engineer_features(df):
    df = df.copy()
    df['comment'] = df['comment'].fillna('')
    # Time features
    df['hour'] = pd.to_datetime(df['created_date']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['created_date']).dt.dayofweek
    # Text features
    df['comment_len'] = df['comment'].apply(len)
    df['word_count'] = df['comment'].apply(lambda x: len(x.split()))
    df['avg_word_len'] = df['comment_len'] / (df['word_count'] + 1)
    df['uppercase_ratio'] = df['comment'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    df['exclamation_count'] = df['comment'].apply(lambda x: x.count('!'))
    df['question_count'] = df['comment'].apply(lambda x: x.count('?'))
    # Interaction feature
    df['if_interact'] = df['if_1'] * df['if_2']
    # Fill NaN for categorical
    for col in ['race', 'religion', 'gender']:
        df[col] = df[col].fillna('missing')
    return df
```

**Why each feature?**
| Feature | Rationale |
|---------|-----------|
| `hour`, `dayofweek` | Comments posted at different times may have different toxicity levels |
| `comment_len` | EDA showed label 3 has shorter comments on average |
| `word_count` | Another proxy for comment length |
| `avg_word_len` | Longer words may indicate more sophisticated/less toxic language |
| `uppercase_ratio` | ALL CAPS often indicates aggressive/toxic comments |
| `exclamation_count` | More exclamation marks → more emotional/aggressive |
| `question_count` | Questions may indicate genuine engagement (less toxic) |
| `if_interact` | Captures interaction between `if_1` and `if_2` — multiplicative combination captures non-linear relationships |

> [!NOTE]
> **Why `if_1 * if_2`?** — This creates an **interaction feature**. If `if_1=0` AND `if_2≠0`, the interaction is 0. This captures combinations that individual features can't. The idea: two features may individually be weak signals, but their combination may be strong.

---

### Cell 5-8: Milestone 1-4 Models (Exploration Phase)

This covers:

#### **Preprocessing Pipeline (ColumnTransformer)**
```python
preprocessor = ColumnTransformer([
    ('tfidf', Pipeline([
        ('vec', TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)),
        ('svd', TruncatedSVD(n_components=200, random_state=42))
    ]), 'comment'),
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
])
```

**What is each component?**

| Component | What it does | Why |
|-----------|-------------|-----|
| `TfidfVectorizer` | Converts text → numerical matrix based on term frequency-inverse document frequency | Better than bag-of-words because it down-weights common words |
| `max_features=10000` | Keep only top 10K features | Reduces dimensionality, prevents overfitting |
| `ngram_range=(1,2)` | Use both unigrams and bigrams | "not good" as a bigram means something different from "not" + "good" separately |
| `sublinear_tf=True` | Replace tf with 1 + log(tf) | Diminishing returns for term frequency — 10 occurrences ≠ 10× more important than 1 |
| `TruncatedSVD(n_components=200)` | Reduce TF-IDF matrix from 10K to 200 dimensions | Reduces noise, computation — similar to PCA but for sparse matrices |
| `StandardScaler` | Zero mean, unit variance for numerical features | Many models (LR, SVM, KNN, MLP) are sensitive to feature scale |
| `OneHotEncoder` | Converts categorical → binary columns | ML models need numerical input |
| `handle_unknown='ignore'` | Unknown categories in test set → all zeros | Prevents crashes if test has categories not seen in training |

> [!IMPORTANT]
> **Why TruncatedSVD instead of PCA?**
> - PCA requires dense input, TF-IDF produces sparse matrices
> - TruncatedSVD works directly on sparse matrices — much more memory efficient
> - Both do dimensionality reduction, but SVD is preferred for NLP/sparse data

> [!NOTE]
> **What is TF-IDF?**
> - TF (Term Frequency): How often a word appears in a document
> - IDF (Inverse Document Frequency): log(N / n_docs_containing_term) — rare words get higher weight
> - TF-IDF = TF × IDF
> - Example: "the" appears everywhere → low IDF → low TF-IDF. "misogynistic" appears rarely → high IDF → high TF-IDF if it appears in a document

#### **Models Explored:**

| Model | Type | Key Parameters | Why Used |
|-------|------|---------------|----------|
| **LogisticRegression** | Linear, parametric | C=1, solver='lbfgs', max_iter=1000 | Strong baseline for text classification |
| **SGDClassifier** | Linear, parametric | loss='log_loss', alpha=1e-4 | Scales well to large datasets, stochastic gradient descent |
| **GaussianNB** | Probabilistic | — | Fast, assumes feature independence (naive assumption) |
| **KNeighborsClassifier** | Non-parametric | n_neighbors=5 | No training needed, but slow at prediction |
| **LinearSVC** | SVM, linear | C=1 | Good for high-dimensional text data |
| **RandomForestClassifier** | Bagging ensemble | n_estimators=100 | Reduces variance, handles non-linear relationships |
| **GradientBoostingClassifier** | Boosting ensemble | n_estimators=100 | Reduces bias sequentially |
| **LightGBM** | Boosting (leaf-wise) | n_estimators=500 | Fastest boosting, handles large datasets |
| **XGBoost** | Boosting (level-wise) | n_estimators=500 | Regularized boosting, handles imbalanced data |
| **MLPClassifier** | Neural network | hidden_layer_sizes=(256,128,64) | Can learn complex non-linear patterns |

---

### Cell 9-10: Final Optimized Pipeline (Milestone 5)

This is the **final submission pipeline** — the most important part.

```python
# Feature columns
ENGINEERED_NUM = ['hour', 'dayofweek', 'comment_len', 'word_count', 'avg_word_len',
                  'uppercase_ratio', 'exclamation_count', 'question_count']
RAW_NUM = ['emoticon_1', 'emoticon_2', 'emoticon_3', 'upvote', 'downvote']
BINARY = ['disability_int', 'if_interact']
CATEGORICAL = ['race', 'religion', 'gender']
```

```python
# TF-IDF: Word-level + Character-level
word_vec_f = TfidfVectorizer(
    max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.90,
    sublinear_tf=True, strip_accents='unicode', norm='l2',
    token_pattern=r'(?u)\b\w\w+\b')

char_vec_f = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 5), max_features=10000,
    min_df=2, sublinear_tf=True, norm='l2')
```

**Why both word and character TF-IDF?**
- **Word TF-IDF** captures semantic meaning ("hate", "stupid")
- **Character TF-IDF** captures patterns that survive misspelling ("h8", "st00pid"), prefixes ("un-", "dis-"), and character-level patterns
- **Combined = 30K features** (20K word + 10K char)

**Key TF-IDF Parameters:**
| Parameter | Value | Why |
|-----------|-------|-----|
| `max_features` | 20000 (word), 10000 (char) | Limit vocabulary size |
| `ngram_range` | (1,2) word, (3,5) char | Bigrams for word, 3-5 char n-grams for patterns |
| `min_df=2` | Minimum 2 documents | Remove extremely rare terms (noise) |
| `max_df=0.90` | Maximum 90% of documents | Remove very common terms (stop words effect) |
| `sublinear_tf=True` | Apply log to TF | Diminishing returns on frequency |
| `strip_accents='unicode'` | Remove accents | Normalize text |
| `norm='l2'` | L2 normalization per row | Each document vector has unit length — prevents long documents from dominating |
| `analyzer='char_wb'` | Character n-grams within word boundaries | "cat" → "ca", "at", "cat" instead of crossing word boundaries |

#### **Feature Assembly:**
```python
X_full_sp = sp_hstack([X_full_w, X_full_c, X_full_s, X_full_ohe]).tocsr()
```
- `hstack` combines: word TF-IDF + char TF-IDF + numerical features + one-hot encoded categoricals
- `tocsr()` converts to Compressed Sparse Row format — efficient for sparse data

#### **LightGBM Configuration:**
```python
lgb_params = dict(
    n_estimators=1200,
    learning_rate=0.10,
    num_leaves=31,
    max_depth=8,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.3,
    reg_alpha=0.1,
    reg_lambda=0.5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
```

| Parameter | Value | Why |
|-----------|-------|-----|
| `n_estimators=1200` | Max 1200 trees | More trees = better, but early stopping prevents overfitting |
| `learning_rate=0.10` | Step size per tree | Lower = more precise but needs more trees. 0.10 is a good balance |
| `num_leaves=31` | Max leaves per tree | Controls tree complexity. LightGBM uses leaf-wise growth |
| `max_depth=8` | Max tree depth | Prevents overly complex trees |
| `min_child_samples=50` | Min samples per leaf | Regularization — prevents learning from tiny groups |
| `subsample=0.8` | 80% of data per tree | Random subsampling reduces overfitting (bagging within boosting) |
| `colsample_bytree=0.3` | 30% of features per tree | Feature randomization reduces overfitting |
| `reg_alpha=0.1` | L1 regularization | Encourages sparse trees (feature selection) |
| `reg_lambda=0.5` | L2 regularization | Smooths predictions, prevents extreme weights |
| `class_weight="balanced"` | Auto-weight classes inversely to frequency | Handles class imbalance — rare classes get higher weight |
| `random_state=42` | Fixed seed | Reproducibility |
| `n_jobs=-1` | Use all CPU cores | Speed |

> [!IMPORTANT]
> **What is `class_weight="balanced"`?** — Automatically assigns weights inversely proportional to class frequencies. For label 3 (2.8%), the weight is much higher, so the model pays more attention to correctly classifying rare labels. Without this, the model would mostly predict label 0.

#### **3-Fold Cross-Validation Training:**
```python
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_proba = np.zeros((len(y_full), num_classes))
test_proba = np.zeros((X_test_sp.shape[0], num_classes))

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_full_sp, y_full)):
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        X_full_sp[tr_idx], y_full[tr_idx],
        eval_set=[(X_full_sp[val_idx], y_full[val_idx])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    oof_proba[val_idx] = clf.predict_proba(X_full_sp[val_idx])
    test_proba += clf.predict_proba(X_test_sp) / n_splits
```

**What's happening step by step:**
1. Split training data into 3 stratified folds
2. For each fold: train on 2 folds, validate on 1 fold
3. **Early stopping** (patience=40): If validation F1 doesn't improve for 40 rounds, stop training → prevents overfitting
4. **OOF (Out-of-Fold) predictions**: Each sample gets a prediction from a model that NEVER saw it during training
5. **Test predictions**: Average test predictions across all 3 folds → ensemble effect, more robust

> [!NOTE]
> **Why 3-fold CV instead of train/test split?**
> - Uses ALL data for both training and validation
> - More robust evaluation (3 different validation sets)
> - Final test predictions are averaged → reduces variance
> - Avoids single unlucky split

#### **F1-Macro Threshold Tuning:**
```python
def neg_f1_macro(log_weights, proba, y_true):
    w = np.exp(log_weights)
    adjusted = proba * w[np.newaxis, :]
    preds = np.argmax(adjusted, axis=1)
    return -f1_score(y_true, preds, average='macro')
```

**What this does:**
1. Takes predicted probabilities and multiplies them by **class-specific weights**
2. Uses `scipy.optimize.minimize` (Nelder-Mead) to FIND the optimal weights that maximize F1-macro
3. Tries multiple starting points (class-frequency-inverse, random, etc.)
4. Applies optimized weights to both OOF predictions and test predictions

**Why?**
- Even with `class_weight='balanced'`, the model might not optimally balance predictions across all classes
- Post-processing the probabilities with learned weights can **squeeze out extra F1-macro points**
- This is essentially a **calibration step** for multi-class classification

---

## 3. Theory Questions Bank (Sorted by Frequency)

### 🔥 MOST ASKED (90%+ chance)

**Q: Explain your notebook in detail (15-25 minutes)**
> Walk through the full pipeline: Problem statement → Data loading → EDA insights → Feature engineering (WHY each feature) → Preprocessing (TF-IDF, scaling, encoding) → Model exploration → Final pipeline (LightGBM + threshold tuning) → Submission

**Q: What is Bagging vs Boosting?**
> - **Bagging** (Bootstrap Aggregating): Train multiple models on random subsets of data IN PARALLEL. Average their predictions. Reduces VARIANCE. Example: Random Forest
> - **Boosting**: Train models SEQUENTIALLY, each correcting the previous model's errors. Reduces BIAS. Example: LightGBM, XGBoost, GradientBoosting
> - **Key difference**: Bagging = parallel, reduces variance. Boosting = sequential, reduces bias

**Q: What is Overfitting vs Underfitting?**
> - **Overfitting**: Model memorizes training data, performs poorly on unseen data. Train accuracy >> validation accuracy. Fix: regularization, dropout, early stopping, more data, simpler model
> - **Underfitting**: Model is too simple to capture patterns. Both train and validation accuracy are low. Fix: more features, more complex model, less regularization, train longer

**Q: What is Regularization? L1 vs L2?**
> - **Regularization**: Adding a penalty to the loss function to prevent overfitting
> - **L1 (Lasso)**: Penalty = sum of |weights|. Encourages sparsity (some weights → exactly 0). Good for feature selection
> - **L2 (Ridge)**: Penalty = sum of weights². Shrinks all weights towards 0 but keeps all. Prevents any single feature from dominating
> - In my LightGBM: `reg_alpha=0.1` (L1) + `reg_lambda=0.5` (L2) → elastic net style

**Q: Explain how TF-IDF works**
> - TF (Term Frequency) = (count of term in document) / (total terms in document)
> - IDF (Inverse Document Frequency) = log(total documents / documents containing the term)
> - TF-IDF = TF × IDF
> - Words that appear frequently in one document but rarely across all documents get high scores
> - With `sublinear_tf=True`: TF is replaced by 1 + log(TF), dampening the effect of very frequent terms

**Q: What is the difference between GridSearchCV and RandomizedSearchCV?**
> - **GridSearchCV**: Tests ALL combinations of hyperparameters. Exhaustive but expensive.  
>   For 3 params with 5 values each = 5³ = 125 combinations × 5 folds = 625 model fits
> - **RandomizedSearchCV**: Tests RANDOM combinations. Faster, often finds good solutions.  
>   You set n_iter (e.g., 50) — useful when the search space is huge
> - Grid is better when search space is small; Random is better for large spaces

**Q: Explain Confusion Matrix / Classification Report**
> - The classification report shows per-class Precision, Recall, F1, Support
> - Confusion matrix: rows = actual, columns = predicted
> - Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean
> - In this problem, F1-macro is the mean of all per-class F1 scores

### 🔶 FREQUENTLY ASKED (50-80% chance)

**Q: How does LightGBM work? How is it different from XGBoost?**
> - Both are gradient boosting frameworks: build trees sequentially, each correcting errors
> - **LightGBM**: Uses leaf-wise (best-first) tree growth → finds the leaf that reduces loss the most, splits it. Faster, better with large data but can overfit on small data
> - **XGBoost**: Uses level-wise (depth-first) tree growth → grows all leaves at the same level before moving deeper. Slower but more balanced trees
> - LightGBM is generally faster for large datasets (like our 198K rows × 30K features)

**Q: What is `learning_rate` and what happens if you change it?**
> - Learning rate controls how much each tree contributes to the final prediction
> - **High LR (e.g., 0.3)**: Faster convergence, but may overshoot the optimum→ less stable
> - **Low LR (e.g., 0.01)**: More precise, needs many more trees → slower but often better results
> - My setting: 0.10 — balanced between speed and precision

**Q: What is `n_estimators`?**
> - The maximum number of boosting trees to build
> - More trees generally = better performance, but diminishing returns and risk of overfitting
> - With early_stopping(40), training stops if no improvement for 40 rounds, so n_estimators=1200 is an upper limit

**Q: What is Data Leakage?**
> - When information from the test/validation set leaks into training
> - Example: Fitting a scaler on ALL data including test, then evaluating on the same test
> - In my notebook, I avoid this by:
>   - Using `StratifiedKFold` where each validation fold is never seen during training
>   - `fit_transform` on train, only `transform` on test

**Q: Why did you use StandardScaler?**
> - Scales features to zero mean, unit variance
> - Models like Logistic Regression, SVM, KNN, and MLP are **sensitive to feature scale**
> - Without scaling, features with larger ranges would dominate the learning
> - Tree-based models (LightGBM, XGBoost, R.F.) do NOT need scaling, but it doesn't hurt

**Q: What is OneHotEncoding? Why not LabelEncoding?**
> - **OneHotEncoding**: Creates binary columns for each category. No ordinal assumption
> - **LabelEncoding**: Assigns integers (0, 1, 2...) which implies an order that doesn't exist
> - For nominal variables (race, religion, gender), OHE is correct because "muslim" is NOT "greater than" "christian"

**Q: What is PCA?**
> - Principal Component Analysis: Linear dimensionality reduction
> - Finds directions (principal components) of maximum variance in the data
> - Projects data onto these directions, keeping top-k components
> - I use TruncatedSVD instead — same idea but works with sparse matrices (TF-IDF output)

**Q: How do you handle missing values?**
> - For categorical (race, religion, gender): Filled with 'missing' — treating NaN as its own category
> - For comment: Filled with empty string ''
> - Rationale: The NaN values are NOT random — they mean the annotator didn't detect that attribute in the comment. So "missing" IS meaningful information

**Q: What is `class_weight='balanced'`?**
> - Automatically adjusts weights inversely proportional to class frequency
> - Weight for class i = n_samples / (n_classes × n_samples_in_class_i)
> - Label 3 (2.8% of data) gets ~9x the weight of label 0 (57.7%)
> - Forces the model to pay equal attention to all classes

### 🔵 OCCASIONALLY ASKED (20-50% chance)

**Q: What is Logistic Regression? Why is it called "regression" if used for classification?**
> - Uses the logistic/sigmoid function to map linear combinations to probabilities [0,1]
> - Called "regression" because it regresses the log-odds of the target
> - For multi-class: uses softmax (a generalization of sigmoid)

**Q: How does SVM work?**
> - Finds the hyperplane that maximizes the margin between classes
> - LinearSVC: linear hyperplane, efficient for high-dimensional data (text)
> - The points closest to the hyperplane are "support vectors"

**Q: What is Naive Bayes? Why is it called "naive"?**
> - Applies Bayes' theorem: P(y|X) ∝ P(X|y)P(y)
> - "Naive" because it assumes all features are **independent** given the class
> - This assumption is almost never true in practice, but NB still works surprisingly well for text

**Q: What is an activation function? What is relu?**
> - Non-linear function applied after each neuron's linear transformation
> - ReLU: f(x) = max(0, x) — fast, prevents vanishing gradients
> - In MLPClassifier: `activation='relu'` used in hidden layers

**Q: How does a Neural Network (MLP) work?**
> - Multiple layers of neurons. Each neuron: output = activation(W·x + b)
> - Forward pass: input → hidden layers → output (softmax for classification)
> - Backward pass: compute gradients of loss, update weights with optimizer (Adam)
> - `hidden_layer_sizes=(256, 128, 64)` = 3 hidden layers with decreasing neurons

**Q: What is `num_leaves` in LightGBM?**
> - Maximum number of leaves per tree
> - Controls model complexity — more leaves = more complex model
> - LightGBM grows leaf-wise, so `num_leaves` directly controls how "deep" a tree can go
> - `num_leaves=31` is a good default

**Q: What is `random_state=42`?**
> - Sets the random seed for reproducibility
> - Ensures that the same code produces the same results every time
> - 42 is just a common convention (from "Hitchhiker's Guide to the Galaxy")

---

## 4. Live Coding Questions Bank

### 🔴 HIGHEST PRIORITY (Practice these by hand!)

**1. Load a dataset and train a model:**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 (macro):", f1_score(y_test, y_pred, average='macro'))
```

**2. Build a Pipeline:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=200))
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

**3. SimpleImputer with mean/most_frequent:**
```python
from sklearn.impute import SimpleImputer
import numpy as np

# Numerical: mean
imp_num = SimpleImputer(strategy='mean')
X_imputed = imp_num.fit_transform(X_train)

# Categorical: most_frequent
imp_cat = SimpleImputer(strategy='most_frequent')
X_cat_imputed = imp_cat.fit_transform(X_train_cat)
```

**4. Train-test split:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

**5. Print columns, find null values, check unique values:**
```python
print(train.columns.tolist())      # column names
print(train.isnull().sum())         # null count per column
print(train['label'].value_counts()) # class frequency
print(train['religion'].unique())    # unique values
```

**6. ColumnTransformer with different preprocessing:**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ct = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'city'])
])
X_transformed = ct.fit_transform(X)
```

**7. RFE (Recursive Feature Elimination):**
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_diabetes

data = load_diabetes()
rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=2)
rfe.fit(data.data, (data.target > data.target.mean()).astype(int))
print("Selected features:", rfe.support_)
```

**8. Plot a bar chart:**
```python
import matplotlib.pyplot as plt
models = ['LR', 'RF', 'LGBM']
scores = [0.65, 0.72, 0.78]
plt.bar(models, scores, color='steelblue')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('Model Comparison')
plt.show()
```

---

## 5. Proctor Tips & Strategies

### 🎤 Presentation Strategy
1. **Time management**: Aim for 20-25 minutes of notebook explanation. Don't rush, don't go too slow
2. **Start with the problem statement**: "This is a multi-class text classification problem with 4 labels on a social media comment dataset..."
3. **Tell a story**: EDA insights → feature engineering decisions → model exploration → final pipeline optimization
4. **Explain WHY at every step**: Don't just say "I used TF-IDF", say "I used TF-IDF because it assigns higher weights to distinctive words, which is important for distinguishing comment categories"
5. **Never pause!**: If you pause, the proctor will ask a question. Keep talking to minimize interruptions
6. **Be honest**: If you don't know something, say "I'm not sure about that, but I believe..." or "I don't know that specific detail"

### 📝 Key Points to Highlight
- F1-macro metric choice (class imbalance)
- Feature engineering rationale (each feature)
- Why LightGBM was chosen as the final model
- Threshold tuning as a post-processing step
- Early stopping to prevent overfitting
- 3-fold CV for robust evaluation

### ⚠️ Common Traps
- Don't say you used AI/LLMs to write the code — say you learned from documentation, TAs, YouTube
- Know the difference between `fit`, `transform`, and `fit_transform`
  - `fit`: Learn parameters from data (e.g., mean/std for scaling)
  - `transform`: Apply learned parameters to data
  - `fit_transform`: Do both in one step (only on TRAINING data!)
- Know what `Pipeline` vs `ColumnTransformer` is:
  - Pipeline: Sequential steps (one feeds into the next)
  - ColumnTransformer: Applies DIFFERENT transformations to DIFFERENT columns, then combines

### 🏆 If Asked "What would you do to improve?"
- Try **BERT/sentence-transformers** for better text embeddings
- Do **more EDA** — word clouds per label, temporal trends
- Try **SMOTE** or other oversampling for class imbalance
- **Stacking/blending** multiple models
- **More feature engineering** — sentiment scores, readability indices
- **Hyperparameter tuning** with RandomizedSearchCV or Optuna

---

> [!CAUTION]
> **CRITICAL REMINDERS:**
> - Practice explaining the notebook **out loud** at least 3 times before the viva
> - Have your notebook open and ready to screen-share
> - Keep a Google Colab tab ready for live coding
> - Know how to load iris, breast_cancer, California housing datasets from sklearn
> - Know `train_test_split`, `Pipeline`, `SimpleImputer`, `StandardScaler` syntax BY HEART
