# MLP Level 2 Viva — Complete Preparation Guide
## Proctor: Lvl2_74

---

# PART 1: PROJECT OVERVIEW

## Q: What is the objective of this project?
**A:** This is a **multi-class text classification** problem. Given online comments with metadata (upvotes, downvotes, emoticons, demographic info), we predict which **category (0, 1, 2, or 3)** a comment belongs to. The competition metric is **F1-macro**, which treats all classes equally important — critical because our dataset is **heavily imbalanced** (class 0 has 114K samples, class 3 has only 5K).

## Q: How did you proceed in this project?
**A:** End-to-end ML pipeline:
1. **Load data** → understand shape, types, missing values
2. **EDA** → visualize distributions, class imbalance, feature correlations
3. **Data cleaning** → handle missing values, convert types, drop/create columns
4. **Feature engineering** → extract text features (word count, caps ratio, etc.), time features, interaction features
5. **Preprocessing** → TF-IDF vectorization + SVD (text), StandardScaler (numeric), OneHotEncoder (categorical)
6. **Train/val split** → 80/20 stratified split
7. **Model training** → 8 models: LogReg, SGD, NB, KNN, SVM, LightGBM, XGBoost, MLP
8. **Hyperparameter tuning** → GridSearchCV on 3 models
9. **Model comparison** → bar charts, confusion matrices
10. **Final pipeline** → 3-fold CV with LightGBM + F1-macro threshold tuning
11. **Submission** → generate predictions on test set

---

# PART 2: LINE-BY-LINE NOTEBOOK WALKTHROUGH

## Cell 1 — Imports
```python
import pandas as pd, numpy as np, re, time, warnings
warnings.filterwarnings('ignore')
```
- **Why filterwarnings('ignore')?** Suppresses convergence warnings from sklearn that clutter output. In production you'd want to see them, but for a competition notebook it keeps things clean.

```python
from sklearn.model_selection import train_test_split, StratifiedKFold
```
- **StratifiedKFold** — preserves class distribution in each fold. Critical for imbalanced data (our class 3 is only 2.7% of data).
- **train_test_split** — splits data into train/validation sets.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
- **TF-IDF** converts text to numeric features. More on this below.

```python
from sklearn.decomposition import TruncatedSVD
```
- **TruncatedSVD** — dimensionality reduction for sparse matrices (TF-IDF output). Regular PCA requires dense matrices = memory explosion.

```python
from sklearn.calibration import CalibratedClassifierCV
```
- **CalibratedClassifierCV** — wraps LinearSVC to provide `predict_proba()` (SVM doesn't natively output probabilities).

---

## Cell 3 — Data Loading & EDA

```python
DATA_DIR = "/kaggle/input/comment-category-prediction-challenge"
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
```
- **Why os.path.join?** Platform-independent path construction. Works on Windows and Linux.

```python
label_counts = train_raw['label'].value_counts().sort_index()
```
- **Why sort_index()?** Shows labels in order 0,1,2,3 instead of by frequency.
- **What does this show?** Heavy imbalance: class 0 = 114K (57.7%), class 3 = 5.5K (2.8%). This is why we use `class_weight='balanced'` and F1-macro instead of accuracy.

### Q: Why is accuracy a bad metric for this dataset?
**A:** Because of **class imbalance**. A dummy model predicting class 0 for everything gets ~57.7% accuracy. That's "good" accuracy but useless. F1-macro averages F1 across all classes equally, so it punishes ignoring minority classes.

---

## Cell 5 — Visualizations

### Q: Why did you plot each graph? What does it show?
These will be asked about in detail. Prepare for each plot:

1. **Label distribution bar chart** → Shows class imbalance. Classes 0 and 2 dominate. This tells us we need balanced class weights or oversampling.

2. **Comment length distribution by label** → Some classes have longer/shorter comments. Class 3 has shorter comments (mean=194 chars) vs class 1 (mean=336 chars). This means comment length is a useful feature.

3. **Crosstab: Race vs Label** → Shows how demographic features relate to labels. E.g., comments mentioning 'black' race are disproportionately label=1. This tells us demographic features carry predictive signal.

4. **Correlation heatmap** → Shows relationships between numeric features. High correlation between features means redundancy — can guide feature selection.

5. **Boxplots** → Show distribution spread and outliers for each feature by class.

---

## Cell 7 — Data Cleaning

```python
train['comment'] = train['comment'].fillna('')
```
- **Why fillna('')?** 1 comment is NaN. TF-IDF can't handle NaN strings, so we replace with empty string.

```python
for col in ['race', 'religion', 'gender']:
    train[col] = train[col].fillna('unknown')
```
- **Why 'unknown' and not drop?** 145K/198K rows have NaN demographics (~73%). Dropping would lose most data. 'unknown' preserves the information that "no demographic was provided" — which itself is informative (missing=class 0 tends to dominate).

```python
train['disability'] = train['disability'].astype(int)
```
- **Why?** Converts boolean True/False to 1/0 for numerical models.

```python
train['post_id'] = train['post_id'].astype(str)
```
- **Why str?** post_id is categorical (52 unique values like 72, 39, etc.), not truly numeric. Converting to string lets OneHotEncoder treat it correctly.

---

## Cell 8 — Feature Engineering

### Q: What is feature engineering? Why do it?
**A:** Creating new informative features from raw data to help models learn patterns. Raw data alone (just the comment text + a few columns) may not capture nuances like comment style, aggression signals, etc.

```python
df["char_len"] = df["comment"].str.len()
df["word_count"] = df["comment"].str.split().str.len().fillna(0).astype(int)
```
- **Why?** Length features are cheap to compute and surprisingly predictive. Toxic/hateful comments (class 1) tend to be longer.

```python
df["caps_ratio"] = df["comment"].str.count(r'[A-Z]') / (df["char_len"] + 1)
```
- **Why +1?** Avoids division by zero for empty comments.
- **Why caps_ratio?** ALL CAPS text often indicates SHOUTING — correlated with toxic content.

```python
df["has_url"] = df["comment"].str.contains(r'http[s]?://', regex=True).astype(int)
```
- **Why?** Comments with URLs may be spam or less toxic. It's a binary signal.

```python
df["vote_total"] = df["upvote"] + df["downvote"]
df["upvote_ratio"] = df["upvote"] / (df["vote_total"] + 1)
df["log_upvote"] = np.log1p(df["upvote"])
```
- **Why log transform?** Upvote/downvote distributions are heavily right-skewed. `log1p` (= log(x+1)) compresses the range, helping linear models. The +1 handles zeros.
- **Why upvote_ratio?** A comment with 10 upvotes and 0 downvotes is very different from 10 upvotes and 100 downvotes. The ratio captures this.

```python
df["if_interact"] = df["if_1"] * df["if_2"]
```
- **Why interaction feature?** Captures the **joint effect** of if_1 and if_2. If if_1=0 (meaning no demographic info), then if_interact=0 regardless of if_2. This creates a signal for "both identity features present."

---

## Cell 10 — Preprocessor Pipeline

### Q: What is data preprocessing?
**A:** Transforming raw data into a format suitable for ML models. Includes handling missing values, encoding categories, scaling numbers, and vectorizing text.

```python
TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, max_df=0.92,
                sublinear_tf=True, strip_accents='unicode', norm='l2')
```

### Q: Why TF-IDF and not Count Vectorizer?
**A:**
- **CountVectorizer** just counts word frequencies. Common words like "the" dominate.
- **TF-IDF** = Term Frequency × Inverse Document Frequency. It **downweights common words** and **upweights rare, informative words**.
- `sublinear_tf=True` applies log scaling: tf = 1 + log(tf). This prevents very frequent words from overwhelming the signal.
- **Result:** TF-IDF gives better features because it captures **how important** a word is to a specific document vs the whole corpus.

### Q: How does TF-IDF work?
**A:**
- **TF(t,d)** = (count of term t in document d) / (total terms in d)
- **IDF(t)** = log(N / df(t)), where N = total documents, df(t) = documents containing term t
- **TF-IDF(t,d)** = TF × IDF
- With `sublinear_tf=True`: TF = 1 + log(count) instead of raw count
- Words appearing in almost every document get low IDF → low TF-IDF (e.g., "the", "is")
- Rare but meaningful words get high IDF → high TF-IDF (e.g., slurs, hate terms)

### Q: What do max_features, ngram_range, min_df, max_df mean?
- **max_features=30000** — keep only top 30K features by term frequency (vocabulary limit)
- **ngram_range=(1,2)** — use unigrams ("hate") AND bigrams ("hate speech")
- **min_df=2** — ignore terms appearing in fewer than 2 documents (removes typos/noise)
- **max_df=0.92** — ignore terms appearing in >92% of documents (removes stop-words like "the")

```python
TruncatedSVD(n_components=150, random_state=42)
```

### Q: What is Truncated SVD? How does it work?
**A:** Truncated SVD (Latent Semantic Analysis) reduces dimensionality by finding the **k most important directions** (components) in the feature space.
- Decomposes the TF-IDF matrix X ≈ U × Σ × V^T
- Keeps only the top k singular values and their corresponding directions
- **Does NOT crop/cut data** — it **projects** data onto lower-dimensional subspace that captures maximum variance
- 30,000 TF-IDF features → 150 SVD components. These 150 components capture the main "topics" in the text.
- **Why not PCA?** PCA needs to center data (subtract mean), which destroys sparsity. TruncatedSVD works directly on sparse matrices.

```python
StandardScaler()
```

### Q: What is feature scaling? Why is it required? Different ways?
**A:** Scaling transforms features to similar ranges so no single feature dominates.
- **Why needed?** Models like LogReg, SVM, KNN use distance/gradient calculations. Feature "upvote" (range 0-1000) would dominate "caps_ratio" (range 0-1) without scaling.
- **Methods:**
  1. **StandardScaler** — Z-score: (x - mean) / std. Output has mean=0, std=1
  2. **MinMaxScaler** — (x - min) / (max - min). Output in [0, 1]
  3. **RobustScaler** — uses median/IQR, robust to outliers
  4. **MaxAbsScaler** — divides by max absolute value, good for sparse data
- **Tree-based models (LightGBM, XGBoost) don't need scaling** — they split on thresholds, not distances.

```python
OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```
- **Why OneHotEncoder?** Converts categorical features (post_id, race, religion, gender) into binary columns.
- **handle_unknown='ignore'** — if test data has a category not seen in training, it creates all-zero row instead of crashing.

---

## Cell 11 — Train/Validation Split

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

### Q: What is random_state=42? Why 42?
**A:**
- **random_state** sets the random seed for reproducibility. Same seed = same split every time.
- **Why 42?** It's just a convention (from "Hitchhiker's Guide to the Galaxy" — "the answer to life, universe, and everything"). Any integer works. The specific number doesn't affect model quality.
- **Why use it?** Without it, every run gives different splits → different results → can't compare experiments fairly.

### Q: What does stratify=y do?
**A:** Ensures the class distribution in train and val matches the original. Without it, random split might put too few class-3 samples in validation (class 3 is only 2.8%).

---

## Cell 13 — Logistic Regression & SGD

### Q: What is Logistic Regression? Why is it called "regression"?
**A:**
- Despite the name, it's a **classification** algorithm
- Called "regression" because it uses a **regression function (sigmoid)** internally: P(y=1|x) = 1 / (1 + e^(-w·x))
- It predicts **probability** of each class, then picks the highest
- For multi-class: uses **softmax** (one-vs-rest or multinomial)
- **5 components:** Data (X,y), Model (linear: w·x + b → sigmoid), Loss (cross-entropy/log loss), Optimization (gradient descent / LBFGS), Evaluation (F1-macro)

```python
LogisticRegression(C=5.0, solver='lbfgs', penalty='l2', max_iter=500,
                   class_weight='balanced', n_jobs=-1, random_state=42)
```
- **C=5.0** — inverse regularization strength. Higher C = less regularization = more complex model
- **penalty='l2'** — L2 regularization (ridge): adds λ·||w||² to loss. Prevents overfitting by penalizing large weights
- **solver='lbfgs'** — optimization algorithm. LBFGS is efficient for small-medium datasets
- **max_iter=500** — maximum gradient descent iterations. If model doesn't converge in 500 steps, it stops (hence the ConvergenceWarning)
- **class_weight='balanced'** — automatically adjusts weights inversely proportional to class frequency. Class 3 (rare) gets higher weight so model doesn't ignore it
- **n_jobs=-1** — use all CPU cores for parallel computation

### Q: Why did you use 500 iterations for Logistic Regression?
**A:** Default is 100, but with 470 features and 4 classes, LBFGS needs more iterations to converge. 500 is a balance between convergence and speed. Even at 500, we see ConvergenceWarnings for high C values — ideally we'd increase to 1000, but time constraints on Kaggle.

### Q: What is SGD Classifier?
**A:** Stochastic Gradient Descent classifier. Instead of computing gradient on ALL data (like LBFGS), it updates weights using ONE sample at a time → much faster for large datasets.
- `loss='modified_huber'` — smooth version of hinge loss, provides probability estimates
- `alpha` — regularization strength (opposite of C). Higher alpha = more regularization.

---

## Cell 15 — Naive Bayes, KNN, SVM

### Q: Why does Naive Bayes perform so poorly (4.68% accuracy)?
**A:** Gaussian NB assumes features are **normally distributed and independent**. Our TF-IDF + SVD features violate both assumptions. SVD components are orthogonal but not Gaussian, and text features have complex dependencies. NB works better with raw count-based features.

### Q: How does KNN work?
**A:** For each test point, find the K nearest training points (by Euclidean distance), then majority vote.
- `n_neighbors=5` — K=5
- `algorithm='ball_tree'` — efficient for high-dimensional data (faster than brute-force)
- **Why mediocre (0.54 F1)?** KNN struggles in high dimensions ("curse of dimensionality") and with imbalanced data.

### Q: How does LinearSVC work?
**A:** Support Vector Machine with linear kernel. Finds the hyperplane that **maximizes the margin** between classes.
- **Margin** = distance from hyperplane to nearest data points (support vectors)
- **C=1.0** — tradeoff between margin width and classification errors. Higher C = stricter (narrow margin, fewer errors on training data)
- **Hinge loss:** max(0, 1 - y·f(x)). Points correctly classified with margin >1 contribute 0 loss.

---

## Cell 17 — LightGBM & XGBoost

### Q: How does LightGBM work?
**A:** Gradient Boosted Decision Trees (GBDT). Builds trees **sequentially**, each tree corrects errors of the previous ones.
1. Start with a base prediction (e.g., class frequencies)
2. Compute **gradient/residual** = difference between prediction and actual
3. Fit a decision tree to predict the residual
4. Add tree prediction (scaled by learning_rate) to current prediction
5. Repeat for n_estimators trees

**LightGBM's unique features:**
- **Leaf-wise** tree growth (instead of level-wise) — grows the leaf with highest loss reduction first → faster, better accuracy
- **Histogram-based** splitting — bins continuous features into ~256 buckets → much faster than exact splits
- **GOSS** (Gradient-based One-Side Sampling) — keeps samples with large gradients, randomly samples from small gradients → faster training

```python
lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.05,
    num_leaves=31, max_depth=8, min_child_samples=30,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.5,
    class_weight='balanced')
```
- **n_estimators=1500** — max number of boosting rounds (trees)
- **learning_rate=0.05** — shrinks each tree's contribution. Lower = more trees needed but better generalization
- **num_leaves=31** — max leaves per tree. Controls complexity. 31 is typical for medium data
- **max_depth=8** — limits tree depth (additional constraint beyond num_leaves)
- **min_child_samples=30** — minimum samples in a leaf. Prevents overfitting on noise
- **subsample=0.8** — use 80% of data per tree (row sampling). Adds randomness, reduces overfitting
- **colsample_bytree=0.8** — use 80% of features per tree. Same benefit
- **reg_alpha=0.1** — L1 regularization on leaf weights
- **reg_lambda=0.5** — L2 regularization on leaf weights
- **early_stopping(50)** — stop if validation loss doesn't improve for 50 rounds

### Q: What is learning rate?
**A:** Controls how much each tree contributes to the final prediction. `prediction += learning_rate × tree_prediction`. Lower LR = smaller steps = needs more trees but less overfitting. It's the key tradeoff: LR=0.05 with 1500 trees vs LR=0.1 with 600 trees.

### Q: How is XGBoost different from LightGBM?
**A:**
| Feature | XGBoost | LightGBM |
|---------|---------|----------|
| Tree growth | Level-wise | Leaf-wise |
| Splitting | Exact or approx | Histogram-based |
| Speed | Slower | Faster |
| Missing values | Built-in handling | Built-in handling |
| Sampling | Random | GOSS (gradient-based) |

Both are GBDT, but LightGBM is typically **2-10x faster** with similar accuracy.

---

## Cell 19 — MLP Classifier

```python
MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam',
              alpha=0.005, batch_size=512, learning_rate='adaptive',
              max_iter=50, early_stopping=True)
```
- **hidden_layer_sizes=(128,)** — one hidden layer with 128 neurons
- **relu** — max(0, x). Introduces non-linearity, avoids vanishing gradient
- **adam** — adaptive optimizer (combines momentum + RMSprop)
- **alpha=0.005** — L2 regularization strength
- **early_stopping=True** — monitors validation loss, stops when it plateaus

---

## Cell 23 — Hyperparameter Tuning

### Q: Why is hyperparameter tuning required?
**A:** Default hyperparameters are generic — they don't know your specific dataset. Tuning finds the **optimal combination** for YOUR data. Example: LogReg with C=0.01 gives F1=0.65, but C=5.0 gives F1=0.71 — that's a huge difference.

### Q: What are some ways to perform HPT?
**A:**
1. **Manual search** — try a few values by hand (what we did initially)
2. **GridSearchCV** — exhaustive search over all combinations. Guaranteed to find best in the grid, but slow
3. **RandomizedSearchCV** — random sampling from parameter distributions. Faster, often finds good solutions
4. **Bayesian optimization** (Optuna, hyperopt) — smart search using probability model
5. **Early stopping** — not HPT per se, but automatically finds optimal number of trees

### Q: Difference between GridSearchCV and RandomizedSearchCV?
**A:**
- **GridSearchCV** — tries EVERY combination. 5 C values × 2 solvers = 10 combinations × 3 folds = 30 fits. Exhaustive but slow.
- **RandomizedSearchCV** — randomly samples N combinations from parameter distributions. If you have 1000 possible combinations, sampling 50 random ones often finds ~95% of the best. Much faster.
- **When to use which?** Grid for small search spaces (<50 combinations), Random for large spaces.

### Q: How do we choose parameters?
**A:**
1. Start with defaults
2. Identify which parameters matter most (learning_rate, C, num_leaves)
3. Do coarse search first (e.g., C = [0.01, 0.1, 1, 10])
4. Then fine search around the best (e.g., C = [3, 5, 7, 10])
5. Use cross-validation to evaluate each combination
6. Pick the combination with best CV score

---

## Cell 25 — Final Pipeline

### Q: Why 3-fold CV for the final pipeline?
**A:** Uses ALL training data for both training and validation (each sample is validated exactly once). Averaging predictions from 3 models reduces variance. 3 folds instead of 5 for speed on Kaggle.

### Q: What is threshold tuning?
**A:** After getting probability predictions, instead of just argmax, we **weight** each class's probability before taking argmax. The weights are optimized (using Nelder-Mead) to maximize F1-macro. This helps because the model may be systematically under-confident on rare classes.

---

# PART 3: WHAT IS A CONFUSION MATRIX?

### Q: What is a confusion matrix and why do we use it?
**A:** A table showing actual vs predicted labels. For 4 classes, it's a 4×4 matrix.
- **Diagonal** = correct predictions
- **Off-diagonal** = errors
- **Why use it?** Accuracy alone hides class-specific errors. Confusion matrix shows exactly WHERE the model fails. E.g., our model often confuses class 3 with class 0 — that's visible in the matrix but hidden by accuracy.
- **Precision** = TP / (TP + FP) — "of all predicted positive, how many are actually positive?"
- **Recall** = TP / (TP + FN) — "of all actual positive, how many did we catch?"
- **F1** = 2 × (P × R) / (P + R) — harmonic mean, balances precision and recall

---

# PART 4: HOUSE PRICE PREDICTION QUESTION

### Q: If house's past price is given and you need to predict future prices, which ML model is suitable?
**A:** This is a **regression** problem (continuous output).
- **Best model:** Gradient Boosted Trees (XGBoost / LightGBM) or **Linear Regression** for interpretability.
- **Algorithm steps (Linear Regression):**
  1. **Data:** Features (area, bedrooms, location, year) → Target (price)
  2. **Model:** price = w₁×area + w₂×bedrooms + ... + b (linear combination)
  3. **Loss function:** MSE = (1/n) × Σ(predicted - actual)²
  4. **Optimization:** Minimize MSE using gradient descent. Update: w = w - lr × ∂MSE/∂w
  5. **Evaluation:** RMSE, MAE, R² score
- **Why gradient boosting?** Captures non-linear relationships (price doesn't scale linearly with area).

---

# PART 5: WHAT HAPPENS IN DIMENSIONALITY REDUCTION?

### Q: What happens when dimensionality reduces? Does it crop/cut data?
**A:** **No, it does NOT crop or cut data.** It **transforms/projects** data into a lower-dimensional space.
- Imagine a 3D object. Its shadow on a wall is 2D — it captures the main shape but loses some detail. That's dimensionality reduction.
- **Mathematically:** TruncatedSVD finds the directions of maximum variance, then projects all data points onto those directions.
- Original: 30,000 TF-IDF features (mostly zeros, lots of noise)
- After SVD: 150 components that capture 80-90% of the information
- **Information loss:** Some, but mostly noise. The signal is preserved.

---

# PART 6: STACKING CLASSIFIER

### Q: Explain working of Stack Classifier
**A:** Stacking combines multiple models by training a **meta-learner** on their predictions:
1. **Level 0:** Train base models (LogReg, SVM, LightGBM, etc.) using K-fold CV
2. Each model makes **out-of-fold predictions** on the training data
3. **Level 1:** Stack these predictions as features, train a meta-learner (usually LogReg) to make the final prediction
4. **Why it works:** Different models capture different patterns. The meta-learner learns which model to trust for which type of input.

*Note: Our notebook uses simple averaging instead of stacking for the final pipeline (simpler and almost as good).*

---

# PART 7: CODING QUESTIONS (if proctor changes and asks coding)

## Write code to perform TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(train['comment'])  # fit on train
X_test_tfidf = tfidf.transform(test['comment'])   # only transform on test (no fit!)
```

## Write code for train/test split
```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

## Write code for GridSearchCV
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'saga']}
gs = GridSearchCV(LogisticRegression(max_iter=500), param_grid, scoring='f1_macro', cv=3)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

## Write code for confusion matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_val, predictions)
disp = ConfusionMatrixDisplay(cm, display_labels=[0,1,2,3])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

## Write code for a basic LightGBM model
```python
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.1, num_leaves=31,
                            class_weight='balanced', random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50)])
preds = model.predict(X_val)
print(f1_score(y_val, preds, average='macro'))
```

## Write StandardScaler from scratch
```python
class MyScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self
    def transform(self, X):
        return (X - self.mean_) / (self.std_ + 1e-8)  # +epsilon to avoid /0
    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

## Write F1 score from scratch
```python
def f1_macro(y_true, y_pred):
    classes = set(y_true)
    f1s = []
    for c in classes:
        tp = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)
    return sum(f1s) / len(f1s)
```

---

# PART 8: QUICK REFERENCE — ALL 20 PROCTOR QUESTIONS WITH SHORT ANSWERS

| # | Question | Key Answer |
|---|----------|-----------|
| 1 | Steps in preprocessing | Handle missing → encode categoricals → scale numerics → vectorize text → reduce dimensions |
| 2 | Feature scaling & why needed | Transform features to similar ranges; needed for distance/gradient-based models |
| 3 | What happens in dimensionality reduction | Projects data to lower-dim subspace preserving max variance; doesn't crop |
| 4 | House price prediction model | Regression problem; use Linear Regression or XGBoost; minimize MSE via gradient descent |
| 5 | Why HPT required | Default params are generic; tuning finds optimal combo for YOUR data |
| 6 | Ways to do HPT | GridSearchCV, RandomizedSearchCV, Bayesian (Optuna), manual search |
| 7 | Grid vs Random search | Grid=exhaustive (slow, guaranteed best in grid); Random=sampled (fast, usually finds ~95% of best) |
| 8 | Project objective | Multi-class comment classification (4 classes); metric = F1-macro |
| 9 | Models used | LogReg, SGD, NB, KNN, LinearSVC, LightGBM, XGBoost, MLP (8 total) |
| 10 | Best model & why | LightGBM (balanced) — handles imbalance via class weights, captures non-linear patterns, fast training |
| 11 | How you proceeded | EDA→Clean→FeatEng→Preprocess→8 models→HPT→Compare→Final CV pipeline→Submit |
| 12 | How LightGBM works | Sequential trees, each corrects errors of previous; leaf-wise growth; histogram splits |
| 13 | Confusion matrix | Actual vs Predicted table; shows WHERE model fails; used to compute P, R, F1 per class |
| 14 | XGBoost vs LightGBM | XGB=level-wise, exact splits, slower; LGBM=leaf-wise, histogram, faster |
| 15 | LogReg: use case & name | Classification via sigmoid/softmax; called "regression" because uses regression function internally |
| 16 | Learning rate | Controls step size in optimization; shrinks each tree's contribution in boosting |
| 17 | How to choose parameters | Coarse grid → fine grid around best → cross-validate each combo |
| 18 | What is random_state | Random seed for reproducibility; same seed = same results |
| 19 | Why 42 | Convention from pop culture; any integer works; doesn't affect quality |
| 20 | Why 500 iterations in LogReg | Default 100 is insufficient for 470 features × 4 classes; 500 balances convergence vs speed |

---

> **Final Tip from past students:** The proctor spends ~30 min on EDA. Know every plot's purpose and interpretation. For the rest, be confident and concise. If you don't know something, say "I'm not sure about the exact details but here's my understanding..." — honesty > bluffing.
