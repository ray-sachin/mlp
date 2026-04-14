# 🎯 ULTIMATE MLP Level 2 Viva Preparation — Proctor Lvl2_74

---

# SECTION A: PSYCHOLOGICAL PROFILE OF PROCTOR Lvl2_74

## Pattern Analysis (12+ student reports, Feb 2024 – Apr 2026)

### 🧠 Proctor Behavior Model

1. **Phase 1 (First ~10 min): Theory bombardment BEFORE notebook**
   - He asks 5-8 theory questions BEFORE you even share screen
   - Questions are ALWAYS the same set (confirmed by 8+ students: "Exactly the same questions")
   - He tests if you know basics FIRST — if you fail here, the rest is harder

2. **Phase 2 (~10-15 min): "Share notebook and explain"**
   - He tells you to share and give an overview
   - He does NOT want line-by-line — he wants HIGH-LEVEL overview
   - One student said: "Wasn't even paying attention to my nb explanation"
   - He skims through notebook, stops at EDA plots and asks WHY

3. **Phase 3 (~5 min): Specific notebook questions**
   - Which models did you use? Why did the best one work?
   - Did you handle null values? How?
   - How many HPT runs? Any score increase?
   - Rarely asks coding (only 1 out of 12 students got coding from him)

### 🎯 His Fixed Question Bank (90% probability these exact questions appear)

| # | Question | Times Asked | Priority |
|---|----------|------------|----------|
| 1 | What is data preprocessing? Steps? | 10/12 | 🔴 CRITICAL |
| 2 | What is feature scaling? Why needed? | 8/12 | 🔴 CRITICAL |
| 3 | What is hyperparameter tuning? Why required? | 10/12 | 🔴 CRITICAL |
| 4 | GridSearchCV vs RandomizedSearchCV? | 8/12 | 🔴 CRITICAL |
| 5 | What is random_state=42? Why 42? | 9/12 | 🔴 CRITICAL |
| 6 | House price prediction — which model + algorithm? | 7/12 | 🔴 CRITICAL |
| 7 | What is dimensionality reduction? Does it crop? | 7/12 | 🔴 CRITICAL |
| 8 | Which models did you try? Best one? Why? | 6/12 | 🟡 HIGH |
| 9 | What is learning rate? | 5/12 | 🟡 HIGH |
| 10 | Why is accuracy bad for this dataset? | 4/12 | 🟡 HIGH |
| 11 | Why EDA? Why did you plot X graph? | 5/12 | 🟡 HIGH |
| 12 | Feature engineering — what and why? | 4/12 | 🟡 HIGH |
| 13 | How did you handle missing/null values? | 5/12 | 🟡 HIGH |
| 14 | Explain TF-IDF vs Count Vectorizer | 3/12 | 🟢 MEDIUM |
| 15 | Explain working of Logistic Regression | 3/12 | 🟢 MEDIUM |
| 16 | Outlier detection and treatment | 3/12 | 🟢 MEDIUM |
| 17 | Difference between Random Forest and Decision Tree | 2/12 | 🟢 MEDIUM |
| 18 | Explain LightGBM / XGBoost working | 3/12 | 🟢 MEDIUM |
| 19 | Confusion matrix — what and why? | 3/12 | 🟢 MEDIUM |
| 20 | XGBoost vs LightGBM difference? | 2/12 | 🟢 MEDIUM |

### 🧩 His Personality Traits
- **Predictable**: Uses the SAME question set every time
- **Nice but detail-oriented on EDA**: Will spend extra time if you have good graphs
- **Hates memorized answers**: One student said he's "fed up with the Hitchhiker's Guide answer" for 42 — so give a GENUINE answer (see below)
- **Values confidence**: Student who got 47/50 said "All depends on how well u speak... I just answered everything confidently"
- **No coding usually**: Only 1/12 students were asked to code. BUT be prepared anyway
- **~27 min duration**: Average viva lasts 25-30 min
- **Rarely interrupts**: Lets you explain, then asks follow-ups

### ⚡ Winning Strategy for Lvl2_74

1. **Memorize the 7 CRITICAL questions and nail them** (80% of your marks)
2. When he says "share notebook", give a **confident 15-min high-level overview** — don't read code
3. **Pre-answer questions in your explanation** — say "I used StandardScaler because..." before he asks
4. On EDA: explain **each graph's PURPOSE and INSIGHT** — he cares about reasoning
5. On random_state=42: **don't say "Hitchhiker's Guide"** — say genuinely "42 is just a convention, any integer works for reproducibility, I use it because it's the community standard"

---

# SECTION B: THE 7 CRITICAL ANSWERS (Memorize these word-for-word)

## 1. "What is data preprocessing? What are the steps?"

**Answer:**
"Data preprocessing is the process of converting raw data into a clean, suitable format for machine learning models. The main steps I followed are:

1. **Handling missing values** — I filled NaN comments with empty string, and NaN demographics (race, religion, gender) with 'unknown' since 73% were missing
2. **Data type conversion** — converted post_id to string (it's categorical, not numeric), and disability to integer
3. **Feature engineering** — created text features like word_count, caps_ratio, has_url, and interaction features like upvote_ratio
4. **Text vectorization** — used TF-IDF to convert text to numbers
5. **Encoding** — OneHotEncoder for categorical features
6. **Scaling** — StandardScaler for numeric features
7. **Dimensionality reduction** — TruncatedSVD to reduce TF-IDF features from 30K to 150"

## 2. "What is feature scaling? Why is it required?"

**Answer:**
"Feature scaling is transforming features to a similar range so no single feature dominates the model.

**Why required:** Models like Logistic Regression and SVM use gradient-based optimization. If 'upvote' ranges from 0 to 1000 but 'caps_ratio' ranges from 0 to 1, the gradient will be dominated by upvote. Scaling fixes this.

**Methods I know:**
- **StandardScaler**: Z-score normalization → (x - mean)/std → output has mean=0, std=1
- **MinMaxScaler**: Scales to [0,1] → (x - min)/(max - min)
- **RobustScaler**: Uses median and IQR → robust to outliers
- I used StandardScaler because my numeric features are roughly normally distributed.

**Note:** Tree-based models like LightGBM don't need scaling — they split on thresholds, not distances."

## 3. "Why is hyperparameter tuning required?"

**Answer:**
"Default hyperparameters are generic — they don't know my specific dataset. Tuning finds the optimal settings for MY data.

For example, in LogisticRegression, C controls regularization. With C=0.01 I got F1=0.65, but with C=5.0 I got F1=0.71. That's a 6% improvement from just tuning one parameter.

I used GridSearchCV with 3-fold cross-validation on 3 models: LogisticRegression (tuned C, solver), SGDClassifier (tuned alpha, loss), and LinearSVC (tuned C, loss, max_iter). The scoring metric was f1_macro to match the competition."

## 4. "GridSearchCV vs RandomizedSearchCV?"

**Answer:**
"Both search for the best hyperparameters, but differently:

- **GridSearchCV** tries **every** combination exhaustively. If I have 5 C values × 2 solvers × 3 folds = 30 model fits. Guaranteed to find the best in the grid, but slow.
- **RandomizedSearchCV** randomly samples N combinations from parameter **distributions**. If I have 1000 possible combos, sampling 50 random ones often finds ~95% of the best. Much faster.

**When to use which:** Grid for small search spaces (<50 combos), Random for large spaces or when you have continuous parameters.

I used GridSearchCV because my search spaces were small (10-24 combos per model)."

## 5. "What is random_state=42? Why 42?"

**Answer:**
"random_state is a seed for the random number generator. Setting it ensures **reproducibility** — same seed gives the same train/test split, same tree structure, same results every run. Without it, every run is different and you can't fairly compare experiments.

As for why 42 specifically — it's just a convention in the data science community. Any integer gives equally valid results. It became popular and is now the standard default across tutorials and documentation. I use it for consistency with community practice."

## 6. "If a house was priced at X rupees 10 years ago, predict its price now. Which model?"

**Answer:**
"This is a **regression** problem since we're predicting a continuous value (price).

I'd use **Gradient Boosted Trees (XGBoost or LightGBM)** because:
- Non-linear relationship between features and price
- Handles mixed feature types (area, bedrooms, location, year)
- Robust to outliers
- Consistently top-performing on tabular data

**Algorithm steps for Linear Regression** (simpler baseline):
1. **Collect features**: area, number of rooms, location, age, previous prices
2. **Model**: price = w₁×area + w₂×rooms + w₃×age + ... + b
3. **Loss function**: MSE = (1/n) × Σ(predicted - actual)²
4. **Optimization**: Gradient descent — update weights: w = w - learning_rate × ∂MSE/∂w
5. **Evaluation**: R² score, RMSE, MAE
6. Repeat until loss converges"

## 7. "What is dimensionality reduction? Does it crop/cut data?"

**Answer:**
"**No, it does NOT crop or cut data.** It **transforms** data by projecting it onto a lower-dimensional space.

Think of it like a shadow — a 3D object's shadow on a wall is 2D. It captures the main shape but loses some detail.

I used **TruncatedSVD** (similar to PCA but works on sparse matrices). It decomposes the TF-IDF matrix: X ≈ U × Σ × Vᵀ, keeps only the top k singular values (k=150).

My 30,000 TF-IDF features became 150 components = 99.5% reduction in dimensionality.

**What reduces?** The number of features/dimensions. All data points are preserved — none are removed. The 150 new features are linear combinations of the original 30,000 that capture maximum variance."

---

# SECTION C: PREDICTED NEW QUESTIONS FOR YOUR NOTEBOOK

These are questions that Lvl2_74 hasn't asked before but COULD ask based on your specific notebook:

## EDA-Specific Questions (he spends 30 min here)

### Q: "Why did you plot the label distribution? What does it show?"
**A:** "I plotted it to check for class imbalance. Class 0 has 114K samples (57.7%) while class 3 has only 5.5K (2.8%) — a 20:1 ratio. This told me I need balanced class weights and F1-macro instead of accuracy."

### Q: "What do you infer from the comment length distribution?"
**A:** "Different classes have different comment lengths. Class 1 (toxic) tends to have longer comments (mean ~336 chars) while class 3 (severe) has shorter ones (~194 chars). This confirms that comment length is a useful predictive feature."

### Q: "What does your correlation heatmap tell you?"
**A:** "It shows the linear relationships between numeric features. High correlation between features means redundancy. For example, upvote and log_upvote are highly correlated (expected, since one is derived from the other), but I keep both because tree models handle it well. It also helped identify which features have some correlation with the target."

### Q: "Why did you plot a crosstab of race vs label?"
**A:** "To understand how demographic features relate to the comment category. I found that comments annotated with certain races are disproportionately in class 1 (toxic). This insight validates that demographic features carry predictive signal."

## Model-Specific Questions

### Q: "Why does LightGBM work best for this problem?"
**A:** "Three reasons: (1) It handles class imbalance natively via class_weight='balanced', (2) It captures non-linear interactions between text features and metadata, and (3) Its histogram-based splitting is efficient with our 470-dimensional SVD features. Combined with 3-fold CV and threshold tuning, it achieves our best F1-macro."

### Q: "Why did Naive Bayes perform so poorly?"
**A:** "Gaussian NB assumes features are normally distributed and independent. Our SVD features violate both assumptions — they're orthogonal but not Gaussian, and text features have complex dependencies. NB works better with raw count features."

### Q: "What is the role of class_weight='balanced'?"
**A:** "It automatically adjusts the weight of each class inversely proportional to its frequency. So class 3 (2.8% of data) gets ~20x more weight than class 0 (57.7%). This prevents the model from just predicting the majority class."

### Q: "Why did you use 3-fold CV instead of 5 or 10?"
**A:** "Trade-off between bias, variance, and computation time. On Kaggle with CPU-only and a 4-hour time limit, 3 folds is the optimal balance. Each fold trains on 66% of data (still statistically robust) while keeping total training time manageable."

### Q: "What is early_stopping in LightGBM?"
**A:** "It monitors validation loss and stops training when it hasn't improved for N rounds (I used N=40). This prevents overfitting — the model uses only the first `best_iteration_` trees, not all 1500. It's like automatic regularization."

### Q: "Why do you combine word n-grams and character n-grams in TF-IDF?"
**A:** "Word n-grams (1,2) capture standard bigrams like 'hate speech'. Character n-grams (3,5) capture subword patterns like misspellings, l33tspeak, and partial words — important for toxic text where people deliberately misspell slurs."

### Q: "What is sublinear_tf=True?"
**A:** "Instead of raw term frequency, it uses TF = 1 + log(TF). This dampens the effect of very frequent terms. A word appearing 100 times isn't 100x more important than one appearing once — log scaling reflects this."

### Q: "What is norm='l2' in TfidfVectorizer?"
**A:** "It normalizes each document's TF-IDF vector to unit length (L2 norm = 1). This ensures documents of different lengths are comparable — a 10-word comment and a 1000-word comment get the same vector magnitude."

---

# SECTION D: COMPLETE THEORY Q&A (Every concept in your notebook)

## Overfitting vs Underfitting

### Q: "What is overfitting? How to prevent it?"
**A:** "Overfitting = model memorizes training data, fails on new data. Signs: high train accuracy, low val accuracy. Prevention:
1. **Regularization** (L1/L2 penalty)
2. **Early stopping** (stop training when val loss plateaus)
3. **Cross-validation** (evaluate on multiple folds)
4. **Reduce model complexity** (fewer trees, fewer leaves)
5. **More training data**
6. **Dropout** (for neural networks)
7. **Subsample/colsample** (use random subset of data/features per tree)"

### Q: "What is underfitting?"
**A:** "Model is too simple to capture patterns. Signs: low train AND low val accuracy. Fix: increase model complexity, more features, remove regularization."

## Bias-Variance Tradeoff
**A:** "Bias = error from wrong assumptions (underfitting). Variance = error from sensitivity to training data (overfitting). Total error = Bias² + Variance + Irreducible noise. You want the sweet spot where both are moderate."

## Bagging vs Boosting

### Q: "Difference between bagging and boosting?"
**A:**
| | Bagging | Boosting |
|---|---|---|
| Strategy | Train models in **parallel** on random subsets | Train models **sequentially**, each corrects errors |
| Example | Random Forest | XGBoost, LightGBM |
| Reduces | **Variance** (overfitting) | **Bias** (underfitting) |
| Data sampling | Bootstrap (random with replacement) | Weighted (focus on hard examples) |
| Overfitting risk | Low | Higher (need regularization) |

## Decision Tree vs Random Forest
**A:** "Decision tree = single tree, prone to overfitting. Random Forest = ensemble of 100+ trees trained on bootstrap samples with random feature subsets. Averaging reduces variance."

## Cross-Validation

### Q: "What is cross-validation?"
**A:** "K-Fold CV splits data into K parts. Train on K-1, validate on 1, repeat K times. Every sample is validated exactly once. Average score is more reliable than a single train/test split."

### Q: "What is StratifiedKFold?"
**A:** "Same as KFold but preserves class distribution. Essential for imbalanced data — ensures each fold has proportional samples of every class."

## Logistic Regression — Deep Dive

### Q: "How does Logistic Regression work internally?"
**A:**
1. Compute linear sum: z = w₁x₁ + w₂x₂ + ... + b
2. Apply sigmoid: P(y=1) = 1/(1+e^(-z))
3. For multi-class: softmax across all classes
4. Loss: Cross-entropy = -Σ[y·log(p) + (1-y)·log(1-p)]
5. Optimize with gradient descent (or LBFGS)

### Q: "What is the difference between parameters and hyperparameters?"
**A:**
- **Parameters**: Learned during training. Weights w₁, w₂, etc. in LogReg
- **Hyperparameters**: Set before training. C, learning_rate, n_estimators. Tuned via GridSearchCV

## Loss Functions

### Q: "What is a loss function?"
**A:** "A loss function measures how wrong the model's predictions are. Training = minimizing the loss.
- **Log loss / Cross-entropy**: For classification (LogReg, MLP)
- **Hinge loss**: For SVM — max(0, 1 - y·f(x))
- **MSE**: For regression
- LightGBM uses **log loss** for classification internally"

## Neural Networks

### Q: "What is an activation function? Name some."
**A:** "Non-linear function applied after each layer to introduce non-linearity. Without it, stacking layers = same as one layer.
- **ReLU**: max(0, x). Input: any real number. Output: [0, ∞). Most common.
- **Sigmoid**: 1/(1+e^(-x)). Input: any real. Output: (0, 1). For binary classification.
- **Tanh**: (e^x - e^(-x))/(e^x + e^(-x)). Input: any real. Output: (-1, 1).
- **Softmax**: Converts logits to probabilities summing to 1. For multi-class output layer."

## Encoding Methods

### Q: "OneHotEncoding vs LabelEncoding?"
**A:**
- **OneHotEncoding**: Creates binary columns. [Red, Blue, Green] → [1,0,0], [0,1,0], [0,0,1]. No ordinal relationship assumed.
- **LabelEncoding**: Assigns integers. Red=0, Blue=1, Green=2. Assumes order (bad for nominal data, okay for ordinal).
- I used OHE because race/religion/gender are nominal (no inherent order).

### Q: "What is handle_unknown='ignore' in OHE?"
**A:** "If test data has a category not seen in training, instead of crashing with an error, it creates an all-zeros row. Safe for production."

## Regularization

### Q: "What is L1 and L2 regularization?"
**A:**
- **L1 (Lasso)**: Adds |w| to loss. Pushes weights to exactly 0 → feature selection.
- **L2 (Ridge)**: Adds w² to loss. Shrinks weights toward 0 but doesn't zero them → prevents overfitting.
- **ElasticNet**: Combination of L1+L2.
- In LogReg: `penalty='l2'` with C=5.0 (C = 1/λ, so lower C = more regularization)

## Precision, Recall, F1

### Q: "Define precision, recall, F1, support"
**A:**
- **Precision** = TP/(TP+FP) → "Of all I predicted positive, how many actually are?"
- **Recall** = TP/(TP+FN) → "Of all actual positives, how many did I catch?"
- **F1** = 2×P×R/(P+R) → Harmonic mean. Both must be high for F1 to be high.
- **Support** = Number of actual instances of that class in the data.
- **Macro avg** = Simple average of per-class metrics. Treats all classes equally.
- **Weighted avg** = Average weighted by support. Gives more weight to larger classes.

## Data Leakage

### Q: "What is data leakage?"
**A:** "When information from the test set leaks into training. Examples:
- Fitting scaler on full data before splitting → scaler knows test distribution
- Using target variable for feature engineering on test data
I avoid it by: fit_transform on train, only transform on test."

---

# SECTION E: CODING QUESTIONS (Common in L2 Vivas)

Even though Lvl2_74 rarely asks coding, other proctors do. If proctor changes last minute, here are the top coding tasks:

## 1. Write a preprocessing pipeline with ColumnTransformer
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_cols = ['upvote', 'downvote', 'char_len', 'word_count']
cat_cols = ['race', 'religion', 'gender']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Usage:
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

## 2. Train test split and print shapes
```python
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('train.csv')
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")
```

## 3. GridSearchCV for any model
```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'saga']}
gs = GridSearchCV(
    LogisticRegression(max_iter=500, random_state=42),
    param_grid=param_grid,
    scoring='f1_macro',
    cv=3,
    verbose=1
)
gs.fit(X_train, y_train)
print(f"Best params: {gs.best_params_}")
print(f"Best score: {gs.best_score_:.4f}")
```

## 4. Confusion matrix with visualization
```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred, digits=4))
```

## 5. Separate numerical and categorical columns
```python
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Numerical: {num_cols}")
print(f"Categorical: {cat_cols}")
```

## 6. Train LightGBM with early stopping
```python
import lightgbm as lgb
from sklearn.metrics import f1_score

model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.1, num_leaves=31,
    class_weight='balanced', random_state=42, verbose=-1)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False)])

preds = model.predict(X_val)
print(f"F1-macro: {f1_score(y_val, preds, average='macro'):.4f}")
print(f"Best iteration: {model.best_iteration_}")
```

## 7. Train XGBoost
```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=6,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42)
xgb.fit(X_train, y_train)
preds = xgb.predict(X_val)
```

## 8. TF-IDF vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(train['comment'])  # fit on train only!
X_test_tfidf = tfidf.transform(test['comment'])         # transform only!
```

## 9. Create a toy dataset and replace '?' with NaN
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, '?', 3, None, 5],
    'B': ['x', 'y', '?', 'z', None],
    'C': [10, 20, 30, '?', 50]
})

df = df.replace('?', np.nan)
print(df)
print(df.isnull().sum())
```

## 10. RFE (Recursive Feature Elimination)
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    n_features_to_select=2
)
rfe.fit(X_train, y_train)
print(f"Selected features: {X_train.columns[rfe.support_].tolist()}")
print(f"Feature ranking: {rfe.ranking_}")
```

## 11. Load sklearn dataset, train DecisionTree, plot it
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
print(f"Accuracy: {dt.score(X_test, y_test):.4f}")

plt.figure(figsize=(15,8))
plot_tree(dt, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True)
plt.show()
```

## 12. KMeans clustering
```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(range(1, 11), inertias, 'o-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

## 13. Write F1-macro from scratch
```python
def f1_macro(y_true, y_pred):
    classes = sorted(set(y_true))
    f1s = []
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f1)
    return sum(f1s) / len(f1s)
```

---

# SECTION F: TRAP QUESTIONS & TRICKY FOLLOW-UPS

## "Why learning rate=0.01, 0.1 and not 1, 2, 3?"
**A:** "Learning rate controls step size. If LR=1 or 2, the optimization overshoots the minimum — like taking huge jumps and missing the valley. Small LR (0.01-0.1) takes small careful steps that converge smoothly. Too small = very slow learning, too large = divergence."

## "If model is overfitting, increase or decrease n_estimators?"
**A:** "**Decrease** (or use early stopping). More trees = more complex model = more overfitting. Early stopping is even better — it automatically finds the right number."

## "If model is overfitting, increase or decrease learning_rate?"
**A:** "**Decrease.** Lower learning rate = each tree contributes less = more regularization. But you'll need more trees to compensate, so pair it with early stopping."

## "Is Random Forest parametric or non-parametric?"
**A:** "**Non-parametric.** It doesn't assume any fixed functional form. The model structure (tree splits) grows with the data."

## "What is 'support' in classification report?"
**A:** "The number of actual occurrences of each class in the test/validation set. For class 3, support might be 1100 — meaning there are 1100 class-3 samples in validation."

## "What is verbose in model training?"
**A:** "Controls how much output the model prints during training. verbose=0: silent, verbose=1: progress updates, verbose=-1 (LightGBM): suppress all warnings."

## "What is n_jobs=-1?"
**A:** "Use all available CPU cores for parallel processing. n_jobs=1: single core, n_jobs=-1: all cores."

## "Is this supervised or unsupervised learning?"
**A:** "**Supervised classification.** We have labeled training data (comments with known categories 0-3). We learn a mapping from features to labels."

## "Name 3 unsupervised algorithms"
**A:** "KMeans clustering, DBSCAN, PCA (for dimensionality reduction)"

## "What is the difference between XGBoost and Linear Regression?"
**A:** "Completely different:
- **Linear Regression** is a linear model for regression. Assumes linear relationship: y = wx+b
- **XGBoost** is gradient boosted decision trees for classification/regression. Non-linear, handles complex interactions, uses ensemble of weak learners"

---

# SECTION G: YOUR NOTEBOOK — QUICK REFERENCE CARD

| Cell | Topic | Key Points to Mention |
|------|-------|----------------------|
| **1** | Imports | "I use sklearn, LightGBM, XGBoost — all allowed libraries" |
| **3** | Data Loading | "198K train, 102K test, 4 classes, heavily imbalanced" |
| **5** | EDA | "Class imbalance, comment length by label, correlation heatmap" |
| **7-8** | Cleaning + FE | "fillna, caps_ratio, word_count, interaction features" |
| **10-11** | Preprocessing | "TF-IDF (word+char), StandardScaler, OHE, TruncatedSVD (30K→150)" |
| **13** | LogReg + SGD | "Tuned C values with balanced class weights" |
| **15** | NB, KNN, SVM | "NB poor (independence assumption), SVM good with linear kernel" |
| **17** | LightGBM + XGB | "Best performers. LGBM: leaf-wise, histogram splits, GOSS" |
| **19** | MLP | "Neural net with ReLU, Adam optimizer, early stopping" |
| **21** | 🆕 Model Comparison | "Bar chart comparing F1-macro of all 8 models, confusion matrices" |
| **23** | 🆕 HP Tuning | "GridSearchCV on 3 models with f1_macro, bar chart of results" |
| **25** | Final Pipeline | "3-fold CV LightGBM + F1-macro threshold tuning" |
| **27** | Submission | "102K predictions, 4-class distribution, saved to CSV" |

---

# SECTION H: LAST-MINUTE CHECKLIST

- [ ] Can you explain EVERY graph in your notebook and WHY you plotted it?
- [ ] Can you define: preprocessing, feature engineering, feature scaling, HPT, dimensionality reduction?
- [ ] Can you explain: TF-IDF, TruncatedSVD, LogReg, LightGBM, XGBoost, MLP?
- [ ] Can you answer the house price prediction question?
- [ ] Can you explain random_state=42 WITHOUT saying Hitchhiker's Guide?
- [ ] Can you differentiate: GridSearchCV vs RandomizedSearchCV?
- [ ] Can you explain why accuracy is bad for imbalanced data?
- [ ] Can you write: Pipeline + ColumnTransformer code from memory?
- [ ] Can you write: GridSearchCV code from memory?
- [ ] Can you explain confusion matrix and identify TP/FP/FN/TN?
- [ ] Do you know: bagging vs boosting, overfitting vs underfitting, bias vs variance?
- [ ] Can you explain what class_weight='balanced' does and why you use it?

> **🏆 Golden Rule from the student who got 47/50:** *"All depends on how well u speak about questions he asks. I just answered everything confidently — he didn't ask any added question. Simple only. Just prepare gform questions before viva."*
