# 🎯 COMPLETE VIVA PREPARATION FOR PROCTOR 118 (LEVEL 1)
## Comment Category Prediction Challenge - Your Roll: 23f2002518

---

# 📋 TABLE OF CONTENTS
1. [Proctor 118 Psychology & Pattern](#proctor-118-pattern)
2. [Your 20-Minute Notebook Presentation Script](#presentation-script)
3. [Every Feature Engineering Explained](#feature-engineering)
4. [Pipeline Structure Deep Dive](#pipeline-structure)
5. [All Models + Hyperparameters](#models-and-hyperparameters)
6. [Correlation & Scaling Questions](#correlation-and-scaling)
7. [IRIS Dataset Coding Question](#iris-dataset)
8. [California Housing Dataset](#california-housing)
9. [Common Theory Questions](#theory-questions)
10. [Emergency Answers](#emergency-answers)

---

# 🧠 PROCTOR 118 PATTERN <a name="proctor-118-pattern"></a>

## What They ALWAYS Ask:
Based on ALL entries for proctor 118:

1. **"Present your notebook" (20 mins)** - EVERY SINGLE TIME
2. **Feature Engineering** - What did you create and WHY?
3. **Correlation** - Which correlations did you use?
4. **Scaling** - Why StandardScaler? Why not others?
5. **Iris Dataset coding** - Load it, show shape, feature matrix
6. **GridSearch vs RandomSearch** - Which is faster?
7. **Model-specific questions** - RF vs XGBoost difference
8. **Parametric vs Non-parametric** - Which models are which?

## Proctor Personality:
- **"Sweetest guy ever"** - multiple students say this
- Patient, doesn't interrupt
- Notes down discrepancies, gives suggestions at END
- Will ask about YOUR choices, not random theory
- If you explain well for 20 min, FEWER questions

## Golden Rule:
**Take your time explaining! Fill 20 minutes with YOUR explanation = fewer questions at end.**

---

# 🎤 YOUR 20-MINUTE PRESENTATION SCRIPT <a name="presentation-script"></a>

## Opening (30 seconds):
> "This is a multi-class text classification problem where we predict comment categories into 4 classes: 0, 1, 2, and 3. The dataset has 198,000 training samples and 102,000 test samples. Our evaluation metric is F1-macro score, and I achieved approximately 0.83."

## Milestone 1: Data Loading & EDA (3 minutes)

> "I started by loading the data and doing exploratory analysis. The key findings were:
>
> 1. **Class Imbalance**: Class 0 is 57.66%, Class 2 is 31.54%, Class 1 is 8.04%, and Class 3 is only 2.76%. This is highly imbalanced, which is why I used class_weight='balanced' in all my models.
>
> 2. **Missing Values**: The demographic columns (race, religion, gender) have about 73% missing values. Instead of dropping these columns, I filled them with 'unknown' and created a binary feature `has_demo_info` to capture whether demographic information was present.
>
> 3. **Text Length Varies**: Comments range from empty to very long. I engineered features like `char_len`, `word_count` to capture this.
>
> 4. **Numerical Features**: We have `upvote`, `downvote`, `if_1`, `if_2` which are continuous. I analyzed their distributions and found they're skewed, so I also created log-transformed versions."

## Milestone 2: Preprocessing (3 minutes)

> "For preprocessing, I did the following:
>
> 1. **Missing Comments**: Filled with empty string (not dropable since comment is our main feature)
>
> 2. **Date Parsing**: Extracted `hour`, `dayofweek`, `month` from `created_date` then dropped the original column
>
> 3. **Categorical Handling**: For race, religion, gender - I filled missing with 'unknown' and kept as categorical for one-hot encoding. This is because values like 'black', 'asian', 'buddhist' have no ordinal (numeric order) relationship.
>
> 4. **Disability**: Converted True/False strings to binary 0/1
>
> 5. **Numeric Columns**: Used `pd.to_numeric` with `errors='coerce'` and filled NaN with 0"

## Milestone 3: Feature Engineering (3 minutes)

> "I created 30+ engineered features in these categories:
>
> **Text Features:**
> - `char_len`: Total characters in comment
> - `word_count`: Number of words
> - `avg_word_len`: Average word length (longer words may indicate formal language)
> - `unique_word_ratio`: Vocabulary diversity (unique words / total words)
> - `caps_ratio`: Proportion of uppercase letters (indicates shouting/emphasis)
>
> **Stylistic Features:**
> - `exclam_count`, `question_count`: Punctuation patterns
> - `caps_word_count`: Words that are ALL CAPS
> - `has_url`: Binary flag for URLs
>
> **Vote Features:**
> - `vote_total`: upvote + downvote
> - `upvote_ratio`: upvote / (upvote + downvote + 1)
> - `log_upvote`, `log_downvote`: Log-transformed to handle skewness
>
> **Critical Feature:**
> - `has_demo_info`: 1 if race/religion/gender is NOT unknown. This is VERY predictive because Class 1 has 83% demographic data filled while other classes have only ~20%."

## Milestone 4: Pipeline (3 minutes)

> "I used sklearn's `ColumnTransformer` to combine multiple preprocessing steps:
>
> 1. **Text Processing (TF-IDF + SVD)**:
>    - Word TF-IDF: max_features=30000, ngram_range=(1,2), then SVD to 150 dimensions
>    - Character TF-IDF: ngram_range=(3,5), then SVD to 80 dimensions
>    - **Why SVD?** TF-IDF creates 30K+ sparse features. SVD (Singular Value Decomposition) reduces dimensionality while preserving most variance. This speeds up training significantly.
>
> 2. **Numeric Scaling (StandardScaler)**:
>    - Applied to engineered and raw numeric features
>    - StandardScaler transforms to mean=0, std=1
>    - **Why StandardScaler?** Because algorithms like Logistic Regression and SGD use gradient descent, which converges faster with standardized features.
>
> 3. **Categorical (OneHotEncoder)**:
>    - `handle_unknown='ignore'` means if test data has a category not seen in training, it becomes all zeros instead of error
>
> 4. **Binary (passthrough)**:
>    - disability is already 0/1, no transformation needed"

## Milestone 5: Models & Hyperparameter Tuning (4 minutes)

> "I tried multiple models:
>
> **1. Logistic Regression:**
> - Hyperparameter tuned: C (regularization strength)
> - Tested C = 0.1, 1.0, 5.0
> - Lower C = more regularization = simpler model
> - Best: C=1.0
>
> **2. SGD Classifier:**
> - Loss = 'modified_huber' (robust to outliers, gives probabilities)
> - Tuned alpha (regularization): 1e-4, 1e-3, 1e-2
> - SGD = Stochastic Gradient Descent = updates weights after each sample, not entire batch
>
> **3. LightGBM (Best Model):**
> - Gradient boosting algorithm
> - Key hyperparameters:
>   - `n_estimators=1200`: Number of trees
>   - `learning_rate=0.10`: How much each tree contributes (higher = faster but may overfit)
>   - `num_leaves=31`: Complexity of each tree
>   - `max_depth=8`: Prevents overfitting by limiting tree depth
>   - `class_weight='balanced'`: Adjusts weights inversely proportional to class frequency
>   - `early_stopping_rounds=40`: Stop if no improvement for 40 rounds
>
> **4. XGBoost:**
> - Similar to LightGBM but different implementation
> - Used sample_weight for class imbalance
>
> **Why LGBM was best?** It handles imbalanced data well, is fast due to histogram-based splitting, and natively supports categorical features."

## Milestone 6: Final Pipeline & Threshold Tuning (3 minutes)

> "For the final submission:
>
> **3-Fold Cross-Validation:**
> - Split training data into 3 parts
> - Train on 2 parts, validate on 1, rotate
> - This gives more stable estimates than single train-test split
>
> **Out-of-Fold (OOF) Predictions:**
> - Each sample is predicted when it's in the validation fold
> - Gives unbiased predictions for all training data
>
> **Threshold Tuning (KEY TO HIGH SCORE):**
> - Default prediction: argmax of probabilities
> - But since classes are imbalanced, we can adjust class weights
> - Used Nelder-Mead optimization to find optimal weights
> - This boosted F1-macro by ~0.07-0.08!
>
> **Final F1-macro: approximately 0.83**"

## Closing (30 seconds):
> "In summary, the key to my score was:
> 1. Good feature engineering, especially `has_demo_info`
> 2. Using class_weight='balanced' in all models
> 3. Threshold tuning for F1-macro optimization
> 4. LightGBM as the final model
>
> Any questions?"

---

# 🔧 FEATURE ENGINEERING EXPLAINED <a name="feature-engineering"></a>

| Feature | What It Is | Why It Matters |
|---------|-----------|----------------|
| `char_len` | Number of characters in comment | Short comments might be spam, long might be detailed feedback |
| `word_count` | Number of words | Similar to char_len but ignores punctuation |
| `avg_word_len` | Average word length | Longer words = more formal vocabulary |
| `unique_word_ratio` | Unique words / Total words | Low ratio = repetitive text |
| `caps_ratio` | % of uppercase letters | ALL CAPS = shouting, possibly toxic |
| `exclam_count` | Number of `!` | Excitement or anger |
| `has_url` | Contains http/www? | Spam often has links |
| `vote_total` | upvote + downvote | Engagement level |
| `upvote_ratio` | upvote / (upvote + downvote + 1) | Quality indicator |
| `has_demo_info` | Is race/religion/gender NOT unknown? | **MOST IMPORTANT** - Class 1 has 83% filled! |
| `log_upvote` | log(upvote + 1) | Handles skewness (few comments have 1000s of votes) |
| `if_interact` | if_1 * if_2 | Interaction between two features |

---

# 🔀 PIPELINE STRUCTURE <a name="pipeline-structure"></a>

Your pipeline has this structure:

```
ColumnTransformer:
│
├── word_tfidf: Pipeline
│   └── TfidfVectorizer (30K features, ngram 1-2)
│   └── TruncatedSVD (reduce to 150 dims)
│
├── char_tfidf: Pipeline
│   └── TfidfVectorizer (20K features, char 3-5 grams)
│   └── TruncatedSVD (reduce to 80 dims)
│
├── eng_num: StandardScaler
│   └── Applied to engineered numeric features (30+ features)
│
├── raw_num: StandardScaler
│   └── Applied to raw numeric features (10 features)
│
├── cat: OneHotEncoder
│   └── Applied to categorical columns (post_id, race, religion, gender)
│
└── bin: passthrough
    └── disability (already 0/1)
```

**What is a Pipeline?**
> A Pipeline chains multiple transformations together. When you call `.fit_transform()`, it applies each step in sequence. This ensures the same transformations are applied to both training and test data.

**What is ColumnTransformer?**
> ColumnTransformer applies different transformations to different columns. Text columns get TF-IDF, numeric columns get scaling, categorical columns get one-hot encoding.

**Why sparse_threshold=0?**
> Forces output to be dense (numpy array) instead of sparse matrix. Some models don't work with sparse matrices.

---

# 🤖 ALL MODELS + HYPERPARAMETERS <a name="models-and-hyperparameters"></a>

## Logistic Regression

```python
LogisticRegression(
    C=1.0,              # Regularization strength (inverse)
    solver="lbfgs",     # Optimization algorithm
    penalty="l2",       # L2 regularization (Ridge)
    max_iter=500,       # Maximum iterations
    class_weight="balanced",  # Adjust for imbalance
)
```

| Parameter | What It Does | Impact |
|-----------|--------------|--------|
| `C` | Inverse regularization | Lower C = more regularization = simpler model, prevents overfitting |
| `solver` | Optimization algorithm | lbfgs is good for multiclass, small-medium datasets |
| `penalty` | Type of regularization | L2 shrinks weights, L1 can zero out features |
| `max_iter` | Max iterations | More iterations if not converging |
| `class_weight` | Class weights | "balanced" = minority classes get higher weight |

**If asked "Why lbfgs?"**
> LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a quasi-Newton method that's efficient for medium datasets and supports multiclass natively.

## SGD Classifier

```python
SGDClassifier(
    loss="modified_huber",
    alpha=1e-3,
    max_iter=500,
    class_weight="balanced"
)
```

| Parameter | What It Does |
|-----------|--------------|
| `loss` | Loss function. "modified_huber" is like hinge loss but smooth, gives probabilities |
| `alpha` | Regularization constant. Higher = more regularization |

**Why SGD?**
> Stochastic Gradient Descent updates weights after each sample (or mini-batch), making it very fast for large datasets. Good for online learning.

## LightGBM (YOUR BEST MODEL)

```python
LGBMClassifier(
    n_estimators=1200,      # Number of boosting rounds (trees)
    learning_rate=0.10,     # How much each tree contributes
    num_leaves=31,          # Max leaves per tree (complexity)
    max_depth=8,            # Max tree depth
    min_child_samples=50,   # Min samples in a leaf
    subsample=0.8,          # % of rows per tree
    colsample_bytree=0.3,   # % of columns per tree
    reg_alpha=0.1,          # L1 regularization
    reg_lambda=0.5,         # L2 regularization
    class_weight="balanced",
    early_stopping_rounds=40,
)
```

| Parameter | Impact |
|-----------|--------|
| `n_estimators` | More trees = better fit but slower. Early stopping finds optimal. |
| `learning_rate` | Lower = need more trees, but more stable. Higher = faster training but might overfit. |
| `num_leaves` | 2^depth for balanced tree. More leaves = more complex model. |
| `max_depth` | Limits tree depth. Prevents overfitting. |
| `min_child_samples` | Min samples to create a leaf. Higher = prevents fitting noise. |
| `subsample` | Row sampling. 0.8 means each tree sees 80% of data. Prevents overfitting. |
| `colsample_bytree` | Feature sampling. 0.3 means each tree sees 30% of features. |
| `reg_alpha` | L1 regularization on leaf weights |
| `reg_lambda` | L2 regularization on leaf weights |

**If asked "What is Gradient Boosting?"**
> Gradient Boosting builds trees sequentially. Each new tree tries to correct the errors of the previous trees. It minimizes a loss function using gradient descent in function space.

**If asked "Difference between Random Forest and XGBoost/LGBM?"**
> - Random Forest: Builds trees in PARALLEL, each independent, averages predictions (bagging)
> - XGBoost/LGBM: Builds trees SEQUENTIALLY, each tree corrects previous errors (boosting)
> - Boosting often achieves higher accuracy but can overfit more easily

## XGBoost

```python
XGBClassifier(
    n_estimators=600,
    learning_rate=0.08,
    max_depth=6,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    gamma=0.1,
    eval_metric='mlogloss',
    early_stopping_rounds=40,
)
```

| Parameter | What It Does |
|-----------|--------------|
| `gamma` | Minimum loss reduction to make a split. Higher = more conservative |
| `min_child_weight` | Minimum sum of instance weights in a child. Similar to min_samples |
| `eval_metric='mlogloss'` | Multiclass log loss for evaluation |

---

# 📊 CORRELATION & SCALING <a name="correlation-and-scaling"></a>

## "What correlation did you use?"

> "I used Pearson correlation for numeric features. For example, I looked at the correlation between `upvote` and `downvote` (positive correlation - controversial posts get both). I also checked correlation of `if_1` and `if_2` with the target labels to identify predictive features."

**If asked "What about categorical correlation?"**
> "For categorical variables like race, religion, I would use Cramér's V or Chi-square test, not Pearson correlation which is for continuous variables."

## "Why StandardScaler?"

**Simple answer:**
> "StandardScaler transforms features to mean=0 and standard deviation=1. This is important because:
> 1. Algorithms using gradient descent (Logistic Regression, SGD) converge faster when features are on similar scales
> 2. It prevents features with larger values from dominating
> 3. For example, `upvote` might be 0-1000 while `caps_ratio` is 0-1. Without scaling, upvote would dominate."

**If asked "Why not MinMaxScaler?"**
> "MinMaxScaler scales to [0,1] range. It's sensitive to outliers because a single extreme value compresses all other values. StandardScaler is more robust because outliers just become large z-scores, but don't affect others as much."

**If asked "Why not RobustScaler?"**
> "RobustScaler uses median and IQR (Interquartile Range), making it even more robust to outliers. I could have used it, but StandardScaler is sufficient for this dataset and more commonly used."

---

# 🌸 IRIS DATASET CODING <a name="iris-dataset"></a>

**This is asked frequently by proctor 118!**

## Basic Code:
```python
from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset
iris = load_iris()

# Feature matrix (X)
X = iris.data
print("Feature matrix shape:", X.shape)  # (150, 4)
print("Feature matrix type:", type(X))   # numpy.ndarray

# Target variable (y)
y = iris.target
print("Target shape:", y.shape)  # (150,)

# Feature names
print("Features:", iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Target names
print("Classes:", iris.target_names)
# ['setosa', 'versicolor', 'virginica']
```

## Key Facts:
- **150 samples, 4 features, 3 classes**
- **50 samples per class** (perfectly balanced!)
- Features are all NUMERIC (length/width measurements)
- Target is 0, 1, 2 (three flower species)

## If asked to create a DataFrame:
```python
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print(df.head())
```

## If asked to do classification:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(kernel='rbf')
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
```

---

# 🏠 CALIFORNIA HOUSING DATASET <a name="california-housing"></a>

**Backup in case asked!**

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load
housing = fetch_california_housing()

X = housing.data
y = housing.target

print("Shape:", X.shape)  # (20640, 8)
print("Features:", housing.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

print("Target: Median house value (in $100,000s)")
```

**Key Facts:**
- **20,640 samples, 8 features**
- **REGRESSION problem** (predicting continuous value)
- Target is median house value in $100,000
- Features include income, house age, location

---

# ❓ THEORY QUESTIONS <a name="theory-questions"></a>

## "GridSearchCV vs RandomizedSearchCV - Which is faster?"

**Answer:**
> "RandomizedSearchCV is FASTER.
> - GridSearchCV tries ALL combinations of parameters. If you have 3 parameters with 10 values each, that's 10×10×10 = 1000 combinations.
> - RandomizedSearchCV randomly samples a fixed number of combinations (set by `n_iter`). If n_iter=100, it only tries 100 combinations.
> - GridSearchCV is exhaustive but slow. RandomizedSearchCV is faster and often finds good hyperparameters."

## "What is F1-Score?"

**Formula:**
> F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Simple explanation:**
> "F1-score is the harmonic mean of Precision and Recall. It's useful when you want to balance both.
> - Precision = Of all predicted positives, how many are actually positive?
> - Recall = Of all actual positives, how many did we predict?
> - F1 balances both. If either is low, F1 is low."

## "Why F1-macro?"

> "F1-macro calculates F1 for each class separately, then takes the unweighted average. This treats all classes equally, which is important for imbalanced datasets. If we used accuracy, we could get 80%+ by just predicting the majority class."

## "What is Cross-Validation?"

> "Cross-validation splits the data into K parts (folds). We train on K-1 folds and validate on 1 fold, rotating which fold is validation. This gives K performance estimates, and we average them. It's more reliable than a single train-test split."

## "Overfitting vs Underfitting?"

> - **Overfitting**: Model learns training data too well, including noise. High training accuracy, low test accuracy. Solution: more regularization, less complex model, more data.
> - **Underfitting**: Model is too simple to capture patterns. Low training AND test accuracy. Solution: more features, more complex model, less regularization.

## "Parametric vs Non-parametric models?"

> - **Parametric**: Fixed number of parameters regardless of data size. Examples: Logistic Regression, Linear Regression, Naive Bayes. Makes assumptions about data distribution.
> - **Non-parametric**: Number of parameters grows with data. Examples: KNN, Decision Trees, Random Forest, LGBM. More flexible, fewer assumptions.

**In your notebook:**
- Parametric: Logistic Regression, SGD
- Non-parametric: LightGBM, XGBoost, Random Forest (Decision Trees)

## "What is a weak learner?"

> "A weak learner is a model that performs only slightly better than random guessing (e.g., >50% accuracy for binary classification). In boosting, we combine many weak learners (simple decision trees) to create a strong learner."

## "Bagging vs Boosting?"

> - **Bagging** (Bootstrap Aggregating): Train multiple models in PARALLEL on random subsets of data. Average their predictions. Example: Random Forest.
> - **Boosting**: Train models SEQUENTIALLY. Each model focuses on the errors of previous models. Example: LGBM, XGBoost, AdaBoost.

---

# 🆘 EMERGENCY ANSWERS <a name="emergency-answers"></a>

**If you don't know something:**
> "I'm not completely sure about that, but I think... [make an educated guess]. I'll definitely look into this after the viva."

**If asked about something not in your notebook:**
> "I didn't implement that in my notebook, but I understand that it works by..."

**If asked to write code you don't know:**
> "Can I refer to the documentation? I know the concept but want to make sure I get the syntax right."

**If asked why you made a choice:**
> "I chose this because... [state benefit]. An alternative would be... [state alternative and its tradeoff]."

---

# 📝 QUICK REFERENCE CARD

## Your Notebook Numbers:
- Training samples: 198,000
- Test samples: 102,000
- Classes: 4 (0, 1, 2, 3)
- Class 0: 57.66%, Class 1: 8.04%, Class 2: 31.54%, Class 3: 2.76%
- Final F1-macro: ~0.83
- Best model: LightGBM with 3-fold CV + threshold tuning

## Your Key Choices:
- Scaling: StandardScaler (gradient descent needs scaled features)
- Missing values: 'unknown' for categorical, 0 for numeric
- Imbalance handling: class_weight='balanced'
- Feature reduction: TruncatedSVD (not PCA, because TF-IDF is sparse)
- Evaluation: F1-macro (treats all classes equally)

## Your Pipeline Components:
1. Word TF-IDF (30K features) → SVD (150 dims)
2. Char TF-IDF (20K features) → SVD (80 dims)
3. StandardScaler on numeric features
4. OneHotEncoder on categorical features
5. Passthrough for binary features

---

# 🔥 RECENT QUESTIONS FROM 2026 (MARCH) <a name="recent-questions"></a>

## March 18-20, 2026 (MOST RECENT!)

### Exact Questions Asked:
1. **"Load iris dataset in colab"** - ASKED EVERY TIME
2. **"Which is faster - GridSearchCV or RandomizedSearchCV?"** - Common question
3. **"Why not accuracy?"** - You need to explain why F1 is better
4. **"Formula of F1-score?"** - Know: 2 × (P × R) / (P + R)
5. **"Is equal weightage given to recall and precision in F1?"** - YES, harmonic mean
6. **"Why F1-macro, not others?"** - Treats all classes equally
7. **"Which models are parametric?"** - LogReg, SGD are parametric; LGBM, trees are non-parametric
8. **"Load iris, show shape of feature matrix, datatype"** - See Iris section above

### Pattern Observed:
- **Always** asks for ID card first
- **20-25 minutes** to explain notebook
- **1-2 questions** at the end (theory + coding)
- **Iris dataset coding** is almost guaranteed
- **Very friendly**, gives tips for Level 2

---

# ⚠️ CRITICAL QUESTIONS YOU MUST KNOW

## "Why not accuracy?"

**Answer:**
> "Accuracy can be misleading for imbalanced datasets. In my dataset, Class 0 is 57.66%. If I just predicted Class 0 for everything, I'd get 57% accuracy! But my F1-score would be terrible because I'd miss all other classes. F1-macro treats all classes equally, which is fair for imbalanced data."

## "If two columns have 0 correlation, will they be collinear?"

**Answer:**
> "No, columns with 0 correlation are NOT collinear. 
> - **Collinearity** means one column can be expressed as a linear combination of others (they move together).
> - **0 correlation** means no LINEAR relationship between the columns.
> - However, there could still be a NON-LINEAR relationship! For example, X and X² have 0 Pearson correlation but are clearly related."

## "min_samples_split and min_samples_leaf question"

**If asked "If a node has 7 samples, would it split? (with min_samples_split=10, min_samples_leaf=5)":**

> "It depends on the parameters:
> - `min_samples_split=10`: Node needs at least 10 samples to consider splitting. If node has 7, it WON'T split.
> - `min_samples_leaf=5`: Each resulting leaf needs at least 5 samples.
> 
> With 7 samples and min_samples_split=10: NO, it won't split (7 < 10).
> 
> With 7 samples and min_samples_leaf=5: NO, because after split you can't have both children with 5+ samples (7 = 2 + 5 or 3 + 4, etc., at least one child would have < 5)."

## "What is your baseline model?"

> "My baseline was Logistic Regression with default parameters. It gave F1-macro of approximately 0.60-0.65. This established a minimum acceptable performance. My final LightGBM model improved this to ~0.83, showing significant improvement over baseline."

## "Why did you use only boosting models?"

> "I actually tried both bagging and boosting models. I tested Logistic Regression, SGD (linear models), and then LightGBM and XGBoost (boosting). Boosting models performed best because they iteratively correct errors and handle imbalanced data well with class_weight='balanced'."

---

# 🎯 FINAL TIPS FOR PROCTOR 118

1. **Take your time** - Fill 20 minutes with your explanation
2. **Explain WHY** - Not just what you did, but why
3. **Mention class imbalance** - It's key to your approach
4. **Know `has_demo_info`** - Your most important engineered feature
5. **Practice Iris dataset loading** - He asks this EVERY TIME!
6. **Be honest** - If you don't know, say so politely
7. **Have Colab/Jupyter ready** - For coding questions
8. **Prepare ID card** - He checks it first
9. **Know F1 formula** - 2 × (P × R) / (P + R)
10. **Know RandomSearch is faster** - He asks this often

---

# 📱 QUICK CODE TO MEMORIZE

## Iris Dataset (MEMORIZE THIS!)
```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data      # Shape: (150, 4), Type: numpy.ndarray
y = iris.target    # Shape: (150,), Type: numpy.ndarray
print(X.shape)     # (150, 4)
print(type(X))     # <class 'numpy.ndarray'>
print(iris.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```

**Good luck! You've got this! 🚀**
