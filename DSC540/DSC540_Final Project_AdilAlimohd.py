### DSC 540 Final Project
### Adil Alimohd
### Stroke Prediction EDA & XGBoost Model Run


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb


os.chdir('/Users/u385722/Desktop') ##setting wd

df = pd.read_csv('stroke_dataset.csv')

##Begin Explortatory Data Analysis (EDA)

#Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Count unique values per column
unique_values = df.nunique()
print("\nUnique Values:\n", unique_values)

# Target variable distribution (stroke)
stroke_distribution = df['stroke'].value_counts(normalize=True) * 100
print("\nStroke Distribution (%):\n", stroke_distribution)

# Categorical column unique values
categorical_columns = df.select_dtypes(include='object').columns
categorical_uniques = {col: df[col].unique() for col in categorical_columns}
print("\nUnique values in categorical columns:")

for col, values in categorical_uniques.items():
    print(f"{col}: {list(values)}")

## Dropping ID column as it has no use for this analysis
    
df.drop(columns=['id'], inplace=True)


## Begin Data Prep

# Fill in missing values with Median
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# One-Hot Encode categorical variables
cat_cols = df.select_dtypes(include='object').columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # Save encoder in case needed later

# Separate features and target variables
X = df.drop(columns=['stroke'])
y = df['stroke']

# Train-test split of 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost classifier
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

## Setting RMSE & r2 variables
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

## Feature Importance Plot
fig, ax = plt.subplots(figsize=(10, 6))  # Wider plot
xgb.plot_importance(
    model,
    importance_type='gain',
    max_num_features=10,        # Top 10 most important
    height=0.5,                 # Thinner bars
    ax=ax
)

# Hide text labels on bars
for txt in ax.texts:
    txt.set_visible(False)

plt.title("Feature Importance (Gain) - Model 1", fontsize=14)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()             # Fixes label cutoff
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.show()

## Dataset is imbalanced, so let's try to balance

model = xgb.XGBClassifier(scale_pos_weight=972/50) #This tells the model to weigh by data weights

print("Classification Report (w/ scale pos weight adjusted manually):\n", classification_report(y_test, y_pred))

## Didn't do much to help so let's try Cross Validation

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score, pos_label=1)
xgb_clf = xgb.XGBClassifier(eval_metric='logloss')

param_grid = {
    'scale_pos_weight': [5, 10, 15, 20],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
}
grid = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring=scorer,
    cv=skf,
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train, y_train)

model_2 = grid.best_estimator_
  
y_pred = model_2.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 6))  # Wider plot
xgb.plot_importance(
    model_2,
    importance_type='gain',
    max_num_features=10,        # Top 10 most important
    height=0.5,             
    ax=ax
)

# Hide text labels on bars
for txt in ax.texts:
    txt.set_visible(False)

plt.title("Feature Importance (Gain) With Cross Validation", fontsize=14)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.show()

print("Tuned Parameters:", grid.best_params_)
  
print("Classification Report (w/ cross validation):\n", classification_report(y_test, y_pred))
