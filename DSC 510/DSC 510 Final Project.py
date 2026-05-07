## Adil Alimohd
## DSC 510
## Mental Health Survey Data Analysis

# Mental Health in Tech — EDA, Cleaning & Classification
# ========================================================
# Run: python mental_health_simple.py
# Change INPUT_FILE below to point to your CSV.

import warnings
import numpy as np
import pandas as pd
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



## checking/setting WD
os.chdir('/Users/u385722/Desktop')
os.getcwd()


# ── File paths (change these as needed) ─────────────────────────────────────
INPUT_FILE  = 'survey.csv'
OUTPUT_CSV  = 'survey_cleaned_model_ready.csv'
OUTPUT_PNG  = 'eda_report.png'

# ── Colours used in all charts ────────────────────────────────────────────────
BG   = '#F8F9FA'
DARK = '#2C3E50'
RED  = '#E85D4A'
BLUE = '#4A90D9'
GOLD = '#F0A500'
MUTE = '#AAB7B8'
LTBL = '#7EC8E3'

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================
print(f"[→] Loading data from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
df_raw = df.copy()   # keep an unmodified copy for the missing-values chart
print(f"    Rows: {len(df)}  Columns: {len(df.columns)}")

# Missing value summary
print("\n── Missing Values (raw) ─────────────────────────────────────")
for col in df.columns:
    n = df[col].isnull().sum()
    if n > 0:
        print(f"  {col:<30} {n:>4}  ({n/len(df)*100:.1f}%)")

# =============================================================================
# STEP 2 — CLEAN DATA
# =============================================================================
print("\n[→] Cleaning data ...")

# --- Gender: collapse 50+ raw strings into three groups ---
male_terms = {
    'male', 'm', 'man', 'cis male', 'cis man', 'male (cis)', 'maile', 'mal',
    'mail', 'malr', 'msle', 'make', 'male-ish', 'guy (-ish) ^_^',
    'male leaning androgynous', 'ostensibly male, unsure what that really means',
    'something kinda male?'
}
female_terms = {
    'female', 'f', 'woman', 'cis female', 'cis-female/femme', 'female (cis)',
    'female (trans)', 'trans-female', 'trans woman', 'femake', 'femail', 'women'
}

cleaned_genders = []
for raw in df['Gender']:
    val = str(raw).strip().lower()
    if val in male_terms:
        cleaned_genders.append('Male')
    elif val in female_terms:
        cleaned_genders.append('Female')
    else:
        cleaned_genders.append('Non-binary/Other')
df['Gender'] = cleaned_genders

# --- Age: replace anything outside 18–75 with the median ---
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.loc[(df['Age'] < 18) | (df['Age'] > 75), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df['Age'].median())

# --- work_interfere: NaN likely means "not affected" → fill as Never ---
df['work_interfere'] = df['work_interfere'].fillna('Never')

# --- self_employed: fill 18 missing values with No ---
df['self_employed'] = df['self_employed'].fillna('No')

print("    Gender normalised | Age outliers fixed | NaN columns imputed")

# =============================================================================
# STEP 3 — BUILD TARGET LABEL
# =============================================================================
# Target = 1 when an employee is likely impacted AND not seeking help.
# Impacted: sought treatment OR work interference is Sometimes/Often
# Not seeking: seek_help is No or Don't know
impacted    = (df['treatment'] == 'Yes') | (df['work_interfere'].isin(['Sometimes', 'Often']))
not_seeking = df['seek_help'].isin(['No', "Don't know"])
target      = (impacted & not_seeking).astype(int)

pos = target.sum()
neg = len(target) - pos
print(f"\n── Target Label ─────────────────────────────────────────────")
print(f"  Negative (0): {neg}   Positive (1): {pos}   Balance: {target.mean():.1%} positive")

# =============================================================================
# STEP 4 — EDA VISUALISATIONS
# =============================================================================
print("\n[→] Generating EDA visualisations ...")

WORK_ORDER   = ['Never', 'Rarely', 'Sometimes', 'Often']
COMPANY_SIZE = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']

fig = plt.figure(figsize=(20, 26), facecolor=BG)
fig.suptitle('Mental Health in Tech — Survey EDA', fontsize=22,
             fontweight='bold', color=DARK, y=0.98)
gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.5, wspace=0.4)

# ── Row 0: Age | Gender | Treatment ──────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.hist(df['Age'], bins=30, color=BLUE, edgecolor='white', linewidth=0.5)
ax.axvline(df['Age'].median(), color=RED, linestyle='--', linewidth=2,
           label=f'Median: {df["Age"].median():.0f}')
ax.set_title('Age Distribution (cleaned)', fontweight='bold', color=DARK)
ax.set_xlabel('Age'); ax.set_ylabel('Count')
ax.legend(); ax.set_facecolor(BG)

ax = fig.add_subplot(gs[0, 1])
gc = df['Gender'].value_counts()
ax.pie(gc.values, labels=gc.index, colors=[BLUE, RED, GOLD],
       autopct='%1.1f%%', startangle=90,
       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Gender Distribution', fontweight='bold', color=DARK)

ax = fig.add_subplot(gs[0, 2])
tc = df['treatment'].value_counts()
bars = ax.bar(tc.index, tc.values, color=[RED, BLUE], edgecolor='white', linewidth=1.5)
ax.set_title('Sought Treatment', fontweight='bold', color=DARK)
ax.set_ylabel('Count'); ax.set_facecolor(BG)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(bar.get_height()), ha='center', fontweight='bold')

# ── Row 1: Work interference | Seek-help | Treatment × interference ───────────
ax = fig.add_subplot(gs[1, 0])
wi = df['work_interfere'].value_counts().reindex(WORK_ORDER)
ax.bar(WORK_ORDER, wi.values, color=[BLUE, LTBL, GOLD, RED], edgecolor='white')
ax.set_title('Work Interference\n(NaN → "Never")', fontweight='bold', color=DARK)
ax.set_ylabel('Count'); ax.set_facecolor(BG)
ax.tick_params(axis='x', rotation=20)

ax = fig.add_subplot(gs[1, 1])
sh = df['seek_help'].value_counts()
ax.bar(sh.index, sh.values, color=[BLUE, MUTE, RED], edgecolor='white')
ax.set_title('Would Seek Help Resources', fontweight='bold', color=DARK)
ax.set_ylabel('Count'); ax.set_facecolor(BG)

ax = fig.add_subplot(gs[1, 2])
cross = pd.crosstab(df['work_interfere'], df['treatment'],
                    normalize='index').reindex(WORK_ORDER)
cross.plot(kind='bar', ax=ax, color=[BLUE, RED], edgecolor='white', width=0.7)
ax.set_title('Treatment Rate\nby Work Interference', fontweight='bold', color=DARK)
ax.set_ylabel('Proportion'); ax.set_facecolor(BG)
ax.tick_params(axis='x', rotation=25)
ax.legend(title='Treatment')

# ── Row 2: Target | Benefits × treatment | Company size ──────────────────────
ax = fig.add_subplot(gs[2, 0])
ct = target.value_counts()
ax.bar(['Not target\n(0)', 'Target\n(1)'],
       [ct.get(0, 0), ct.get(1, 0)],
       color=[BLUE, RED], edgecolor='white')
ax.set_title('Proposed Target Label\nDistribution', fontweight='bold', color=DARK)
ax.set_ylabel('Count'); ax.set_facecolor(BG)
for i, v in enumerate([ct.get(0, 0), ct.get(1, 0)]):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold')

ax = fig.add_subplot(gs[2, 1])
bt = pd.crosstab(df['benefits'], df['treatment'], normalize='index')
bt.plot(kind='bar', ax=ax, color=[BLUE, RED], edgecolor='white', width=0.7)
ax.set_title('Treatment Rate by\nEmployer Benefits', fontweight='bold', color=DARK)
ax.set_ylabel('Proportion'); ax.set_facecolor(BG)
ax.tick_params(axis='x', rotation=30)
ax.legend(title='Treatment')

ax = fig.add_subplot(gs[2, 2])
ns = df['no_employees'].value_counts().reindex(COMPANY_SIZE)
ax.bar(range(len(COMPANY_SIZE)), ns.values, color=LTBL, edgecolor='white')
ax.set_xticks(range(len(COMPANY_SIZE)))
ax.set_xticklabels(COMPANY_SIZE, rotation=35, ha='right', fontsize=8)
ax.set_title('Company Size Distribution', fontweight='bold', color=DARK)
ax.set_ylabel('Count'); ax.set_facecolor(BG)

# ── Row 3: Missing-values heatmap | Top countries ─────────────────────────────
ax = fig.add_subplot(gs[3, :2])
miss      = df_raw.isnull().astype(int)
cols_miss = miss.columns[miss.sum() > 0].tolist()
if cols_miss:
    sns.heatmap(miss[cols_miss].T, ax=ax, cmap=[BLUE, RED],
                cbar=False, yticklabels=cols_miss)
ax.set_title('Missing Values Map (red = missing)', fontweight='bold', color=DARK)
ax.set_xlabel('Row Index')

ax = fig.add_subplot(gs[3, 2])
top_countries = df_raw['Country'].value_counts().head(8)
ax.barh(top_countries.index[::-1], top_countries.values[::-1],
        color=BLUE, edgecolor='white')
ax.set_title('Top Countries', fontweight='bold', color=DARK)
ax.set_xlabel('Count'); ax.set_facecolor(BG)

# ── Row 4: Correlation matrix ─────────────────────────────────────────────────
ax  = fig.add_subplot(gs[4, :])
cdf = df.copy()
cdf['work_interfere']              = cdf['work_interfere'].map({'Never':0,'Rarely':1,'Sometimes':2,'Often':3})
cdf['leave']                       = cdf['leave'].map({'Very easy':0,'Somewhat easy':1,"Don't know":2,'Somewhat difficult':3,'Very difficult':4})
cdf['seek_help']                   = cdf['seek_help'].map({'Yes':0,"Don't know":1,'No':2})
cdf['treatment']                   = cdf['treatment'].map({'No':0,'Yes':1})
cdf['family_history']              = cdf['family_history'].map({'No':0,'Yes':1})
cdf['benefits']                    = cdf['benefits'].map({"Don't know":0,'No':1,'Yes':2})
cdf['mental_health_consequence']   = cdf['mental_health_consequence'].map({'No':0,'Maybe':1,'Yes':2})
cdf['obs_consequence']             = cdf['obs_consequence'].map({'No':0,'Yes':1})
cdf['mental_vs_physical']          = cdf['mental_vs_physical'].map({"Don't know":0,'No':1,'Yes':2})
cdf['target']                      = target

corr_cols = ['work_interfere','leave','seek_help','treatment','family_history',
             'benefits','mental_health_consequence','obs_consequence',
             'mental_vs_physical','target']
corr_matrix = cdf[corr_cols].corr()
mask        = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, ax=ax, mask=mask, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.5})
ax.set_title('Correlation Matrix — Key Features & Target', fontweight='bold', color=DARK)

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"[✓] EDA chart saved → {OUTPUT_PNG}")
plt.close()

# =============================================================================
# STEP 5 — ENCODE FEATURES FOR MODEL
# =============================================================================
print("\n[→] Encoding features ...")

df_model = df.copy()

# Drop columns not useful for prediction
df_model = df_model.drop(columns=['Timestamp', 'state', 'comments', 'Country',
                                   'treatment', 'seek_help'], errors='ignore')

# Ordinal encoding
df_model['work_interfere'] = df_model['work_interfere'].map({'Never':0,'Rarely':1,'Sometimes':2,'Often':3})
df_model['no_employees']   = df_model['no_employees'].map({'1-5':0,'6-25':1,'26-100':2,'100-500':3,'500-1000':4,'More than 1000':5})
df_model['leave']          = df_model['leave'].map({'Very easy':0,'Somewhat easy':1,"Don't know":2,'Somewhat difficult':3,'Very difficult':4})

# Binary yes/no columns → 1/0
binary_cols = ['self_employed', 'family_history', 'remote_work', 'tech_company',
               'benefits', 'wellness_program', 'anonymity', 'obs_consequence']
for col in binary_cols:
    df_model[col] = df_model[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

# Tri-state columns (No=0, Maybe/Don't know/Some of them=1, Yes=2)
tristate_map  = {'No': 0, "Don't know": 1, 'Maybe': 1, 'Some of them': 1, 'Yes': 2}
tristate_cols = ['care_options', 'mental_health_consequence', 'phys_health_consequence',
                 'coworkers', 'supervisor', 'mental_health_interview',
                 'phys_health_interview', 'mental_vs_physical']
for col in tristate_cols:
    df_model[col] = df_model[col].map(tristate_map).fillna(1).astype(int)

# Gender: one-hot encode
df_model = pd.get_dummies(df_model, columns=['Gender'], drop_first=False)

# Add target
df_model['target'] = target.values

df_model.to_csv(OUTPUT_CSV, index=False)
print(f"[✓] Model-ready CSV saved → {OUTPUT_CSV}")
print(f"    Shape: {df_model.shape}  |  Features: {df_model.shape[1] - 1}")

# =============================================================================
# STEP 6 — BASELINE MODEL EVALUATION
# =============================================================================
print("\n[→] Evaluating baseline classifiers ...")

X = df_model.drop(columns=['target'])
y = df_model['target']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest':    RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print("\n── 5-Fold Cross-Validation Results ──────────────────────────")
for name, model in models.items():
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    f1  = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"  {name:<25} AUC={auc.mean():.3f}±{auc.std():.3f}  F1={f1.mean():.3f}±{f1.std():.3f}")

# Feature importances from Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n── Random Forest Feature Importances (top 10) ───────────────")
for feat, score in importances.head(10).items():
    bar = '█' * int(score * 100)
    print(f"  {feat:<35} {score:.4f}  {bar}")

print("\n[✓] Pipeline complete.")
