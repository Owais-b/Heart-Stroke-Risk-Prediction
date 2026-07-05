import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv("dataset/dataset.csv")      # <-- change filename if needed

# Remove ID column
df.drop("id", axis=1, inplace=True)

# ---------------------------------------------------
# Features & Target
# ---------------------------------------------------

X = df.drop("stroke", axis=1)
y = df["stroke"]

# ---------------------------------------------------
# Numerical & Categorical Columns
# ---------------------------------------------------

numerical_cols = [
    "age",
    "avg_glucose_level",
    "bmi"
]

categorical_cols = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

# ---------------------------------------------------
# Preprocessing
# ---------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ---------------------------------------------------
# Train-Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# Apply Preprocessing
# ---------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ---------------------------------------------------
# Balance Dataset using SMOTE
# ---------------------------------------------------

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\nDataset Before SMOTE")
print(y_train.value_counts())

print("\nDataset After SMOTE")
print(y_train_balanced.value_counts())

# ---------------------------------------------------
# Random Forest
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ---------------------------------------------------
# Train
# ---------------------------------------------------

model.fit(
    X_train_balanced,
    y_train_balanced
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)[:,1]

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

print("\n==============================")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("==============================\n")

print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score")
print(roc_auc_score(y_test, y_prob))

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------

joblib.dump(model, "models/stroke_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("\nModel Saved Successfully!")