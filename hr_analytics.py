import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)


# Load dataset
df = pd.read_csv("C:/Users/kusam/OneDrive/Desktop/dataproj/data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display attrition pie chart
attrition_counts = df['Attrition'].value_counts()
labels = ['No', 'Yes']
plt.figure(figsize=(5, 5))
plt.pie(attrition_counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=140)
plt.title('Attrition Distribution')
plt.show()

# X = all features, y = target
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_rf_pred))
print("\nClassification Report:\n", classification_report(y_test, y_rf_pred))

# Feature importance
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:10], y=features[indices][:10])
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.show()

# Add predictions to test data
X_test_copy = X_test.copy()
X_test_copy["Actual_Attrition"] = y_test.values
X_test_copy["Predicted_Attrition"] = y_rf_pred

# Save to file
X_test_copy.to_csv("attrition_predictions.csv", index=False)
