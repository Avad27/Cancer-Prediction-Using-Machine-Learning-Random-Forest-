# scripts/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle
import os

# Ensure the models directory exists
os.makedirs('app/models', exist_ok=True)

# Load the dataset from a local file
cancer = pd.read_csv('data/Cancer.csv')  # Update the path to your local file

# Define target (y) and features (X)
y = cancer['diagnosis']
X = cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

# Select and train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Save the trained model to a file
with open('app/models/trained_cancer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Model evaluation
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))