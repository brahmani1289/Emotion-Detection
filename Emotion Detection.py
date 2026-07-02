import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("emotion_dataset.csv")

# Features and labels
X = data[['Heart Rate', 'Age']]   # numeric features
y = data['Emotion']               # target label

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model (increase iterations for convergence)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model and scaler
joblib.dump(model, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")

# Prediction function
def predict_emotion(heart_rate, age):
    vec = pd.DataFrame([[heart_rate, age]], columns=['Heart Rate', 'Age'])
    vec_scaled = scaler.transform(vec)  # scale input
    return model.predict(vec_scaled)[0]

# Example prediction
print(predict_emotion(80, 25))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
