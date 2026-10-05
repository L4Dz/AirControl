import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'data', 'raw_landmarks.csv')

df = pd.read_csv(csv_path)

target_classes = ['fist', 'palm', 'peace', 'point']
df = df[df['label'].isin(target_classes)].copy()

label_map = {
    'fist': 'rock',
    'palm': 'paper',
    'peace': 'scissors',
    'point': 'point'
}
df['label'] = df['label'].map(label_map)

X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

models_dir = os.path.join(script_dir, '..', 'models')
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, 'rps_point_model.pkl')

joblib.dump(model, model_save_path)
print(f"Model saved successfully to: {os.path.abspath(model_save_path)}")
