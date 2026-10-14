import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("🔄 Tuning Machine Learning Engine for Better Real-Account Accuracy...")

# 1. Load data safely
data_path = os.path.join('data', 'instagram_fake_profile_dataset.csv')
df = pd.read_csv(data_path)

# 2. Extract features cleanly
X = pd.DataFrame()
X['has_profile_pic'] = df['profile pic']
X['username_digit_ratio'] = df['nums/length username']
X['fullname_words'] = df['fullname words']
X['fullname_digit_ratio'] = df['nums/length fullname']
X['name_equals_username'] = df['name==username']
X['bio_length'] = df['description length']
X['is_private'] = df['private']
X['post_count'] = df['#posts']
X['followers_count'] = df['#followers']
X['following_count'] = df['#follows']

# 3. Dynamic Feature Ratios (Handles extreme zero cases gracefully)
X['follower_following_ratio'] = X['followers_count'] / (X['following_count'] + 1)
X['engagement_score'] = X['post_count'] / (X['followers_count'] + 1)

y = df['fake']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Tuned Classifier (Added class_weight='balanced' to prevent bias against real accounts)
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=6, 
    min_samples_split=5,
    class_weight='balanced', 
    random_state=42
)
model.fit(X_train, y_train)

# 6. Evaluation Reports
predictions = model.predict(X_test)
print(f"✅ Tuned Engine Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\n--- Detailed Performance Metrics ---")
print(classification_report(y_test, predictions, target_names=['Genuine', 'Fake']))

# 7. Print Feature Weights (Excellent data point for your project viva!)
print("\n--- Feature Importance Standings ---")
for name, importance in sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")

# 8. Save updated model
joblib.dump(model, 'fake_profile_model.pkl')
print("\n📦 Optimized model successfully saved as 'fake_profile_model.pkl'!")
