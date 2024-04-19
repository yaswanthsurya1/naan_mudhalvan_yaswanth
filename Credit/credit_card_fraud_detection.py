import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# Read the dataset
df = pd.read_csv('creditcard.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Split the dataset into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Initialize and train the SVC model
model_svc = SVC()
model_svc.fit(X_train, y_train)

# Evaluate the model
train_score = model_svc.score(X_train, y_train)
test_score = model_svc.score(X_test, y_test)

print("Training Accuracy:", train_score)
print("Testing Accuracy:", test_score)

# Make predictions
y_predict = model_svc.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_predict, labels=[1, 0])
confusion = pd.DataFrame(cm, index=['is Fraud', 'is Normal'], columns=['predicted fraud', 'predicted normal'])

# Plot confusion matrix
sns.heatmap(confusion, annot=True)

# Print classification report
print(classification_report(y_test, y_predict))
