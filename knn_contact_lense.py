import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Read the dataset
dataset = pd.read_csv('contact_lenses.csv')
X = dataset.iloc[:, [0, 1, 2, 3]].values
y = dataset.iloc[:, -1].values

# performing a train test split on the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Preprocessing
# Label encoding for categorical columns
columns_to_label_encode = [0, 1, 2, 3]
label_encoder = LabelEncoder()

for col in columns_to_label_encode:
    X_train[:, col] = label_encoder.fit_transform(X_train[:, col])
    X_test[:, col] = label_encoder.transform(X_test[:, col])

# Normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying KNN algorithm with Euclidean distance
classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predict the output for the test set
y_pred = classifier.predict(X_test)

print(y_pred)

# evaluate our model (confusion matrix and accuracy score and f1_score)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Confusion matrix is:")
print(cm)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("Accuracy score is:",ac)

# Create a bar chart to display the accuracy score
categories = ['Accuracy']
values = [ac]

plt.bar(categories, values, color='blue')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.show()

print("F1 Score:", f1)