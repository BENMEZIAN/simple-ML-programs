# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score


#  open the dataset and slicing it into independent and dependent variables
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
y = dataset.iloc[:, -1].values

# performing a train test split on the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying gaussian naive bayes algorithm
classifier = GaussianNB() # instantiate the model
classifier.fit(X_train, y_train) # fit the model

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2'], yticklabels=['Class 1', 'Class 2'])
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

"""
# print the preprocessed_df
preprocessed_df = pd.DataFrame(data=X_train, columns=dataset.columns[:-1])
preprocessed_df['contact-lenses'] = y_train
print(preprocessed_df)
"""