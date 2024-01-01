import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

#  open the dataset and slicing it into independent and dependent variables
dataset = pd.read_csv('diabetes.csv')

num_rows_to_remove = 200
label_to_remove = 'tested_negative'
negative_rows = dataset[dataset['class'] == label_to_remove]
rows_to_remove = negative_rows.head(num_rows_to_remove).index
dataset = dataset.drop(rows_to_remove)

X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
y = dataset.iloc[:, -1].values

# performing a train test split on the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying decision tree algorithm
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 4, min_samples_leaf = 5) 
classifier.fit(X_train, y_train)

# Hyperparameter tuning with GridSearchCV
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 4, 5, 6], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=100), param_grid, cv=5)
grid_search.fit(X_train, y_train)
classifier = grid_search.best_estimator_


# Plot the decision tree
plt.figure(figsize=(12, 6))
plot_tree(classifier, filled=True, feature_names=["preg", "plas", "pres", "skin","insu","mass","pedi","age"], class_names=classifier.classes_)
plt.show()

# Predict the output for the test set
y_pred = classifier.predict(X_test)

print(y_pred)

# evaluate our model (confusion matrix and accuracy score and f1_score)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Confusion matrix is:")
print(cm)
print("Accuracy score is:",ac)
print("F1 Score:", f1)

# Get predicted probabilities for the positive class
y_probs = classifier.predict_proba(X_test)[:, 1]

# Binarize the labels
y_test_bin = label_binarize(y_test, classes=classifier.classes_)

# Compute ROC curve and ROC AUC score for the positive class
fpr, tpr, _ = roc_curve(y_test_bin, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()