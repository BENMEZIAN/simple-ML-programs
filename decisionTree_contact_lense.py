import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,roc_curve, auc
import matplotlib.pyplot as plt

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

# Applying decision tree algorithm
classifier = DecisionTreeClassifier(criterion = "entropy",max_depth = 5, min_samples_leaf = 5) 
classifier.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(12, 6))
plot_tree(classifier, filled=True, feature_names=["age", "spectacle-prescrip", "astigmatism", "tear-prod-rate"], class_names=classifier.classes_)
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

# Get predicted probabilities for each class
y_probs = classifier.predict_proba(X_test)

# Compute ROC curve and ROC AUC score for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Binarize the labels
y_test_bin = label_binarize(y_test, classes=classifier.classes_)

# Compute ROC curve and ROC AUC score for each class
for i in range(len(classifier.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(len(classifier.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'Class {classifier.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()