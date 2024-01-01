# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your dataset with pandas
# Replace 'your_dataset.csv' with the actual file path or URL to your dataset
data = pd.read_csv('diabetes.csv')

# Separate features and labels
X = data.drop('class', axis=1)  # Adjust 'class' to the name of your target column
y = data['class']

# Handle categorical variables (one-hot encoding)
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')  # Change to the number of classes for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Change for multi-class classification
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
