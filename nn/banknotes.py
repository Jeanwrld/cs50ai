import csv
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row[4] == "0" else 0
        })

# Separate data into training and testing groups
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

# Create a neural network with layers
model = tf.keras.models.Sequential()

# Add a hidden layer with 8 units, with ReLU activation
#model.add(tf.keras.layers.Dense(8, input_shape=(4,), activation="relu"))#dense layers-nodes connected
#8 nodes, neurons
#cleaner version
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


# Add output layer with 1 unit, with sigmoid activation
#model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
#1 unit to compute either counterfeit or authentic

# Train neural network
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
X_training = np.array(X_training, dtype=np.float32)
y_training = np.array(y_training, dtype = np.float32)
model.fit(X_training, y_training, epochs=20)#train it on training data and label, go over 20 times


# Evaluate how well model performs

X_testing = np.array(X_testing, dtype=np.float32)
y_testing = np.array(y_testing, dtype=np.float32)

model.evaluate(X_testing, y_testing, verbose=2)

