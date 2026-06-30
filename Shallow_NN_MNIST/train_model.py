import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================
# STEP 1: Load the MNIST Dataset
# ==========================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ==========================================
# STEP 2: Normalize the Images
# ==========================================
x_train = x_train / 255.0
x_test = x_test / 255.0

# ==========================================
# STEP 3: Flatten the Images
# ==========================================
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# ==========================================
# STEP 4: Build the Shallow Neural Network
# ==========================================
model = Sequential()

model.add(Dense(10, input_shape=(784,), activation="softmax"))

# ==========================================
# STEP 5: Compile the Model
# ==========================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# STEP 6: Train the Model
# ==========================================
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32
)

# ==========================================
# STEP 7: Evaluate the Model
# ==========================================
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\n==============================")
print("Test Loss :", test_loss)
print("Test Accuracy :", test_accuracy)
print("==============================")

# ==========================================
# STEP 8: Make Predictions
# ==========================================
predictions = model.predict(x_test)

predicted_digit = predictions[0].argmax()

print("\nPredicted Digit :", predicted_digit)
print("Actual Digit    :", y_test[0])

# ==========================================
# STEP 9: Save the Model
# ==========================================
model.save("mnist_shallow_model.keras")

print("\nModel saved successfully!")