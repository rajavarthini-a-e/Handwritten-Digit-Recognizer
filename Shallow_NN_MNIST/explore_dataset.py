import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# =====================================================
# STEP 1: Load the MNIST Dataset
# =====================================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# =====================================================
# STEP 2: Print Original Dataset Information
# =====================================================
print("\n========== ORIGINAL DATA ==========\n")

print("Type of x_train :", type(x_train))
print("Type of y_train :", type(y_train))

print("\nTraining Images Shape :", x_train.shape)
print("Training Labels Shape :", y_train.shape)

print("\nTesting Images Shape :", x_test.shape)
print("Testing Labels Shape :", y_test.shape)

# =====================================================
# STEP 3: Display the First Image (BEFORE FLATTENING)
# =====================================================
print("\nFirst Image Label :", y_train[0])

plt.imshow(x_train[0], cmap="gray")
plt.title(f"Digit : {y_train[0]}")
plt.axis("off")
plt.show()

# =====================================================
# STEP 4: Normalize the Pixel Values
# =====================================================
x_train = x_train / 255.0
x_test = x_test / 255.0

print("\n========== AFTER NORMALIZATION ==========\n")
print("Maximum Pixel Value :", x_train.max())
print("Minimum Pixel Value :", x_train.min())

# =====================================================
# STEP 5: Flatten the Images
# =====================================================
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print("\n========== AFTER FLATTENING ==========\n")

print("Training Shape :", x_train.shape)
print("Testing Shape  :", x_test.shape)

# =====================================================
# STEP 6: Print First Flattened Image
# =====================================================
print("\nFirst Flattened Image (784 values):\n")
print(x_train[0])

print("\nFirst Label :", y_train[0])