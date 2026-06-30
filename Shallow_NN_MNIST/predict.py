import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# STEP 1: Load the Trained Model
# ==========================================
print("Loading trained model...")
model = load_model("mnist_shallow_model.keras")

# ==========================================
# STEP 2: Read the Uploaded Image
# ==========================================
print("Reading image...")

image = cv2.imread("uploads/digit.jpeg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found!")
    exit()

# ==========================================
# STEP 3: Resize Image to 28x28
# ==========================================
print("Resizing image to 28x28...")

image = cv2.resize(image, (28, 28))

# ==========================================
# STEP 4: Invert the Image
# MNIST digits are white on black background
# ==========================================
print("Inverting image...")

image = 255 - image

# ==========================================
# STEP 5: Normalize Pixel Values
# ==========================================
print("Normalizing image...")

image = image.astype("float32") / 255.0

# ==========================================
# STEP 6: Flatten the Image
# ==========================================
print("Flattening image...")

image = image.reshape(1, 784)

# ==========================================
# STEP 7: Predict
# ==========================================
print("Predicting...")

prediction = model.predict(image)

predicted_digit = np.argmax(prediction)

confidence = np.max(prediction) * 100

# ==========================================
# STEP 8: Display Result
# ==========================================
print("\n==============================")
print("Prediction Completed")
print("==============================")

print(f"Predicted Digit : {predicted_digit}")
print(f"Confidence       : {confidence:.2f}%")

# ==========================================
# STEP 9: Display the Processed Image
# ==========================================
display_image = image.reshape(28, 28)

cv2.imshow("Processed Image", display_image)

cv2.waitKey(0)
cv2.destroyAllWindows()