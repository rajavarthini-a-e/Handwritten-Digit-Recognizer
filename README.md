Handwritten Digit Recognition using Shallow Neural Network
Overview
Handwritten Digit Recognition using Shallow Neural Network is a Deep Learning project that classifies handwritten digits (0–9) using the MNIST dataset.
The system uses TensorFlow and Keras for model development, OpenCV for image preprocessing, and Streamlit for building an interactive web application. Users can upload a handwritten digit image, and the trained model predicts the digit along with its confidence score.
Features
Digit Classification

Recognizes handwritten digits from 0 to 9.
Predicts the uploaded digit using a trained Shallow Neural Network.

Image Preprocessing

Converts the uploaded image to grayscale.
Resizes the image to 28 × 28 pixels.
Inverts colors to match the MNIST dataset format.
Normalizes pixel values for consistent model input.
Flattens the image into a 784-dimensional input vector.

Model Prediction

Predicts the handwritten digit from the processed image.
Displays the confidence score for the predicted digit.

Interactive Web Application

Upload handwritten digit images directly through the browser.
View the uploaded image instantly.
Get real-time prediction results through a Streamlit interface.

Technologies Used

Python
TensorFlow
Keras
OpenCV
NumPy
Streamlit

Project Workflow

Load the MNIST dataset.
Train a Shallow Neural Network.
Save the trained model.
Upload a handwritten digit image.
Preprocess the uploaded image.
Predict the digit using the trained model.
Display the predicted digit and confidence score.

Prediction Mapping
InputActionUploaded Digit ImagePreprocessed to 28×28 GrayscaleProcessed ImageFlattened into 784-dimensional VectorTrained ModelPredicts Digit (0–9)Softmax OutputGenerates Confidence Score
Installation
bashpip install tensorflow streamlit opencv-python numpy
Run the Project
Train the Model
bashpython train_model.py
Predict from a Local Image
bashpython predict.py
Launch the Streamlit Application
bashstreamlit run app.py
Applications

Handwritten Digit Recognition
OCR (Optical Character Recognition) Fundamentals
Deep Learning Education
Image Classification
Neural Network Learning Projects

Future Enhancements

Improve preprocessing for real-world handwritten images.
Build a Deep Neural Network (DNN) for performance comparison.
Replace the Shallow Neural Network with a Convolutional Neural Network (CNN).
Display prediction probabilities for all digits.
Deploy the application using Streamlit Community Cloud.

Author
Rajavarthini A
Aspiring AI & Machine Learning Engineer | Deep Learning Enthusiast
