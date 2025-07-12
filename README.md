# Gender Detection using MobileNetV2

## Project Overview
This project is an automatic gender detection system using facial recognition technology. The system utilizes MobileNetV2, a pre-trained Convolutional Neural Network (CNN) model, to classify faces as either Male or Female based on the facial features. The system also integrates OpenCV for face detection from images or video feeds.

## Key Features
- **Real-time Gender Prediction**: Automatically detects gender (Male/Female) using a webcam feed.
- **Face Detection**: Uses OpenCV's Haar Cascade Classifier to detect faces in real-time.
- **MobileNetV2**: A pre-trained lightweight CNN model is used for accurate gender classification.
- **Easy to Use**: Simple interface for capturing face images through the webcam and displaying the gender prediction.
- **High Accuracy**: Achieves high accuracy in gender classification based on facial features.

## Technology Stack
- **Deep Learning**: MobileNetV2 for gender classification.
- **Face Detection**: OpenCV for detecting faces from video feeds.
- **Webcam Integration**: OpenCV for capturing video from the webcam and processing frames.
- **Python Libraries**: TensorFlow, OpenCV, Numpy, and others for model training and inference.

## Installation

To run this project locally, follow these steps:

### Step 1: Clone the repository

git clone https://github.com/YourUsername/Gender-Detection.git
Step 2: Install dependencies
Ensure you have Python 3.x installed. Install the required dependencies using pip:


pip install -r requirements.txt
Dependencies include:

TensorFlow

OpenCV

Numpy

Step 3: Run the application
To start the gender detection system, run the following command:


python gender_recognition.py
This will open a webcam feed and start predicting the gender of the detected faces in real-time.

How It Works
The system uses OpenCV's Haar Cascade Classifier to detect faces in the webcam feed.

Once a face is detected, the image of the face is passed through the MobileNetV2 model to predict the gender.

The result (Male or Female) is displayed on the video feed in real-time.

If no face is detected, the system will not make a prediction.

Example Usage
After running the gender_recognition.py script, the webcam feed will appear. The system will automatically detect faces and predict gender based on the detected face. The predicted gender (Male/Female) will be displayed on the webcam feed.

Sample Output:

Gender: Male
Sample Webcam Feed:

Dataset
The system uses a pre-trained MobileNetV2 model, which was trained on the ImageNet dataset. For gender classification, the model has been fine-tuned using a gender-specific dataset to predict Male and Female based on facial features.

Evaluation
The MobileNetV2 model provides high accuracy for gender classification. The model can predict gender with an accuracy of over 90% when tested on real-world datasets.

Model Performance:
Accuracy: High (depends on the quality and resolution of the input images).

Inference Speed: Fast, with real-time predictions on webcam feeds.
