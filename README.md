Attendance System Based on Face Recognition

Project Overview
This project involves creating an automated attendance system using facial recognition technology. The system aims to replace traditional manual attendance methods with a more efficient, accurate, and secure approach. Using Convolutional Neural Networks (CNN) and the Inception-ResNet-V2 model, the system automatically verifies and records attendance by recognizing faces, reducing administrative workload, and minimizing human error.

Key Features
Face Recognition-based Attendance: Uses facial recognition to mark attendance automatically.

High Accuracy: Achieves up to 97.71% accuracy with the Inception-ResNet-V2 model.

Data Integration: Integrated with a Laravel-based web application for seamless user interaction.

Easy to Use: Simple user interface where participants take a selfie, and the system auto-fills the attendance form.

Gender Detection: Enhanced with gender detection using the MobileNetV2 model for additional functionality.

Technology Stack
Deep Learning: Inception-ResNet-V2 for facial recognition.

Web Framework: Laravel for the front-end and Flask for the back-end API.

Data Processing: VSCode for model training, data augmentation, and integration.

API: RESTful API using Flask to handle image processing and face recognition predictions.

Face Detection: OpenCV for detecting faces from images.

Gender Classification: MobileNetV2 for detecting the gender of the individuals.

Installation
To run this project locally, follow these steps:

Step 1: Clone the repository
bash

git clone https://github.com/YourUsername/Attendance-System.git
Step 2: Install dependencies
For Laravel (front-end):

bash

cd attendance-system-laravel
composer install
For Flask API (back-end):

bash

cd flask-api
pip install -r requirements.txt
Step 3: Set up the database in Laravel
bash

php artisan migrate
Step 4: Start the Laravel server
bash

php artisan serve
Step 5: Run the Flask API server
bash

python app.py
Step 6: Access the system
Access the system through the Laravel web interface.

How It Works
Participants access the attendance page.

They click on "Take Photo," and the system captures their face.

The image is sent to the Flask API, which processes it using the trained CNN model.

If a match is found, the system auto-fills the attendance form with participant details (name, ID, position).

If no match is found, an alert is shown.

Dataset
The system uses a facial image dataset with front, left, and right view photos of participants. These images are collected via high-resolution cameras (minimum 1080p) to ensure the model's performance and accuracy.

Evaluation
Model Performance
The Inception-ResNet-V2 model achieves an accuracy of 97.71%, ensuring high reliability for real-time attendance tracking.

Confusion Matrix
The model demonstrates excellent results, with perfect classification (all values on the diagonal).

Conclusion
This face recognition-based attendance system provides a fast, accurate, and user-friendly solution to attendance management. It eliminates manual data entry errors, speeds up the process, and makes it more efficient for both administrators and participants.
