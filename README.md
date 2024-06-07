# Potato Leaf Disease Prediction Web Application

This web application uses a trained deep learning model to predict the condition of potato leaves from uploaded images. It classifies the images into three categories: Early Blight, Late Blight, and Healthy.

## Project Structure

- `app.py`: The main Flask application script.
- `potato_leaf_model.h5`: The pre-trained model file.
- `templates/`: Directory containing HTML templates for the web pages.
  - `index.html`: The main page where users can upload an image for prediction.

## Setup Instructions

### Prerequisites

- Python 
- Flask
- TensorFlow
- NumPy
- Pillow (PIL)

### Installation

1. Clone the Repository:
  
   git clone https://github.com/uttam-bn/potato-disease-classification.git
   
   cd potato-leaf-prediction

2. Install the Required Packages

### Running the Application

1. Run the Flask Application:
    python app.py

2. Open Your Web Browser:
    Go to http://127.0.0.1:5000/ to access the application.
