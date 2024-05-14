from flask import Flask, request, jsonify,  render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Load the saved model
model = load_model('potato_leaf_model.h5')

# Initialize Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['file']
    
    # Read the file contents as bytes
    img_bytes = file.read()
    
    # Convert the bytes to an in-memory file-like object
    img_stream = io.BytesIO(img_bytes)
    
    # Preprocess the image
    img = image.load_img(img_stream, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize pixel values
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get class label
    class_names = ["EARLY_BLIGHT", "LATE_BLIGHT", "HEALTHY"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    # Return prediction as JSON response
    return jsonify({'prediction': predicted_class, 'confidence': float(confidence)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
