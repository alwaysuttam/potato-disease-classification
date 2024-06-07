from flask import Flask, request, jsonify,  render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Load the saved model
model = load_model('potato_leaf_model.h5')


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
   
    file = request.files['file']
    
    img_bytes = file.read()
    
    
    img_stream = io.BytesIO(img_bytes)
    
    
    img = image.load_img(img_stream, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  
    
    
    prediction = model.predict(img_array)
    
    class_names = ["EARLY_BLIGHT", "LATE_BLIGHT", "HEALTHY"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    
    return jsonify({'prediction': predicted_class, 'confidence': float(confidence)})


if __name__ == '__main__':
    app.run(debug=True)
