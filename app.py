from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS  # Import CORS

# Limit GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

# Load the model
model = tf.keras.models.load_model('fruit_quality_model4.h5')  # Update with your model path

# Class labels
class_names = {
    0: 'Apple_Bad', 1: 'Apple_Good', 2: 'Apple_mixed',
    3: 'Banana_Bad', 4: 'Banana_Good', 5: 'Banana_mixed',
    6: 'Guava_Bad', 7: 'Guava_Good', 8: 'Guava_mixed',
    9: 'Lemon_mixed', 10: 'Lime_Bad', 11: 'Lime_Good',
    12: 'Orange_Bad', 13: 'Orange_Good', 14: 'Orange_mixed',
    15: 'Pomegranate_Bad', 16: 'Pomegranate_Good', 17: 'Pomegranate_mixed'
}

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

@app.route('/')
def index():
    return 'Welcome to the Fruit Prediction API!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Open the image
        img = Image.open(file.stream)
        img = img.resize((150, 150))  # Resize to model's expected input size (150x150)
        img = np.array(img) / 255.0  # Normalize image

        # Ensure the image has the correct dimensions
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names.get(predicted_index, "Unknown")

        # Return prediction as JSON
        return jsonify({
            'prediction': str(predicted_index),
            'label': predicted_label
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
