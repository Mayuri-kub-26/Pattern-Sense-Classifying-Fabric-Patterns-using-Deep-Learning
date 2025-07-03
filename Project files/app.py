from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the correctly trained CNN model
model = load_model("PatternSense_ResNet50.h5e")  # Make sure this is a properly trained model

# Update class names in the same order they were trained
# You can get the correct order using: train_generator.class_indices
class_names = ['floral', 'polka_dot', 'striped']  # Example: update based on your training folders

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Save uploaded image
        img_path = os.path.join('static', file.filename)
        file.save(img_path)

        # Preprocess image
        img = image.load_img(img_path, target_size=(224, 224))  # must match training size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize only if your training did

        # Predict
        pred = model.predict(img_array)
        predicted_index = np.argmax(pred)
        predicted_class = class_names[predicted_index]

        # DEBUG: print raw prediction
        print("Predicted probabilities:", pred)
        print("Predicted class:", predicted_class)

        return render_template('result.html', prediction=predicted_class, image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
