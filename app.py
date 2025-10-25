from flask import Flask, request, render_template, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Load your trained model from root dir
MODEL_PATH = "custom_cnn_best.keras"
model = load_model(MODEL_PATH)

# Preprocess uploaded image
def preprocess(img_file, target_size=(256, 256)):
    """Preprocess uploaded image for model prediction"""
    try:
        # Read image from file stream
        img_stream = io.BytesIO(img_file.read())
        img = Image.open(img_stream)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    
    if request.method == "POST":
        file = request.files.get("file")
        
        if not file or file.filename == '':
            error = "Please select an image file"
        else:
            try:
                # Check file extension
                allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
                if not ('.' in file.filename and 
                       file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                    error = "Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP)"
                else:
                    # Reset file pointer
                    file.seek(0)
                    
                    # Preprocess image
                    x = preprocess(file)
                    
                    # Make prediction
                    preds = model.predict(x, verbose=0)
                    label = int(preds[0][0] > 0.5)  # Binary classification
                    confidence = float(preds[0][0])
                    
                    result = {
                        "label": label, 
                        "confidence": confidence,
                        "filename": file.filename
                    }
                    
            except Exception as e:
                error = f"Error processing image: {str(e)}"
    
    return render_template("upload.html", result=result, error=error)

if __name__ == "__main__":
    print("Starting Flask application...")
    print("Model loaded successfully!")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
