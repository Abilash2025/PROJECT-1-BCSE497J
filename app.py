import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
from flask import Flask, request, render_template, redirect, url_for

# Initialize the Flask app
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
app.config['GRADCAM_FOLDER'] = 'static/gradcam_images'

# Load the pre-trained model
model = load_model('final_model.h5')
index = ['glioma', 'notumor', 'meningioma', 'pituitary', 'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', 'Breast Cancer', 'No Breast Cancer']

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRADCAM_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Check if a file is in the request
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file:
            # Load the image using OpenCV to resize it to 150x150
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (150, 150))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            cv2.imwrite(filepath, resized_img)  # Save resized image

            img_array = image.load_img(filepath, target_size=(150, 150))
            input_image = image.img_to_array(img_array)
            input_image = np.expand_dims(input_image, axis=0)
            predictions = model.predict(input_image)
            predicted_class = np.argmax(predictions, axis=1)
            prediction_label = index[predicted_class[0]]

            # Generate Grad-CAM
            last_conv_layer = model.get_layer('top_conv')  # Ensure this matches your model
            grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(input_image)
                loss = preds[:, predicted_class[0]]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            pooled_grads = pooled_grads.numpy()
            heatmap = np.mean(conv_outputs.numpy() * pooled_grads, axis=-1)

            # Normalize the heatmap
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            heatmap = cv2.resize(heatmap, (150, 150))
            heatmap = np.uint8(255 * heatmap)

            # Overlay the heatmap on the original image
            img_array = np.array(img_array)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_image = heatmap_colored * 0.4 + img_array * 255
            output_image_path = os.path.join(app.config['GRADCAM_FOLDER'], 'gradcam_' + file.filename)
            cv2.imwrite(output_image_path, superimposed_image.astype('uint8'))

            filepath = filepath.replace("\\", "/")
            output_image_path = output_image_path.replace("\\", "/")

            print("Original Image Path:", filepath)
            print("Grad-CAM Image Path:", output_image_path)

            return render_template('index.html', 
                                   prediction=prediction_label, 
                                   original_image=filepath, 
                                   gradcam_image=output_image_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
