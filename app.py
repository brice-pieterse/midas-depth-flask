from flask import Flask, request, jsonify
import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import base64
from flask_cors import CORS
from midas.copyModel import load_model 
import logging


app = Flask(__name__)

app.logger.setLevel(logging.INFO)

CORS(app)  # Enable CORS for all routes


# Function to process the image using MiDaS model
def process_image(img):
    # Load MiDaS model and move it to the appropriate device
    device = torch.device("cpu")

    loaded_model, loaded_transform, loaded_net_w, loaded_net_h = load_model(device)

    model = loaded_model
    transform = loaded_transform
    target_size = img.shape[1::-1]
    image = transform({"image": img/255})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        height, width = sample.shape[2:]
        prediction = model(sample)

        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return prediction


@app.route('/', methods=['GET'])
def hello():
    return jsonify({ 'success': "Hi!" })


@app.route('/', methods=['POST'])
def test():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'})
    return jsonify({ 'success': "Image found!" })


@app.route('/process_image', methods=['POST'])
def predict_depth_map():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'})

    image_file = request.files['image']
    image_np = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    original_image_rgb = np.flip(image_np, 2)

    depth_map = process_image(original_image_rgb)

    normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    # Convert depth map to an image
    colormap = plt.get_cmap('gray')
    reconstructed_image = (colormap(normalized_depth_map) * 255).astype(np.uint8)

    # Convert image to base64 string
    retval, buffer = cv2.imencode('.jpg', cv2.cvtColor(reconstructed_image, cv2.COLOR_RGBA2BGRA))
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image_base64': img_str})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Modify host and port as needed