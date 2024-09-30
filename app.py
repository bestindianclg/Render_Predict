from flask import Flask, request, render_template
import numpy as np
import cv2
from sklearn.cluster import KMeans
import os

app = Flask(__name__, static_url_path='/static')

# Create a folder for uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_colors(image_path, n_colors=10):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Use KMeans to cluster the colors
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    # Get the colors and their percentages
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    label_counts = np.bincount(labels)
    total_count = len(labels)

    # Prepare colors and their percentages
    color_percentage = [(colors[i], label_counts[i] / total_count) for i in range(n_colors)]
    return color_percentage

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    colors = []
    image_url = None
    color_codes = []
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Predict colors
            colors = predict_colors(image_path)

            # Serve the uploaded image
            image_url = f'/static/uploads/{file.filename}'

            # Prepare color codes
            color_codes = ['#' + ''.join(f'{c:02x}' for c in color[0]) for color in colors]

    # Convert colors to RGB format for display
    colors_rgb = ["rgb({}, {}, {})".format(c[0][0], c[0][1], c[0][2]) for c in colors]

    return render_template('index.html', colors=colors_rgb, image_path=image_url, color_codes=color_codes)


if __name__ == '__main__':
    app.run()
