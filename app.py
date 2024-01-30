from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_dominant_color(img):
    height, width, _ = np.shape(img)
    data = np.reshape(img, (height * width, 3))
    data = np.float32(data)

    number_clusters = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, _, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)
    dominant_color = centers[0]
    return dominant_color

def change_hue(image, target_hue, dominant_color):
    image_hsv = rgb2hsv(image)
    hue_channel = image_hsv[:, :, 0]
    hue_channel[image[:, :, 0] == dominant_color[0]] = 0
    hue_shift = target_hue
    hue_channel_shifted = (hue_channel + hue_shift) % 180
    image_hsv[:, :, 0] = hue_channel_shifted
    image_hue_changed = hsv2rgb(image_hsv)
    return image_hue_changed

def save_image(image, output_folder, index):
    image_uint8 = (image * 255).astype(np.uint8)
    output_path = os.path.join(output_folder, f"changed_hue_{index}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))
    print(f"Image saved to: {output_path}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        single_image = request.files['single_image']
        group_image = request.files['group_image']
        if single_image and allowed_file(single_image.filename) and group_image and allowed_file(group_image.filename):
            single_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(single_image.filename))
            group_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(group_image.filename))

            single_image.save(single_image_path)
            group_image.save(group_image_path)

            img_single = cv2.imread(single_image_path)
            img_single_rgb = cv2.cvtColor(img_single, cv2.COLOR_BGR2RGB)

            dominant_color = find_dominant_color(img_single_rgb)

            img_group = cv2.imread(group_image_path)
            img_group_rgb = cv2.cvtColor(img_group, cv2.COLOR_BGR2RGB)

            height, width, _ = np.shape(img_group_rgb)
            data_group = np.reshape(img_group_rgb, (height * width, 3))
            data_group = np.float32(data_group)

            number_clusters_group = 15
            criteria_group = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags_group = cv2.KMEANS_RANDOM_CENTERS

            _, _, centers_group = cv2.kmeans(data_group, number_clusters_group, None, criteria_group, 10, flags_group)

            output_folder = os.path.join(os.path.dirname(__file__), 'static/output_changed_images')
            os.makedirs(output_folder, exist_ok=True)

            generated_images = []

            for i, target_color in enumerate(centers_group):
                img_changed = change_hue(img_single_rgb.copy(), target_color[0], dominant_color)
                save_image(img_changed, output_folder, i)
                generated_images.append(f"output_changed_images/changed_hue_{i}.jpg")

            return render_template('index.html', images=generated_images)

        return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)