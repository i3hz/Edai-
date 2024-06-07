# use set FLASK_APP=app.py first
# or on powershell use $env:FLASK_APP = "app.py"

from flask import Flask, request, render_template, send_file
from PIL import Image, ImageFilter, ImageSequence
import os
import cv2
import numpy as np
app = Flask(__name__)

import filter

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filter_type = request.form.get('filter')
        if file:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext in ['.jpg', '.jpeg', '.png']:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                img = apply_filter(img, filter_type)
                cv2.imwrite('static/filtered_image.jpg', img)
                return render_template('show_image.html', filename='filtered_image.jpg')
            elif file_ext == '.gif':
                img = Image.open(file)
                img = apply_filter_gif(img, filter_type)
                img[0].save('static/filtered_image.gif', format='GIF', save_all=True, append_images=img[1:], loop=0)
                return render_template('show_image.html', filename='filtered_image.gif')
    return render_template('upload_image.html')

def apply_filter(img, filter_type):
    if filter_type == 'grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        img = cv2.blur(img, (5, 5))
    elif filter_type == 'homomorphic':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = filter.apply_homomorphic_filter(img)
    elif filter_type == 'water':
        img = filter.apply_watercolor(img)
    elif filter_type == 'cartoonify':
        img = filter.apply_cartoonify(img)
    elif filter_type == 'sketch':
        img = filter.apply_sketch(img)
    elif filter_type == 'edgeDetection':
        img = filter.apply_edgeDetection(img)
    return img

def apply_filter_gif(img, filter_type):
    frames = []
    for frame in ImageSequence.Iterator(img):
        frame = frame.convert("RGB")
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        frame_cv = apply_filter(frame_cv, filter_type)
        frames.append(Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)))
    return frames

if __name__ == '__main__':
    app.run(debug=True)
