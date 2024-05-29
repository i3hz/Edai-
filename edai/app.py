#use set FLASK_APP=app.py first
#or on powershell use $env:FLASK_app = "hello.py"

from flask import Flask, request, render_template, send_file
from PIL import Image, ImageFilter
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
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            if filter_type == 'grayscale':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif filter_type == 'blur':
                img = cv2.blur(img, (5,5))
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
            cv2.imwrite('static/filtered_image.jpg', img)
            return render_template('show_image.html', filename='filtered_image.jpg')
    return render_template('upload_image.html')

if __name__ == '__main__':
    app.run(debug=True)
