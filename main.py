from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
from label_unwrapper import LabelUnwrapper  # Place your original class in label_unwrapper.py
# flask-cors is required to allow cross-origin requests
from flask_cors import CORS

# Initialize Flask app
# app = Flask(__name__)
app = Flask(__name__)
CORS(app)

# Predefined label points (percentages)
PREDEFINED_SHAPE = [
    {"x": 0.012232142857142842, "y": 0.2219140625},
    {"x": 0.48655701811449864, "y": 0.14404355243445227},
    {"x": 0.9632539682539681, "y": 0.2171875},
    {"x": 0.9466567460317459, "y": 0.7276953125},
    {"x": 0.48447501824501454, "y": 0.7952298867391453},
    {"x": 0.023134920634920626, "y": 0.7258984375}
]

@app.route('/unwrap_label', methods=['POST'])
def unwrap_label():
    try:
        # Get image file from request
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({"error": "Image is required"}), 400

        # Read image from uploaded file
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Normalize predefined points
        points = [[p['x'], p['y']] for p in PREDEFINED_SHAPE]

        # Unwrap the label
        unwrapper = LabelUnwrapper(src_image=image, percent_points=points)
        unwrapped_image = unwrapper.unwrap()

        # Perform OCR on the unwrapped image
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(unwrapped_image, config=custom_config)

        # Convert unwrapped image to a format suitable for sending in response
        _, buffer = cv2.imencode('.jpg', unwrapped_image)
        unwrapped_image_encoded = buffer.tobytes()

        return jsonify({
            "text": text,
            "unwrapped_image": unwrapped_image_encoded.hex()
        })

    except Exception as e:
        # Log the error
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
