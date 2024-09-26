from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import xml.etree.ElementTree as ET

from helper_functions import count_overlaps_in_dataset, convert_annotations_to_dict, assign_device_to_model
from spsg_config import config

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload_images', methods=['POST'])
def upload_images():
    # Check if images and bounding box file are in the request
    if 'images' not in request.files:
        return jsonify({"error": "No images uploaded"}), 400
    if 'bounding_boxes' not in request.files:
        return jsonify({"error": "No bounding box file uploaded"}), 400

    # Save uploaded images
    image_files = request.files.getlist('images')
    img_paths = []
    for file in image_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img_paths.append(file_path)

    # Save the uploaded bounding box XML file
    bounding_box_file = request.files['bounding_boxes']
    bounding_box_filename = secure_filename(bounding_box_file.filename)
    bounding_box_filepath = os.path.join(app.config['UPLOAD_FOLDER'], bounding_box_filename)
    bounding_box_file.save(bounding_box_filepath)

    # Parse the XML bounding box file and extract the data
    bounding_box_data = convert_annotations_to_dict(bounding_box_filepath)

    # Call your counting function and pass bounding box data
    total_bboxes, unique_objects = count_overlaps_in_dataset(
        startIdx=0, 
        device='cpu', 
        running_overlap_count=0,
        matching_model = assign_device_to_model(config, gpu=False), 
        method='SPSG', 
        coord_dict=bounding_box_data  # Pass extracted bounding box data
    )

    # Return the result to the frontend
    return jsonify({"total_bounding_boxes": total_bboxes, "unique_objects": unique_objects})

if __name__ == '__main__':
    app.run(debug=True)
