from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import os
import numpy as np
from scipy import ndimage
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the YOLO model
model = YOLO('model/bestc.pt')

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Known mass of sendok (spoon) in grams
SENDOK_MASS = 23  # Average of 20-26g

# Updated calorie density estimates (calories per 100 grams)
CALORIE_DENSITY = {
    'sendok': 0,
    'ayam manis': 200,  # Estimate for ayam suwir (shredded chicken)
    'bakso': 202,
    'mie': 250,
}

# Density estimates (grams per cubic centimeter)
FOOD_DENSITY = {
    'sendok': 7.9,
    'ayam manis': 0.7,
    'bakso': 0.5,
    'mie': 0.5,
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_geometric_features(result):
    masks = result[0].masks
    names = result[0].names
    data = []

    if masks is not None:
        mask_array = masks.data.cpu().numpy()
        class_indices = result[0].boxes.cls.cpu().numpy()
        orig_shape = masks.orig_shape
        scale_factor = (orig_shape[0] / mask_array.shape[1], orig_shape[1] / mask_array.shape[2])

        for i in range(mask_array.shape[0]):
            single_mask = mask_array[i]
            class_name = names[int(class_indices[i])]

            # Calculate area
            area = np.count_nonzero(single_mask) * scale_factor[0] * scale_factor[1]

            # Calculate perimeter
            perimeter = np.sum(ndimage.sobel(single_mask) != 0) * np.mean(scale_factor)

            # Calculate compactness (circularity)
            compactness = 4 * np.pi * area / (perimeter ** 2)

            # Calculate aspect ratio
            y, x = np.nonzero(single_mask)
            aspect_ratio = np.ptp(x) / np.ptp(y) if np.ptp(y) != 0 else 1

            data.append((class_name, area, perimeter, compactness, aspect_ratio))

    return data

def estimate_volumes_and_calories(data, sendok_length_cm):
    sendok_data = next((item for item in data if item[0].lower() == 'sendok'), None)
    
    if sendok_data is None:
        # If no sendok is detected, use a default calibration
        pixel_to_cm_ratio = 1  # Adjust this value based on your typical image resolution
        volume_to_mass_ratio = 1  # Adjust this value based on your typical food density
    else:
        sendok_area = sendok_data[1]
        pixel_to_cm_ratio = sendok_length_cm / np.sqrt(sendok_area)
        sendok_volume = estimate_volume(sendok_data, pixel_to_cm_ratio)
        volume_to_mass_ratio = SENDOK_MASS / sendok_volume

    volumes_and_calories = []
    for item_data in data:
        name = item_data[0]
        volume_cm3 = estimate_volume(item_data, pixel_to_cm_ratio)

        # Calibrate mass based on sendok
        mass_grams = volume_cm3 * volume_to_mass_ratio

        # Look up calorie density
        calorie_density = CALORIE_DENSITY.get(name.lower(), 100)

        # Calculate calories
        calories = (mass_grams / 100) * calorie_density

        volumes_and_calories.append((name, volume_cm3, mass_grams, calories))

    return volumes_and_calories

def estimate_volume(item_data, pixel_to_cm_ratio):
    name, area, perimeter, compactness, aspect_ratio = item_data

    # Estimate length and width
    length_cm = np.sqrt(area) * pixel_to_cm_ratio
    width_cm = area / perimeter * pixel_to_cm_ratio * 2

    # Estimate volume using multiple methods
    volume_cylinder = np.pi * (width_cm/2)**2 * length_cm / 3
    volume_ellipsoid = 4/3 * np.pi * (length_cm/2) * (width_cm/2)**2
    volume_box = length_cm * width_cm * width_cm / 3

    # Weight the volume estimates based on compactness and aspect ratio
    weight_cylinder = compactness
    weight_ellipsoid = 1 - abs(aspect_ratio - 1) / max(aspect_ratio, 1/aspect_ratio)
    weight_box = 1 - compactness

    total_weight = weight_cylinder + weight_ellipsoid + weight_box
    volume_cm3 = (volume_cylinder * weight_cylinder +
                  volume_ellipsoid * weight_ellipsoid +
                  volume_box * weight_box) / total_weight

    return volume_cm3

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image with YOLO
        results = model(file_path, conf=0.5)
        
        # Calculate geometric features
        data = calculate_geometric_features(results)
        
        # Known length of Sendok in cm
        sendok_length_cm = 19
        
        # Estimate volumes and calories
        estimates = estimate_volumes_and_calories(data, sendok_length_cm)
        
        # Plot the results
        annotated_img = results[0].plot(conf=True, labels=True)
        
        # Save the processed image
        processed_filename = f"processed_{filename}"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, annotated_img)
        
        # Generate URL for the processed image
        image_url = f"{request.url_root}processed/{processed_filename}"
        
        # Prepare the response
        response_data = {
            'image_url': image_url,
            'objects': [
                {
                    'name': name,
                    'volume_cm3': float(volume),
                    'mass_grams': float(mass),
                    'calories': float(calories)
                }
                for name, volume, mass, calories in estimates
            ]
        }
        
        return jsonify(response_data)
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    app.run(debug=True)