from pycocotools.coco import COCO
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import time


# Step One - Load COCO Annotations
def load_coco_annotations(json_path):
    coco = COCO(json_path)

    # Get all annotation ids
    all_ann_ids = coco.getAnnIds()

    # Load annotations
    annotations = []
    for ann_id in all_ann_ids:
        annotation = coco.loadAnns(ann_id)[0]
        annotations.append({'label': annotation['category_id'], 'bbox': annotation['bbox']})

    return annotations


# Step Two - Detect Circle in Image
def detect_circles(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply GaussianBlur to reduce noise
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply HoughCircles with adjusted parameters
    circles = cv2.HoughCircles(image_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=50, param2=30, minRadius=20,
                               maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Return circles with x, y, and radius
        return circles[0, :]
    else:
        return None


# Step Three - Predict Circle quality using Trained Model
def predict_circle_quality(model, detected_circles):
    if detected_circles is not None:
        # Use x, y, and radius coordinates for prediction (detected_circles[:, :3])
        predictions = model.predict(detected_circles[:, :3])
        probabilities = model.predict_proba(detected_circles[:, :3])

        return predictions, probabilities
    else:
        return None, None


# Step Four - Highlight predicted quality in the Image
def highlight_predicted_quality(image_path, detected_circles, predictions, probabilities, cat_id_to_label,
                                output_folder):
    image = cv2.imread(image_path)

    if detected_circles is not None:
        for circle, prediction, probability in zip(detected_circles, predictions, probabilities):
            cx, cy, radius = circle
            label = cat_id_to_label[prediction]

            if label == 'Good':
                color = (0, 255, 0)
            elif label == 'Bad':
                color = (0, 0, 255)
            elif label == 'Moderate':
                color = (255, 0, 0)

            # Calculate bounding box coordinates
            x_min, y_min = int(cx - radius), int(cy - radius)
            x_max, y_max = int(cx + radius), int(cy + radius)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save the highlighted image to the output folder
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, image)


# Step Five - Train and Evaluate the Model
def train_evaluate_model(annotations):
    # Split data into features and labels
    X = np.array([[ann['bbox'][0] + ann['bbox'][2] / 2, ann['bbox'][1] + ann['bbox'][3] / 2, ann['bbox'][2] / 2]
                  for ann in annotations])
    y = np.array([ann['label'] for ann in annotations])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = KNeighborsClassifier(n_neighbors=13)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, accuracy, report


# Step Six - Define a Mapping from Category ID to Label
def get_cat_id_to_label():
    return {
        1: 'Good',
        2: 'Bad',
        3: 'Moderate',
        # Add more mappings as needed
    }


# Step Seven - Check Unannotated Dataset
def check_unannotated_dataset(image_folder, model, cat_id_to_label, output_folder):
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        detected_circles = detect_circles(image_path)
        predictions, probabilities = predict_circle_quality(model, detected_circles)
        highlight_predicted_quality(image_path, detected_circles, predictions, probabilities, cat_id_to_label,
                                    output_folder)


# Load COCO annotations
coco_annotations = load_coco_annotations("path_to/annote.json")

# Train and evaluate the model
start_time = time.time()
trained_model, accuracy, report = train_evaluate_model(coco_annotations)
end_time = time.time()
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)
print(f"Training and evaluation time: {end_time - start_time:.2f} seconds")

# Get category ID to label mapping
cat_id_to_label = get_cat_id_to_label()

# Path to the folder containing unannotated images/test image
unannotated_image_folder = "path"

# Specify the output folder for storing result images
output_folder = "path"

# Check the unannotated dataset
check_unannotated_dataset(unannotated_image_folder, trained_model, cat_id_to_label, output_folder)

