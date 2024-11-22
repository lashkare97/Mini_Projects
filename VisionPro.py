import os
import cv2
import numpy as np
import csv
from PIL import Image
import time
start_time = time.time()

# Input and output folder paths
input_folder = 'path'
output_folder = 'path'

# include sources to Original Dataset (input), Cropped (if required) and Operators. Also include for Composite and Test folder.

# Get a list of all .bmp files in the input folder
bmp_files = [file for file in os.listdir(input_folder) if file.endswith('.bmp')]

# Counter for renaming files
counter = 1

# Process each .bmp file
for bmp_file in bmp_files:
    # Read the BMP image
    image_bmp = cv2.imread(os.path.join(input_folder, bmp_file))
    # Convert to JPG format by saving with JPEG compression
    output_file = str(counter) + '.jpg'
    cv2.imwrite(os.path.join(output_folder, output_file), image_bmp, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    counter += 1

# Cropping, Resizing and applying Edge Filters (Sobel-Canny-Laplacian)

# Specify the source and destination directories

img_files = [f for f in os.listdir(path1) if f.endswith('.JPG')]  # A list of all the image files in the source directory
img_files.sort()  # Sort the list of image files in alphabetical order

# Iterate through each image file in the list
for img_file in img_files:
    img = cv2.imread(os.path.join(path1, img_file))  # Read the image using OpenCV

    # starting with image resizing
    img1 = cv2.imread(os.path.join(path1, img_file))  # Read the image using OpenCV
    resize = cv2.resize(img1, (256, 256))
    dst_path3 = os.path.join(path3, img_file)  # Save the processed image to the destination directory with the same name
    cv2.imwrite(dst_path3, resize)

    # starting with Sobel operator
    img2 = cv2.imread(os.path.join(path3, img_file))
    c = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    x_sobel = cv2.Sobel(c, cv2.CV_64F, 0, 1, ksize=5)
    dst_path4 = os.path.join(Sobel, img_file)  # Save the processed image to the destination directory with the same name
    cv2.imwrite(dst_path4, x_sobel)

    # starting with Canny operator
    canny = cv2.Canny(c, 50, 120)
    np.clip(c, 0, 1)
    dst_path5 = os.path.join(Canny, img_file)
    cv2.imwrite(dst_path5, canny)

    # starting with Laplacian operator
    laplacian = cv2.Laplacian(c, cv2.CV_16S)
    dst_path6 = os.path.join(Laplacian, img_file)
    cv2.imwrite(dst_path6, laplacian)

# Composite of Images

# Define the paths of the three directories where the images are located

list_name = list(set(os.listdir(Canny)) & set(os.listdir(Laplacian)) & set(os.listdir(Sobel)))
sequence = [Canny, Laplacian, Sobel]

# Process each common image
for image_name in list_name:
# Initialize an empty list to store images for merging
    merged_images = []

# Merge subsequent images into the composite image
    for folder in sequence[1:]:
        image_path = os.path.join(folder, str(image_name))
        image = cv2.imread(image_path)
        merged_images.append(image)

# Create a blank canvas to merge the images
    composite_image = np.zeros_like(merged_images[0])

# Merge the images by taking the maximum pixel values
    for image in merged_images:
        composite_image = np.maximum(composite_image, image)

# Save the composite image in the output folder
        output_path = os.path.join(Composite, str(image_name))
        cv2.imwrite(output_path, composite_image)

# to CSV
data = [['Image Name', 'Width', 'Height']]

for filename in os.listdir(Composite):
    if filename.endswith('.jpg'):
        img_path = os.path.join(Composite, filename)
        with Image.open(img_path) as img:
            width, height = img.size
            data.append([filename, width, height])

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)


# Initialize SIFT
sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=12)

# Loop through all the images inside Composite folder and extract their SIFT features
for filename in os.listdir(Composite):
    img_path = os.path.join(Composite, filename)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filter to the grayscale image
    gray_filtered = cv2.GaussianBlur(gray, (5, 5), 0)

    # Extract SIFT features
    kp, des = sift.detectAndCompute(gray_filtered, None)

    # Draw keypoints on the image
    img_with_kp = cv2.drawKeypoints(img, kp, None)

    # Save the image with keypoints
    output_path = os.path.join(Composite, filename)
    cv2.imwrite(output_path, img_with_kp)


# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Create FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load the test image and extract its SIFT features
test_img = cv2.imread(os.path.join(Test, "test.jpg")) # change image name to test
test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
test_kp, test_des = sift.detectAndCompute(test_gray, None)

# Apply scaling invariance, illumination variance, and rotation invariance to test descriptors
test_des /= (test_des.sum(axis=1, keepdims=True) + 1e-7)  # L2 normalization of descriptors

best_match = None
best_match_count = 0

# Loop through the images in the composite folder and match them with the test image
for filename in os.listdir(Composite):
    img_path = os.path.join(Composite, filename)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract SIFT features
    kp, des = sift.detectAndCompute(gray, None)

    # Apply scaling invariance, illumination variance, and rotation invariance to database descriptors
    des /= (des.sum(axis=1, keepdims=True) + 1e-7)  # L2 normalization of descriptors

    # Match the descriptors using FLANN
    matches = flann.knnMatch(test_des, des, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append(m)

    # Update the best match if the number of good matches is higher
    if len(good_matches) > best_match_count:
        best_match = filename
        best_match_count = len(good_matches)

print(f"Best match: {best_match}")

print("time elapsed: {:.2f}s".format(time.time() - start_time))