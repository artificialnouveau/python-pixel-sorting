import argparse
import cv2
import numpy as np
from PIL import Image
from random import choice
import matplotlib.pyplot as plt

def pixel_sort(image, direction):
    sorted_pixels = np.sort(image, axis=direction)
    sorted_image = Image.fromarray(sorted_pixels)
    return sorted_image

# Create ArgumentParser object and add arguments
parser = argparse.ArgumentParser(description='Perform object segmentation and pixel sorting on an image.')
parser.add_argument('image_path', help='The path to the image file.')

# Parse the arguments
args = parser.parse_args()

# Load the image
image = cv2.imread(args.image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection
edges = cv2.Canny(blur, 50, 200)

# Perform a dilation and erosion to close gaps in between object edges
dilation = cv2.dilate(edges, None, iterations=2)
erosion = cv2.erode(dilation, None, iterations=1)

# Find contours in the erosion and initialize the mask to segment the object from the image
contours, _ = cv2.findContours(erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(erosion.shape, dtype="uint8")

# Find the largest contour and draw it on the mask
largest_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(mask, [largest_contour], -1, 255, -1)

# Remove the contours from the image and show the resulting images
image = cv2.bitwise_and(image, image, mask=mask)

# Convert back to PIL image for pixel sorting
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Perform pixel sorting
direction = choice([0, 1])  # Randomly choose a direction
sorted_image = pixel_sort(np.array(pil_image), direction)

# Display the image
plt.imshow(sorted_image)
plt.show()
