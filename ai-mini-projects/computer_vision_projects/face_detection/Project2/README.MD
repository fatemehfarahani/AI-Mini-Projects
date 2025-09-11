# Face Detection in Images with Python (Colab)

This project demonstrates face detection in a set of images using Python and the `face_recognition` and `OpenCV` libraries on Google Colab.

## Project Goal

The main goal of this project is to practice using different face detection libraries in Python, process a folder of images, detect faces in each image, and sort the images based on the number of detected faces.

## Features

- Reads all images from a folder in Google Drive
- Resizes all images to a fixed size for easier processing
- Detects faces in each image using the `face_recognition` library
- Sorts and displays images from the least to the most faces detected
- Draws rectangles around detected faces in each image
- Shows the number of faces detected for each image

## Notes

- Some images may not have all faces detected, especially if the image quality is low, faces are too small, or the faces are at unusual angles.
- The main purpose of this project is to get familiar with using libraries like `face_recognition` and `OpenCV` for face detection in Python.

## Requirements

- Google Colab (recommended for easy installation)
- face_recognition
- opencv-python
- numpy
- matplotlib

## How to Use

1. Upload your images to a folder in your Google Drive (e.g., `MyDrive/img`).
2. Mount your Google Drive in Colab.
3. Run the notebook cells to install the required libraries, read the images, detect faces, and display the results.