# MNIST Digit Classification with TensorFlow

This mini project demonstrates how to classify handwritten digits from the MNIST dataset using a simple neural network built with TensorFlow and Keras.

## Features

- Loads the MNIST dataset (images of handwritten digits 0-9)
- Normalizes the images for better training
- Compiles and evaluates a neural network model
- Visualizes a sample test image and shows the true and predicted label

## Requirements

- Python 3.x
- tensorflow
- numpy
- matplotlib

## How to Run

1. Install the required libraries:
    ```bash
    pip install tensorflow numpy matplotlib
    ```
2. Run the script:
    ```bash
    python mnist_digit_classification.py
    ```


*(You will also see the image of the digit with its true label)*

## Notes

- The model is trained and evaluated on the MNIST dataset.
- You can change the `index` variable to test the model on different test images.
- The code uses `np.argmax` to get the predicted digit from the model's output.

---

**Enjoy digit classification with TensorFlow!**