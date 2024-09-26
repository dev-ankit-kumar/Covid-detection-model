import cv2
import numpy as np
import argparse
import tensorflow.compat.v1 as tf

# Disable TensorFlow 2.x behavior
tf.disable_v2_behavior()

# Enable eager execution
tf.enable_eager_execution()

# Importing only needed modules
from tensorflow.keras.models import load_model

# Define a function for testing the image
def test(img_path):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Load the pre-trained model
    model = load_model('covid19.h5')

    # Read and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    result_img = cv2.resize(img, (600, 600))
    img = np.reshape(img, [1, 224, 224, 3])

    # Predict the result
    array = model.predict(img)
    result = array.argmax(axis=-1)

    # Map the result to a label
    prediction = 'normal' if result[0] == 1 else 'covid'
    print("Result:", prediction)

    # Determine the color for the text based on the prediction
    color = (0, 255, 0) if prediction == 'normal' else (0, 0, 255)

    # Add text to the result image
    cv2.putText(result_img, prediction, (25, 25), font, 1, color, 2, cv2.LINE_AA)

    return result_img

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to testing x-ray image")
args = vars(ap.parse_args())

# Perform the test and get the result image
result_img = test(args["image"])

# Show the result image
cv2.imshow("Result", result_img)
cv2.waitKey(0)

# Save the model
# model.save('path/to/save/model.h5')
