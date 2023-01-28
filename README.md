## Brief Description

This code is a Kivy application that uses a TensorFlow model to detect cracks in images. The model is a pre-trained MobileNet V2 model that has been fine-tuned on a dataset of images of cracked and non-cracked surfaces. The app has three main screens: the "Homepage", "Detectionpage", and "Metricspage". The Homepage serves as the starting point for the app, and the Detectionpage allows the user to select an image and run the model on it to get a prediction. The Metricspage shows metrics like precision, recall, and F1 score for the model. The app also uses the tkinter library to open a file dialog and select an image. The app also uses the keras library to load the model and perform the prediction. The app also uses custom metrics recall, precision and f1 defined in the code to evaluate the model performance.


## Requirements

To install the packages listed in a requirements.txt file, you can use the following command:
   
   
    pip install -r requirements.txt
    

## Usage

To run the app, you can use the following command:
   
  
    python3 app.py
    

