# Table of Content

## Problem statement 
In the world of sports and entertainment, the increasing volume of digital content has created a demand for efficient and accurate ways to organize and categorize images.  One particular challenge is automating the classification of images featuring sports celebrities. 

## Objective
The aim of this project is to create a system that can accurately identify and categorize images of the athletes.

## Dataset
The dataset employed in this project is obtained from the google by using fatkun. Fatkun Batch Download Image is a Google Chrome extension specifically designed for downloading multiple images from a web page at once. By using this extension i have downloaded images of different sport celebrities which contains MS Dhoni, Virat Kohli, PV Sindhu, Maria Sharapova and Roger Federer.
This images is used for further process.

## Data Pipeline

* Data Collection: Gather images of sport celebrities from google. I have downloaded 99 images of each sport celebrity.

* Data Cleaning: Remove duplicate images to ensure dataset cleanliness. Verify the quality and correctness of image annotations or labels associated with each sport celebrity.
Handle any missing or blur images to create a reliable dataset.Basically all peoples is identified by their face. So based on these assumption load one image on IDE and then by using cascade function of opencv detect the eyes and the face of image. The Haar cascade function is applied to all the images present in the dataset. This step leads to get croppd images of all dataset.

* Feature Engineering: In this step first input image is convert to array.The image is then converted to grayscale using OpenCV's cv2.cvtColor function.The pixel values are converted to floating-point format and normalized to the range [0,1]. The two dimensional i.e. 2D wavelet transform is applied to the preprocessed image using the Haar wavelet.The pywt.wavedec2 function computes the wavelet coefficients. The wavelet coefficients are stored in a list (coeffs_H).The approximation coefficients (coeffs_H[0]) are set to zero because it effectively removing the low-frequency components. The inverse wavelet transform (pywt.waverec2) is applied to reconstruct the image. The 'w2d' function is used to apply a 2D Haar wavelet transform to the input image and returns the image after removing low-frequency components, effectively capturing high-frequency details in the image.

* Model Training: I have used CNN (convolutional nural network) model for training which is best suit for image classification model. In this i have import required libraries for model training. An ImageDataGenerator is created to perform data augmentation, which artificially increases the diversity of the training dataset by applying random transformations to the images. 
The data is split into training and testing sets using train_test_split. Pixel values in the image data are normalized to be between 0 and 1. The data is reshaped to fit the input requirements of the CNN. A sequential model is created using Keras.Convolutional layers (Conv2D) with ReLU activation, max-pooling layers (MaxPooling2D), batch normalization (BatchNormalization), a flatten layer (Flatten), and dense layers (Dense) with ReLU activation are added to the model. A dropout layer (Dropout) is included to prevent overfitting. The output layer has a softmax activation function with 5 units,because it's a classification task with 5 classes.The model is trained using the fit method. Model is saved into pickle file using joblib module.


## update
Use the image_classifier.ipynb file as the updated version, as it has achieved the best testing accuracy compared to the previous model. Additionally, I have used hyperparameter tuning to enhance the accuracy.
