# RealTimeSignRecognition

This project was implemented as part of the project "Stream Data Processing" at Wrocław University of Science and Technology.
It was created to recognize NicIcons on real time.

## Technologies
* Anaconda python
* Keras
* TENSORFLOW
* NVIDIA CUDA
* NVIDIA CUDNN

## Local usage

To use our app locally you should clone this repository
If you choose to use anaconda distibution of python use the following( in environment of your choosing):
1. conda install tensorflow (tensorflow-gpu if you choose to use GPU acceleration)
2. conda install keras (keras-gpu accordingly)
3. install and configure CUDA and CUDNN to work with tensorflow

## File description

1. Run ModelFit.py to teach the model given (adjust path to database and database size)
2. Run Tester.py to validate model (adjust as above)
3. DataAugmentation.py Run this file perform data augmentation and save it to hard drive 
4. ConfusionMatrix.py constructs confusion matrix for model
