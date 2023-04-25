****************************************************************************
EMOTION RECOGNITION USING FACIAL EXPRESSIONS: A TOOL FOR E-MEETING PLATFORMS
****************************************************************************

***This setup assumes Python is installed and has been added as a PATH to the System Environment Variables (SEV)

For instrvutions on how to RUN a Python file through Command Prompt (CMD) and how to SET Python to the SEV, please see: https://www.wikihow.com/Use-Windows-Command-Prompt-to-Run-a-Python-File

This code is for training a Convolutional Neural Network (CNN) to classify emotions from facial expressions using the Oheix FER2013 dataset.

************
Installation
************

To run this code, you will need to install the following packages and libraries:

NumPy
Matplotlib
TensorFlow
Keras
Scikit-learn
IPython

You can install these packages using pip through CMD. For example:

pip install numpy matplotlib tensorflow keras scikit-learn ipython

****************************
Package Version Requirements
****************************

ipython==7.34.0
keras==2.11.0
matplotlib==3.7.1
numpy==1.24.1
opencv_python==4.6.0.66
Pillow==9.5.0
scikit_learn==1.2.2
tensorflow==2.9.1

For further info and how to utilise GPU or CPU for Tensorflow:

https://www.tensorflow.org/install/pip

**Please note, if running the training program __main.py, GPU is much faster than CPU if you have the ability.

****
DATA
****

The original dataset used in this project is the Oheix FER2013 dataset, which contains 35,887 grayscale images of size 48x48 pixels.
The dataset is split into three sets: training, validation, and test sets. You can download the dataset from:

https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

*************
Project Files
*************

EmotionDetector.py: this class is for the application - run this to run the project application.

__main.py: this is the training model for the project - run this to train the specified dataset.

CNNModel.py		)
ModelEvaluator.py	>  these classes run in conjuction with __main.py
DataGenerator.py	)

tf-gpu.yaml

archive			- store images for the CNN model
emotion_class		- store images for emotion classification
plots				- store plots from test and the best model_weights.h5 file
session_stream		- store the session stream

*****************
Project Directory
*****************
EmotionDetector 	- "plots/1.Model/model_weights.h5"
			- "session_stream/session_{timestamp}.mp4"
DataGenerator 	- "archive/images"
CCNModel		- "model_weights.h5"
ModelEvaluator 	- "conf_matrix.png"
			- "conf_matrix_test.png"


*********************
Anaconda Environments
*********************

If using Anaconda3 for TensorFlow environment, tf-gpu.yaml has the env packages pre-installed - importing the .yaml file into Anaconda3 as an environment will allow the packages to run with no installation required.

***
CMD
***

To run the application through CMD - if all package requirements are installed and set to active PATHS on the system, open command centre > navigate to the project folder > type

python EmotionDetector.py

If set up correctly, this will start the application as intended.

If using Anaconda3, activate the environment first > open CMD > type

conda activate <env name (tf-gpu if importing preset env)>

Once active type >

python EmotionDetector.py

