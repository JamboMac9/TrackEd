****************************************************************************
EMOTION RECOGNITION USING FACIAL EXPRESSIONS: A TOOL FOR E-MEETING PLATFORMS
****************************************************************************

***This setup assumes Python is installed and has been added as a PATH to the System Environment Variables (SEV)

===========================
Project Directory Structure
===========================
* tracked		
	* "model_weights.h5"
	===================================
	Emotion Classes for captured images
	===================================
	* emotion_class
		* angry
	* emotion_class
		* disgust
	* emotion_class
		* fear
	* emotion_class
		* happy
	* emotion_class
		* neutral
	* emotion_class
		* surprise
	* emotion_class
		* sad
	* session_stream
		* session_{timestamp}.mp4

* track_train_files
	* DataGenerator 		
		* archive
			* images"
	* CNNModel		
		* "model_weights.h5"
	* ModelEvaluator 		
		* "conf_matrix.png"
		* "conf_matrix_test.png"

=============================
QUICK SETUP & RUNNING TrackEd
=============================
For a quick setup that involves minimal installation, users are advised to install the Anaconda3 for TensorFlow environment. This allows the user to simply import the tf-gpu.yaml file which contains all the project dependencies directly into the environment. This will allow the packages to run without any further installation.

====================
Run within Anaconda3
====================
To run TrackEd within Anaconda3 environment, open 'tracked' folder and run the TrackEd.py file. This will run using the device webcam, giving the option to 'Play' or 'Quit' - Application will only run when user clicks the Play button - click the Play button again to stop the program. Quit exits the application.

If user prefers a different IDE, please follow the **Installation Guide** below.

===================
Run on Command Line
===================
To run TrackEd through CMD - if all package requirements are installed and set to active PATHS on the system, open command line (CMD) > navigate to the project folder > type

***python TrackEd.py***

If set up correctly, this will start the application as intended.

If using Anaconda3, activate the environment first > open CMD > type

conda activate <env name (tf-gpu if importing preset env)>

Once active type >

***python TrackEd.py***

For instructions on how to RUN a Python file through Command Prompt (CMD) and how to SET Python to the SEV, please see: https://www.wikihow.com/Use-Windows-Command-Prompt-to-Run-a-Python-File

======================
TRAIN AND TEST TrackEd
======================

To train, first the data needs to be imported.

The original dataset used in this project is the Oheix FER2013 dataset, which contains 35,887 grayscale images of size 48x48 pixels.
The dataset is split into three sets: training, validation, and test sets. You can download the dataset from:

https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

This particular dataset is split into Train and Validation folders - this needs to be set to Train and Test, with the Train folder using an 80:20 Train-Validation split, with the directory as below:

* tracked_train_files
	* archive	
		* 1.train
		* 2.validation
		* 3.test

Once data has been imported, simply open the Anaconda3 environment, import the tf-gpu.yaml file, open and run __main.py.

==================
Installation Guide
==================

If user is not using the tf-gpu.yaml file then to run this code the following packages and libraries need to be installed:

NumPy
Matplotlib
TensorFlow
Keras
Scikit-learn
IPython

User can install these packages using pip through CMD. For example:

***pip install numpy matplotlib tensorflow keras scikit-learn ipython***

============================
Package Version Requirements
============================

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

=============
Project Files
=============

TrackEd.py: this class is for the application - run this to run the project application.

__main.py: this is the training model for the project - run this to train the specified dataset.

CNNModel.py		)
ModelEvaluator.py	>  these classes run in conjuction with __main.py
DataGenerator.py	)

tf-gpu.yaml

emotion_class		- store images for emotion classification
plots			- store plots from test and the best model_weights.h5 file
session_stream		- store the session stream
