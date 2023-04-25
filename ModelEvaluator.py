"""
Author: Jamie McGrath

A class for evaluating and visualising the performance of a machine learning 
model. Providing methods for plotting and analysing the training and validation 
accuracy/loss, generating and visualising confusion matrices, obtaining the 
best epoch during training, and loading the best model from a file.

Params:
~ history (tf.keras.callbacks.History):
    training history of the model.
~ model (tf.keras.models.Model):
    trained machine learning model.
~ valid_generator (tf.keras.preprocessing.image.ImageDataGenerator):
    image data generator for validation data.
~ test_generator (tf.keras.preprocessing.image.ImageDataGenerator):
    image data generator for test data.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

class ModelEvaluator:

    def __init__(self, history, model, valid_generator, test_generator):
        """
        Constructor for the ModelEvaluator class.

        Args:
            history (History): history object obtained after training the model.
            model (Model): trained CNN model to be evaluated.
            valid_generator (Sequence): validation data generator used for evaluation.
            test_generator (Sequence): test data generator used for evaluation.
        """
        self.history = history
        self.model = model
        self.valid_generator = valid_generator
        self.test_generator = test_generator

    def plot_loss_acc(self, history):
        """
        Plot the training and validation accuracy and loss curves.

        Params:
            history (tf.keras.callbacks.History): 
                training history of the model.
        """
        # get the training and validation accuracy and loss from the model history
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # create a range of epochs
        epochs = range(1, len(acc) + 1)

        # plot the training and validation accuracy and loss
        plt.figure(figsize=(24, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Epoch')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Epoch')
        plt.show()

    def plot_valid_matrix(self, model, valid_generator):
        """
        Plot the confusion matrix and print the classification report of the 
        validation data.

        Params:
            model (tf.keras.models.Model): The trained machine learning model.
            valid_generator (tf.keras.preprocessing.image.ImageDataGenerator): 
                image data generator for validation data.
        """
        # predict the classes of the validation data
        y_pred = np.argmax(model.predict(valid_generator), axis=-1)
        y_true = valid_generator.classes
        class_labels = list(valid_generator.class_indices.keys())

        # print the classification report of the validation data
        print(classification_report(y_true, y_pred, target_names=class_labels))

        # create and show the confusion matrix of the validation data
        conf_matrix = confusion_matrix(y_true, y_pred)
        display = ConfusionMatrixDisplay(conf_matrix, 
            display_labels=valid_generator.class_indices)
        display.plot()
        plt.show()

        # save the confusion matrix plot
        plt.savefig('conf_matrix.png')
        plt.close()

        # print the confusion matrix of the validation data
        print(conf_matrix)

    def plot_test_matrix(self, model, test_generator):
        """
        Plot the confusion matrix and print the classification report of the 
        test data.

        Params:
            model (tf.keras.models.Model): The trained machine learning model.
            test_generator (tf.keras.preprocessing.image.ImageDataGenerator): 
                image data generator for test data.
        """
        # predict the classes of the test data
        Y_test = model.predict(test_generator)
        y_test = np.argmax(Y_test, axis=1)

        # print the classification report of the test data
        print('Classification Report')
        print(classification_report(test_generator.classes, y_test))

        # create and show the confusion matrix of the test data
        print('Confusion Matrix')
        cm = confusion_matrix(test_generator.classes, y_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
            display_labels=test_generator.class_indices)
        disp.plot()
        plt.show()

        # save the confusion matrix plot
        plt.savefig('conf_matrix_test.png')
        plt.close()

    def get_best_epoch(self):
        '''
        This function retrieves the best epoch during the model training process 
        based on the validation accuracy.
        
        Params:
            None - uses the self.history attribute of the ModelEvaluator instance
        '''
        valid_acc = self.history.history['val_accuracy']
        best_epoch = valid_acc.index(max(valid_acc)) + 1
        best_acc = max(valid_acc)
        print(
            'Best Validation Accuracy Score {:0.5f}, for epoch {}'.format(
                best_acc, best_epoch))
        return best_epoch

    def load_best_model(self, file_path):
        '''
        This function loads the best trained model from a file specified by the 
        file_path parameter.
        
        Params:
            file_path: string representing the file path of the saved model.
        '''
        self.model = load_model(file_path)
