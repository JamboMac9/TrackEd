'''
The main method for the Convolutional Neural Network - linking
the DataGenerator, CNModel and ModelEvaluator classes.

~ Class: DataGenerator
~ Description: generates data for training, validation, and testing.
~ Class methods: train_gen(), valid_gen(), test_gen()

~ Class: CNNModel
~ Description: defines and compiles the convolutional neural network 
(CNN) model for image classification.
~ Class methods: create_model(), compile_model()

~ Class: ModelEvaluator
~ Description: evaluates and visualises the performance of the 
trained CNN model.
~ Class methods: plot_loss_acc(), plot_valid_matrix(), plot_test_matrix()
'''

# importing required classes
from DataGenerator import DataGenerator
from CNNModel import CNNModel
from ModelEvaluator import ModelEvaluator

if __name__ == '__main__':
    
    # create objects of variables required
    data_gen = DataGenerator()
    train_gen = data_gen.train_gen()
    valid_gen = data_gen.valid_gen()
    test_gen = data_gen.test_gen()
    
    # create object of CNNModel class
    cnn_model = CNNModel(valid_gen, train_gen)

    # create and compile model
    cnn_model.create_model()
    history = cnn_model.compile_model()

    # create an object of ModelEvaluator class
    evaluator = ModelEvaluator(history, cnn_model.model, valid_gen, test_gen)
    
    # evaluate the model
    evaluator.plot_loss_acc(history)
    evaluator.plot_valid_matrix(cnn_model.model, valid_gen)
    evaluator.plot_test_matrix(cnn_model.model, test_gen)
