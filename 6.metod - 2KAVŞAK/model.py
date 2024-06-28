import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim_1, output_dim_1, input_dim_2, output_dim_2):
        self._input_dim_1 = input_dim_1
        self._output_dim_1 = output_dim_1
        self._input_dim_2 = input_dim_2
        self._output_dim_2 = output_dim_2
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model_1 = self._build_model_1(num_layers, width)
        self._model_2 = self._build_model_2(num_layers, width)


    def _build_model_1(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        inputs = keras.Input(shape=(self._input_dim_1,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim_1, activation='linear')(x)

        model1 = keras.Model(inputs=inputs, outputs=outputs, name='my_model_1')
        model1.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return model1
    
    
    def _build_model_2(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        inputs = keras.Input(shape=(self._input_dim_2,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim_2, activation='linear')(x)

        model2 = keras.Model(inputs=inputs, outputs=outputs, name='my_model_2')
        model2.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return model2
    

    def predict_one_1(self, state1):
        """
        Predict the action values from a single state
        """
        state1 = np.reshape(state1, [1, self._input_dim_1])
        return self._model_1.predict(state1)
    
    
    def predict_one_2(self, state2):
        """
        Predict the action values from a single state
        """
        state2 = np.reshape(state2, [1, self._input_dim_2])
        return self._model_2.predict(state2)


    def predict_batch_1(self, states1):
        """
        Predict the action values from a batch of states
        """
        return self._model_1.predict(states1)
    
    
    def predict_batch_2(self, states2):
        """
        Predict the action values from a batch of states
        """
        return self._model_2.predict(states2)


    def train_batch_1(self, states1, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model_1.fit(states1, q_sa, epochs=1, verbose=0)
        
        
    def train_batch_2(self, states2, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model_2.fit(states2, q_sa, epochs=1, verbose=0)


    def save_model_1(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model_1.save(os.path.join(path, 'trained_model_1.h5'))
        plot_model(self._model_1, to_file=os.path.join(path, 'model_1_structure.png'), show_shapes=True, show_layer_names=True)
        
        
    def save_model_2(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model_2.save(os.path.join(path, 'trained_model_2.h5'))
        plot_model(self._model_2, to_file=os.path.join(path, 'model_2_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim_1(self):
        return self._input_dim_1


    @property
    def output_dim_1(self):
        return self._output_dim_1
    
    
    @property
    def input_dim_2(self):
        return self._input_dim_2


    @property
    def output_dim_2(self):
        return self._output_dim_2


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim_1, input_dim_2, model_path):
        self._input_dim_1 = input_dim_1
        self._input_dim_2 = input_dim_2
        self._model_1 = self._load_my_model_1(model_path)
        self._model_2 = self.load_my_model_2(model_path)


    def _load_my_model_1(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model_1.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")
            
            
    def _load_my_model_2(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model_2.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one_1(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim_1])
        return self._model_1.predict(state)
    
    
    def predict_one_2(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim_2])
        return self._model_2.predict(state)


    @property
    def input_dim_1(self):
        return self._input_dim_1
    
    @property
    def input_dim_2(self):
        return self._input_dim_2