import numpy as np   # We recommend to use numpy arrays
from sklearn.base import BaseEstimator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense


class model(BaseEstimator):
   

    def __init__(self):
       
        self.num_train_samples = 0
        self.num_feat = 1
        self.num_labels = 1
        self.is_trained = False

        self.model = Sequential()
        self.model.add(Conv2D(128, (3, 3), input_shape=(40, 40, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256))

        self.model.add(Dense(128))

        self.model.add(Dense(64))

        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                           metrics=['accuracy'])

    def fit(self, X, y):
      
        self.num_train_samples = X.shape[0]
        X = X.reshape((self.num_train_samples, 40, 40, 3))
        self.model.fit(X / 255.0, y, epochs=12)
        self.is_trained = True

    def predict(self, X):
      
        num_test_samples = X.shape[0]
        X = X.reshape((num_test_samples, 40, 40, 3))
        return self.model.predict_proba(X / 255.0)