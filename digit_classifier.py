import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping


class DigitClassifier:
    def __init__(self):
        self.model = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, (3,3), padding='same',
                       activation='tanh', input_shape=(28, 28, 1)))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(32, (3,3), padding='same',
                       activation='tanh'))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(128, (3,3), padding='same',
                       activation='tanh'))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(256, (3,3), padding='same',
                       activation='tanh'))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        
        loss = SparseCategoricalCrossentropy()
        optimizer = Adam(0.0001)
        # optimizer = Adam(learning_rate=1e-3)
        self.model.compile(loss=loss, optimizer=optimizer,
                           metrics=[SparseCategoricalAccuracy()])
        print(self.model.summary())

    def load_model(self):
        self.model = load_model('models/digit_classifier.hdf5')

    def save_model(self):
        self.model.save('models/digit_classifier.hdf5')

    def train(self, x, y, **kwargs):
        x = x.reshape(-1, 28, 28, 1)
        es = EarlyStopping(patience=5)
        self.model.fit(x, y, validation_split=0.2, batch_size=32, epochs=50, callbacks=[es])

    def predict(self, x_test):
        """
        :param x_test: a numpy array with dimension (N,D)
        :return: a numpy array with dimension (N,)
        """
        x_test = x_test.reshape(-1, 28, 28, 1)
        y_predict = self.model.predict(x_test, batch_size=32)
        print(y_predict.argmax(axis=1))
        return y_predict.argmax(axis=1)

    def evaluate(self, x_test, y_test):        
        x_test = x_test.reshape(-1, 28, 28, 1)
        return self.model.evaluate(x_test, y_test)
