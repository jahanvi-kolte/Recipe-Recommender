from keras import models
from models import cdae
import os
import numpy as np
import time

class CDAE():

    def __init__(self, input_dim, hidden_dim=200, hidden_activation='sigmoid',
                 output_activation='sigmoid', loss='binary_crossentropy', optimizer='adam', dropout_prob=0.5,
                 alpha=0.01):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # round(input_dim * hidden_dim_fraction_of_input)
        self.dropout_prob = dropout_prob
        self.alpha= alpha
        self.model = cdae.create_cdae_model(input_dim, self.hidden_dim, hidden_activation, output_activation,
                                                  loss, optimizer, dropout_prob, alpha)



    def recommend_by_user_likes(self, user_profile, topn=10):
        prediction = self.model.predict(x=user_profile)

        #Only select the items not liked so far
        prediction = prediction * (user_profile == 0)

        #Get items in sorted order based on their prediction value
        prediction = np.argsort(prediction)[:, ::-1]

        return np.array(prediction[:, :topn])

    def train_model(self, train_data, batch_size=1000, epochs=200):
        return self.model.fit(x=train_data, y=train_data, validation_split=0.1, batch_size=batch_size, epochs=epochs, shuffle=True,
                              verbose=1)

    def save_model(self, folder_path, model_label):
        model_name = '_'.join([model_label, str(self.hidden_dim),str(time.time())]) + '.h5'
        self.model.save(os.path.join(folder_path, model_name))

    def load_model(self, model_path):
        self.model = models.load_model(model_path)
        self.model._make_predict_function()


if __name__ == '__main__':
    print("CDAE class")