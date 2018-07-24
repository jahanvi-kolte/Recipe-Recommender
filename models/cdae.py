from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l2

def create_cdae_model(input_dim, hidden_dim, hidden_activation, output_activation, loss, optimizer, dropout_prob, alpha):

    inputs = Input((input_dim,), name='Utility_Matrix')
    modified_input = Dropout(dropout_prob)(inputs)
    hidden_nodes = Dense(hidden_dim, activation=hidden_activation, kernel_regularizer=l2(alpha), bias_regularizer=l2(alpha))(
        modified_input)
    output = Dense(input_dim, activation=output_activation)(hidden_nodes)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


