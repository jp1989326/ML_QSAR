from keras.layers import Input, Dense, Lambda
from keras.models import Model

x = Input(shape=(666, ))
h1 = Dense(2, activation = 'relu')(x)
o1 = Dense(1, activation = 'relu')(h1)
o2 = Dense(1, activation = 'relu')(h1)
model = Model(inputs=[x], outputs=[o1, o2])
