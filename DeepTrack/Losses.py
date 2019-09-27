from tensorflow import keras

K = keras.backend

def mae(T, P):
    T = K.flatten(T)
    P = K.flatten(P)
    return K.mean(K.abs(P - T))

def mse(T, P):
    T = K.flatten(T)
    P = K.flatten(P)
    return K.mean(K.square(P - T))

def rmse(T, P):
    T = K.flatten(T)
    P = K.flatten(P)
    return K.root(K.mean(K.square(P - T)))