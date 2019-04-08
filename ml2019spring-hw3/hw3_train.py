from keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import pickle
from keras import initializers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator

def preprocess_x(train):
    train_x = []
    for f in train['feature']:
        temp = np.array(f.split(' ')).astype(int)
        train_x.append(temp)
    
    train_x = np.array(train_x)
    train_x = train_x.reshape((train_x.shape[0],48,48,1))
    return train_x
def output_result(filename,predict_value):
    id_ = []
    for i in range(predict_value.shape[0]):
        id_.append(i)
    output = pd.DataFrame(columns=['id','label'])
    output['id'] = id_
    output['label'] = predict_value
    output.to_csv(filename,index = False)

def preprocess_y(train):
    train_y = pd.to_numeric(train['label'])
    train_y = to_categorical(train_y)
    return train_y

def get_new_model():
    model = Sequential() # initializing CNN
    model.add(Convolution2D(64, kernel_size=3, strides=1, padding="same", input_shape=(48, 48, 1), kernel_initializer=initializers.he_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))
    
    model.add(Convolution2D(128, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))
    
    model.add(Convolution2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))
    
    model.add(Convolution2D(512, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    
    model.add(Dense(512, kernel_initializer=initializers.he_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, kernel_initializer=initializers.he_normal(seed=None)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(7))
    model.add(Activation("softmax"))
    optim = Adam()
    
    model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == '__main__':
    train_file = sys.argv[1]
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
    session = tf.Session(config=config)
    KTF.set_session(session)
    
    train = pd.read_csv(train_file,engine='python')
    train_x = preprocess_x(train)
    train_y = preprocess_y(train)
    
    floatZoomRange = 0.2
    genTrain = ImageDataGenerator(rotation_range=25,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.1,
                                  zoom_range=[1-floatZoomRange, 1+floatZoomRange],
                                  horizontal_flip=True)
                                  genTrain.fit(train_x)
                                  model.fit_generator(genTrain.flow(train_x/255, train_y, batch_size=128),
                                                      steps_per_epoch=5*train_x.shape[0]//128,
                                                      verbose=1,
                                                      epochs=150, shuffle=True)
                                  model.save('model_data_gen_5.h5')
