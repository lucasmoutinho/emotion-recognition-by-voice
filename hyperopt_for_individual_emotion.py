import json
from datetime import datetime

import numpy as np
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential

from modules.dataset_manipulation import DatasetManipulation
from modules.metrics import CustomMetrics
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from modules.dataset_loader import DatasetLoader
import os



def data():
    WANTED_EMOTION = 0
    dataset = DatasetManipulation()
    X_train, y_train, X_test, y_test = dataset.get_train_test_split_for_emotion(WANTED_EMOTION)
    return X_train, y_train, X_test, y_test


def create_model(X_train, y_train, X_test, y_test):
    WANTED_EMOTION = 0
    WANTED_EMOTION_NAME = {
        '0': 'neu',
        '1': 'des',
        '2': 'med',
        '3': 'ale',
        '4': 'rai',
        '5': 'sur',
        '6': 'tri'
    }[str(WANTED_EMOTION)]
    RUN_NAME = WANTED_EMOTION_NAME + datetime.now().strftime("%Y%m%d%H%M")
            
    y_test_labels = [np.argmax(y_inst) for y_inst in y_test]

    num_rows = X_train[0].shape[0]
    num_columns = X_train[0].shape[1]
    num_channels = 1

    batch_size = 128
    epochs = 300
        
    model = Sequential()
    model.add(Conv2D(filters={{choice([32, 64, 128, 256])}},
                     kernel_size=2,
                     input_shape=(num_rows, num_columns, num_channels),
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout({{choice([0.2,0.4,0.6])}}))

    model.add(Conv2D(filters={{choice([32, 64, 128, 256])}},
                     kernel_size=2,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout({{choice([0.2,0.4,0.6])}}))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(2, activation='softmax'))
        
    # Compile the keras model
    
    if not os.path.exists("./hyperopt_results/{}".format(WANTED_EMOTION_NAME)):
        os.makedirs("hyperopt_results/{}".format(WANTED_EMOTION_NAME))
        

    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['adam', 'sgd', 'rmsprop'])}},
                  metrics=['accuracy'])

    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                  patience=20, min_lr=0.000001)
    mcp_save = ModelCheckpoint("hyperopt_results/{}/{}.h5".format(WANTED_EMOTION_NAME, RUN_NAME),
                               save_best_only=True, monitor='val_accuracy',
                               mode='max')
    result = model.fit(X_train, y_train, batch_size=batch_size,
                       epochs=epochs, validation_data=(X_test, y_test),
                       callbacks=[mcp_save, lr_reduce], verbose=1)
    
    validation_acc = np.amax(result.history['val_accuracy'])
    model.load_weights("hyperopt_results/{}/{}.h5".format(WANTED_EMOTION_NAME, RUN_NAME))

    print('Best validation acc of epoch:', validation_acc)
    info_best_validation_acc = validation_acc
    predictions = model.predict(X_test, verbose=1)
    predictions_labels = []
    for predict in predictions:
        predictions_labels.append(predict.tolist().index(np.amax(predict)))
    
    correct = 0
    total = len(predictions_labels)
    for index in range(0, len(predictions_labels)):
        if predictions_labels[index] == y_test_labels[index]:
            correct += 1
    
    
    print("average: {}".format(correct / total))
    info_average = correct / total

    info_dict = {
        'run_name': RUN_NAME,
        'best_validation_acc': info_best_validation_acc,
        'average': info_average
    }
        
    with open('hyperopt_results/{}/{}.json'.format(WANTED_EMOTION_NAME, RUN_NAME), 'w+') as outfile:
        json.dump(json.dumps(info_dict), outfile)
    
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def run():
    best_run, best_model = optim.minimize(
        model=create_model,
        data=data,
        algo=tpe.suggest,
        max_evals=20,
        trials=Trials()
    )

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(best_model)
    return best_run, best_model


best_run, best_model = run()
