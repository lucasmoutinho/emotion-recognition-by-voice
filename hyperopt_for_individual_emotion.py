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


def data():
    wanted_emotion = WANTED_EMOTION
    dataset = DatasetManipulation()
    return dataset.get_train_test_split_for_emotion(wanted_emotion)


def create_model(X_train, y_train, X_test, y_test):
    RUN_NAME = WANTED_EMOTION_NAME + datetime.now().strftime("%Y%m%d%H%M")
    CHOICES_DICT = {
        'filter_1': {{choice([32, 64, 128, 256, 564])}},
        'filter_2': {{choice([32, 64, 128, 256, 564])}},
        'dropout_1': {{choice([0.2,0.4,0.6])}},
        'dropout_2': {{choice([0.2,0.4,0.6])}},
        'optimizer': {{choice(['adam', 'sgd', 'rmsprop', 'adadelta'])}}

    }
    num_rows = X_train[0].shape[0]
    num_columns = X_train[0].shape[1]
    num_channels = 1

    batch_size = 16
    epochs = 400

    model = Sequential()
    model.add(Conv2D(filters=CHOICES_DICT['filter_1'],
                     kernel_size=2,
                     input_shape=(num_rows, num_columns, num_channels),
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(CHOICES_DICT['dropout_1']))

    model.add(Conv2D(filters=CHOICES_DICT['filter_2'],
                     kernel_size=2,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout(CHOICES_DICT['dropout_2']))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(2, activation='softmax'))

    # Compile the keras model
    model.compile(loss='categorical_crossentropy',
                  optimizer=CHOICES_DICT['optimizer'],
                  metrics=['accuracy'])

    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                  patience=20, min_lr=0.000001)
    mcp_save = ModelCheckpoint(RUN_NAME+".h5",
                               save_best_only=True, monitor='val_accuracy',
                               mode='max')

    result = model.fit(X_train, y_train, batch_size=batch_size,
                       epochs=epochs, validation_data=(X_test, y_test),
                       callbacks=[mcp_save, lr_reduce], verbose=1)

    validation_acc = np.amax(result.history['val_accuracy'])

    model.load_weights(RUN_NAME+".h5")

    y_test_labels = [np.argmax(el) for el in y_test]

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
        'choices': CHOICES_DICT,
        'best_validation_acc': info_best_validation_acc,
        'average': info_average,
        'f1': CustomMetrics().f1(predictions_labels, y_test_labels)
    }

    with open('result_hyperopt.json', 'w+') as outfile:
        json.dump(json.dumps(info_dict), outfile)

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def run():
    best_run, best_model = optim.minimize(
        model=create_model,
        data=data,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials())

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(best_model)
    return best_run, best_model


best_run, best_model = run()
