import os

import pandas as pd


class DatasetLoader:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def translate_emotion(self, file_path, type_='default'):
        filename = file_path.split('/')[-1]
        token = filename.split('-')[0]

        return {
            'neu': 0,
            'des': 1,
            'med': 2,
            'ale': 3,
            'rai': 4,
            'sur': 5,
            'tri': 6
        }[token]


    def get_all_filepaths(self, path):
        result_filepaths = []
        for inst in os.listdir(path):
            recursive_file_instances = []
            if os.path.isdir("{}/{}".format(self.dataset_path, inst)):
                recursive_file_instances = self.get_all_filepaths("{}/{}".format(path, inst))
                for filepath in recursive_file_instances:
                    result_filepaths.append(filepath)

            else:
                result_filepaths.append("{}/{}".format(path, inst))

        return result_filepaths + recursive_file_instances

    def get_dataset(self):
        X_dataset = []
        Y_dataset = []
        file_paths = self.get_all_filepaths(self.dataset_path)

        for file_path in file_paths:
            try:
                actor, gender, filename = self.get_extra_info(file_path)
                inst = pd.read_csv(file_path, delimiter=';')
                X_dataset.append(inst.values[2:])
                Y_dataset.append([self.translate_emotion(file_path), actor, filename])
            except:
                pass

        return X_dataset, Y_dataset

    def get_extra_info(self, file_path):
        filename = file_path.split('/')[-1]
        actor = filename.split('-')[1]
        gender = filename.split('-')[1][0]
        return actor, gender, filename

    @staticmethod
    def get_genre(file_path):
        file_name = file_path.split('/')[-1]
        token = file_name.split('-')[1]
        if token[0] == 'f':
            return 'F'
        else:
            return 'M'

    def get_emotion(self, file_path):
        filename = file_path.split('/')[-1]
        emotion_token = filename.split('-')[0]
        return emotion_token