import os

import pandas as pd


class TimeSeriesDatasetLoader:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def translate_emotion(self, file_path, type_='default'):
        filename = file_path.split('/')[-1]
        token = filename.split('-')[0]

        if type_ == 'default':
            return {
                'ale': 0,
                'des': 1,
                'med': 2,
                'neu': 3,
                'rai': 4,
                'sur': 5,
                'tri': 6
            }[token]

        elif type_ == 'emotion_type':
            if token in ['ale', 'sur']:
                return 0
            if token == 'neu':
                return 1
            if token in ['des', 'rai', 'tri', 'med']:
                return 2

        elif type_ == 'russel':
            if token == 'neu':
                return 0
            if token == 'ale':
                return 1
            if token == 'sur':
                return 2
            if token in ['des', 'rai', 'med']:
                return 3
            if token == 'tri':
                return 4

    def get_all_filepaths(self):
        result_filepaths = []

        for inst in os.listdir(self.dataset_path):
            recursive_file_instances = []
            if os.path.isdir("{}/{}".format(self.dataset_path, inst)):
                recursive_file_instances = self.get_all_filepaths("{}/{}".format(self.dataset_path, inst))
                for filepath in recursive_file_instances:
                    result_filepaths.append(filepath)

            else:
                result_filepaths.append("{}/{}".format(self.dataset_path, inst))

        return result_filepaths + recursive_file_instances

    def get_dataset(self, type_='default'):
        X_dataset = []
        Y_dataset = []
        file_paths = self.get_all_filepaths()

        for file_path in file_paths:
            try:
                inst = pd.read_csv(file_path, delimiter=';')
                emotion = self.translate_emotion(file_path, type_)

                X_dataset.append(inst.values)
                Y_dataset.append(int(emotion))
            except:
                pass

        return X_dataset, Y_dataset

    def get_genre_split_dataset(self):
        X_dataset = []
        Y_dataset = []
        file_paths = self.get_all_filepaths()

        for file_path in file_paths:
            try:
                inst = pd.read_csv(file_path, delimiter=';')
                emotion = self.translate_emotion(file_path)
                genre = self.get_genre(file_path)
                X_dataset.append(inst.values)
                Y_dataset.append([int(emotion), genre])
            except:
                pass

        return X_dataset, Y_dataset

    @staticmethod
    def get_genre(file_path):
        file_name = file_path.split('/')[-1]
        token = file_name.split('-')[1]
        if token[0] == 'f':
            return 'F'
        else:
            return 'M'
