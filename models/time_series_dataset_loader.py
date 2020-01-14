import numpy as np
import pandas as pd
import os

class TimeSeriesDatasetLoader:
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def translate_emotion(self, filepath):
        filename = filepath.split('/')[-1]
        token = filename.split('-')[0]
        return {
            'ale': 0,
            'des': 1,
            'med': 2,
            'neu': 3,
            'rai': 4,
            'sur': 5,
            'tri': 6
        }[token]    
        
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
    
    def get_dataset(self):
        X_dataset = []
        Y_dataset = []
        filepaths = self.get_all_filepaths()
        
        for filepath in filepaths:
            inst = pd.read_csv(filepath, delimiter=';')
            emotion = self.translate_emotion(filepath)
            X_dataset.append(inst.values)
            Y_dataset.append(int(emotion))
        
        return X_dataset, Y_dataset
                