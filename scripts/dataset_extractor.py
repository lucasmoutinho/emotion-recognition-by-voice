from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import os
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split # to split dataset into train and test
from sklearn.preprocessing import Normalizer
from numpy import loadtxt # Linear Algebra

DATABASE_NAME = "db1"
OUTPUT_PATH = "../../Output/{}/".format(DATABASE_NAME)
AUDIO_OUTPUT_FOLDER = "chroma"

def translate_emotion(token, database=None):
    if database == "db2":
        return int(token)
    else:
        return {
            'ale': 0,
            'des': 1,
            'med': 2,
            'neu': 3,
            'rai': 4,
            'sur': 5,
            'tri': 6
        }[token]

def translate_person(token):
    gender = 0 if token[0] == 'm' else 1
    person_id = next(iter(re.compile('([0-9])+').findall(token)))
    return gender, person_id

def translate_sentence(token):
    if token[0] == 's':
        stype = 0
    elif token[0] == 'l':
        stype = 1
    elif  token[0] == 'q':
        stype = 2
    else:
        stype = 3
        
    return stype, token

def build_dataset():
    dataset = []
    columns = None
    for folder in os.listdir("{}/{}".format(OUTPUT_PATH, AUDIO_OUTPUT_FOLDER)):                                                                             
        for filename in os.listdir("{}/{}/{}".format(OUTPUT_PATH, AUDIO_OUTPUT_FOLDER, folder)):
            filepath = "{}/{}/{}/{}".format(OUTPUT_PATH, AUDIO_OUTPUT_FOLDER, folder, filename)

            if DATABASE_NAME == "db1":
                filename_tokens = re.split('-|\.',filename)
                emotion = translate_emotion(filename_tokens[0])
                gender, person_id = translate_person(filename_tokens[1])
                sentence_type, sentence_id = translate_sentence(filename_tokens[2])
            else:
                emotion = translate_emotion(re.split('-|\.', filename)[2], "db2")

            csv_out = open(filepath)
                    
            document_content = csv_out.read()
            
            content = document_content[-2].split(',')
            csv_out.close()
            data_instance = content + [emotion]
            dataset.append(data_instance)
    
    ncolumns = [column for column in filter(lambda x: x != '', columns)]
    return dataset, ncolumns
            

dataset, columns = build_dataset()
df = pd.DataFrame(dataset, columns=columns) 
df.to_csv('emobase_dataset.csv')

