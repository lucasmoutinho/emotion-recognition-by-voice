import csv
from datetime import datetime
from time_series_model import TimeSeriesModel

vars = [
    {'dataset': 'Original',
     'features': "MFCC",
     'dataset_path': '../src/Features/Original/MFCC/',
     'checkpoint_file': '../models/model_checkpoints/{}original_mfcc_1.h5'
     },
    {'dataset': 'Original',
     'features': "MFCC 2x",
     'dataset_path': '../src/Features/Original/MFCC_2/',
     'checkpoint_file': '../models/model_checkpoints/{}original_mfcc_2.h5'
     },
    {'dataset': 'Original',
     'features': "MFCC 5x",
     'dataset_path': '../src/Features/Original/MFCC_5/',
     'checkpoint_file': '../models/model_checkpoints/{}original_mfcc_5.h5'
     },
    {'dataset': 'Original',
     'features': "MFCC 10x",
     'dataset_path': '../src/Features/Original/MFCC_10/',
     'checkpoint_file': '../models/model_checkpoints/{}original_mfcc_10.h5'
     },
    {'dataset': '1secsplit',
     'features': "MFCC",
     'dataset_path': '../src/Features/1secsplit/MFCC_1/',
     'checkpoint_file': '../models/model_checkpoints/{}1secsplit_mfcc_1.h5'
     },
    {'dataset': '1secsplit',
     'features': "MFCC 2x",
     'dataset_path': '../src/Features/1secsplit/MFCC_2/',
     'checkpoint_file': '../models/model_checkpoints/{}1secsplit_mfcc_2.h5'
     },
    {'dataset': '1secsplit',
     'features': "MFCC 5x",
     'dataset_path': '../src/Features/1secsplit/MFCC_5/',
     'checkpoint_file': '../models/model_checkpoints/{}1secsplit_mfcc_5.h5'
     },
    {'dataset': '1secsplit',
     'features': "MFCC 10x",
     'dataset_path': '../src/Features/1secsplit/MFCC_10/',
     'checkpoint_file': '../models/model_checkpoints/{}1secsplit_mfcc_10.h5'
     },
    {'dataset': '2secsplit',
     'features': "MFCC",
     'dataset_path': '../src/Features/2secsplit/MFCC/',
     'checkpoint_file': '../models/model_checkpoints/{}2secsplit_mfcc_1.h5'
     },
    {'dataset': '2secsplit',
     'features': "MFCC 2x",
     'dataset_path': '../src/Features/2secsplit/MFCC_2/',
     'checkpoint_file': '../models/model_checkpoints/{}2secsplit_mfcc_2.h5'
     },
    {'dataset': '2secsplit',
     'features': "MFCC 5x",
     'dataset_path': '../src/Features/2secsplit/MFCC_5/',
     'checkpoint_file': '../models/model_checkpoints/{}2secsplit_mfcc_5.h5'
     },
    {'dataset': '2secsplit',
     'features': "MFCC 10x",
     'dataset_path': '../src/Features/2secsplit/MFCC_10/',
     'checkpoint_file': '../models/model_checkpoints/{}2secsplit_mfcc_10.h5'
     },
    {'dataset': '3secsplit',
     'features': "MFCC",
     'dataset_path': '../src/Features/3secsplit/MFCC/',
     'checkpoint_file': '../models/model_checkpoints/{}3secsplit_mfcc_1.h5'
     },
    {'dataset': '3secsplit',
     'features': "MFCC 2x",
     'dataset_path': '../src/Features/3secsplit/MFCC_2/',
     'checkpoint_file': '../models/model_checkpoints/{}3secsplit_mfcc_2.h5'
     },
    {'dataset': '3secsplit',
     'features': "MFCC 5x",
     'dataset_path': '../src/Features/3secsplit/MFCC_5/',
     'checkpoint_file': '../models/model_checkpoints/{}3secsplit_mfcc_5.h5'
     },
    {'dataset': '3secsplit',
     'features': "MFCC 10x",
     'dataset_path': '../src/Features/3secsplit/MFCC_10/',
     'checkpoint_file': '../models/model_checkpoints/{}3secsplit_mfcc_10.h5'
     }
]

model_variations = [
    {'type': 'default', 'activation_layer_size': 7},
    {'type': 'emotion_type', 'activation_layer_size': 2},
    {'type': 'russel', 'activation_layer_size': 5}
]

results = []
for model_var in model_variations:
    for var in vars:
        model = TimeSeriesModel(var['dataset_path'], var['checkpoint_file'].format(model_var['type']), type_=model_var['type'],
                                activation_layer_size=model_var['activation_layer_size'])
        scores = model.run()
        scores['dataset'] = var['dataset']
        scores['model_variation'] = model_var['type']
        scores['features'] = var['features']
        results.append(scores)

csv_columns = ['dataset', 'model_variation', 'features', 'val_accuracy', 'val_top3_acc', 'accuracy', 'top3_acc']
csv_file = "results{}.csv".format(datetime.now().strftime("%d%m%Y%H%M"))
with open(csv_file, 'w+') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for data in results:
        writer.writerow(data)

print("done!")
