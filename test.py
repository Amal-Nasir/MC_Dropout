import pandas as pd
import numpy as np

data = pd.read_csv('/home/aalmansour/source/lidc_slices/lidc_files/image_label_mapping.csv') 
print(type(data['label'].values))
print(data['label'])
