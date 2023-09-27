import pydicom 
import os.path
import pandas as pd
import numpy as np
import csv

image_folder = '/home/aalmansour/source/lidc_slices/crops'
csv_file_path = '/home/aalmansour/source/lidc_slices/lidc_files/image_label_mapping.csv'
data1 = pd.read_csv("/home/aalmansour/source/lidc_slices/lidc_files/LIDC_20130817_AllFeatures2D_AllSlices.csv")
data2 = pd.read_csv("/home/aalmansour/source/lidc_slices/lidc_files/Agreement_Binary_Rating.csv") 

# Define the target size (71x71)
target_size = (71, 71)

# Select rows from df1 with IDs present in df2
selected_rows = data1[data1['noduleID'].isin(data2['noduleID'])]
import pdb; pdb.set_trace()
# Create a dictionary to map IDs to classifications and ratings
id_mapping = data2.set_index('noduleID').to_dict()

# Map the values to the first DataFrame
selected_rows['Agreement'] = selected_rows['noduleID'].map(id_mapping['Agreement'])
selected_rows['Agreement_name'] = selected_rows['noduleID'].map(id_mapping['Agreement_name'])
selected_rows['Malignancy'] = selected_rows['noduleID'].map(id_mapping['Malignancy'])
#selected_rows['RadiologistID'] = selected_rows['noduleID'].map(id_mapping['RadiologistID'])

# Create a dictionary to store the mapping between image IDs and labels
image_label_mapping = {}

# Iterate through the DataFrame rows
for index, row in selected_rows.iterrows():
    instance_id = row['InstanceID']
    nodule_id = row['noduleID']
    radiologist_id = row['RadiologistID']
    label = row['Malignancy']
    agreement = row['Agreement']
    
    # Construct the full path to the DICOM image file
    image_file_path = os.path.join(image_folder, f'{instance_id}.dcm')
    print(image_file_path)
    #import pdb; pdb.set_trace()
    # Check if the DICOM image file exists
    if os.path.exists(image_file_path):
        # Store the mapping between image ID and label
        image_label_mapping[instance_id] = {
            'nodule_id': nodule_id,
            'radiologist_id': radiologist_id,
            'label': label,
            'agreement': agreement,
            'dcm_data': image_file_path  # You can store the DICOM data for further processing if required
        }
    else:
        print(f"Image file '{instance_id}.dcm' not found.") 

# Extract the data from the dictionary into a list of dictionaries (each dictionary represents a row)
data_to_save = [
    {'instance_id': instance_id, 'nodule_id': data['nodule_id'], 'image': data['dcm_data'], 'label': data['label'], 'agreement': data['agreement'], 'radiologist_id': data['radiologist_id']} 
    for instance_id, data in image_label_mapping.items()
]

# Define the CSV header (field names)
field_names = ['nodule_id', 'instance_id', 'image', 'label','agreement', 'radiologist_id']

# Write the data to the CSV file
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=field_names)
    
    # Write the header
    writer.writeheader()
    
    # Write the data rows
    writer.writerows(data_to_save)

print(f'Data saved to {csv_file_path}')
#data3 = pd.read_csv(csv_file_path) 
#print(len(data3))