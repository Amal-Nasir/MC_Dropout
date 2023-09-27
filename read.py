import os

# Replace 'folder_path' with the path to the folder you want to count files in
folder_path = '/home/aalmansour/source/lidc_slices/crops'

# Use os.listdir to get a list of all files in the folder
files_in_folder = os.listdir(folder_path)

# Use len() to count the number of files in the list
file_count = len(files_in_folder)

print(f'The folder contains {file_count} files.')