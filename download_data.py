import os
import pickle
import torch
#import torchvision
import zipfile
from google_drive_downloader import GoogleDriveDownloader as gdd 
import gdown

path = './/data//'  # give the path of the directory in which we would like to extract the data

isExist = os.path.exists(path)
if isExist:
    print("data already present")
   
else:
    print("creating data folder")
    # Create a new directory because it does not exist
    os.makedirs(path)
   
data_path = './data/'

isExist = os.path.exists(data_path)

gdd.download_file_from_google_drive(file_id='185YOfFcoF_r9QbV7YBbWgSgx9ik9JoAh',  #follow step 2 to find the file_id and dest_path
                                    dest_path='./data/crops.zip',
                                    unzip=True)                                   #unzip=true means we can see the progress of the data downloaded from the google drive in the code editor

## Fetch data from Google Drive
# Root directory for the dataset
data_root = './data/' 
# Path to pkl with the images
dataset_folder = f'{data_root}/' #this will create a new folder name "download" and the dataset of the folder will be extracted in the download folder

#import pdb; pdb.set_trace()
# URL for the pkl
url = 'https://drive.google.com/file/d/185YOfFcoF_r9QbV7YBbWgSgx9ik9JoAh/view?usp=sharing' #copy this URL and replace the file ID with the other file ID that would be needed to extract the data

# Path to download the pkl
download_path = f'{data_root}/crops.zip'

# Create required directories
if not os.path.exists(data_root):
    os.makedirs(data_root)
    os.makedirs(dataset_folder)

# Download the pkl from google drive
gdown.download(url, download_path, quiet=False)

# Unzip the downloaded file
with zipfile.ZipFile(download_path, 'r') as ziphandler:  #this unzip the folder once downlaoded 
    #if ziphandler.endswith(".dcm"):
    ziphandler.extractall(dataset_folder)

# Open the zip file
#with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Check if the file has a .dcm extension
    #if filename.endswith(".dcm"):
    # Extract all contents to the destination folder
    #zip_ref.extractall(path=destination_folder)

#print(f'Unzipped {zip_file_path} to {destination_folder}')

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch CUDA Version is", torch.version.cuda)
print("Whether CUDA is supported by our system:", torch.cuda.is_available())

# Load the data back into memory
file_pathimages = './data/'
with open(file_pathimages, 'rb') as f:
    images = pickle.load(f)
