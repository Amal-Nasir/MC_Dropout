import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def getNormed(this_array, this_min = 0, this_max = 255, set_to_int = True):
    new_var = this_array.copy()
    rat = (this_max - this_min)/(new_var.max() - new_var.min())
    new_var = new_var * rat
    new_var -= new_var.min()
    new_var += this_min
    if set_to_int:
        return new_var.astype('uint8')
    return new_var

def pad_img(dcm):
    # Get the current image size
    image_height, image_width = dcm.Rows, dcm.Columns

    # Calculate padding dimensions
    padding_height = max(71 - image_height, 0)
    padding_width = max(71 - image_width, 0)

    # Calculate the top, bottom, left, and right padding
    top_padding = padding_height // 2
    bottom_padding = padding_height - top_padding
    left_padding = padding_width // 2
    right_padding = padding_width - left_padding

    # Create a black pixel (pixel value of 0)
    black_pixel = 0

    # Pad the DICOM pixel data with black pixels
    padded_data = np.pad(
        getNormed(dcm.pixel_array),
        ((top_padding, bottom_padding), (left_padding, right_padding)),
        constant_values=black_pixel
    )

    return padded_data

def get_data(img):
    dcm = []
    # Iterate through the DataFrame rows
    #for img in data:
    # Construct the full path to the DICOM image file
    dcm1 = pydicom.dcmread(img)
    #dcm = dcm1.pixel_array
    #dcm = getNormed(dcm1)
    dcm = pad_img(dcm1)
    
    return dcm
original = pd.read_csv('/home/aalmansour/source/lidc_slices/lidc_files/image_label_mapping.csv')
test_un = pd.read_csv('/home/aalmansour/source/lidc_slices/test_uncertainty_stats.csv')
test_un.rename(columns={'Image_ID': 'instance_id'}, inplace=True)
merged_test_un = pd.merge(original, test_un, on='instance_id', how='inner')

# Get a list of DICOM file paths in the directory
dicom_files = [get_data(image) for image in merged_test_un['image']]

# Create subplots to display the images
num_images = len(dicom_files)
num_rows = 6  # Number of rows in the grid
num_cols = num_images // num_rows + (num_images % num_rows > 0)  # Adjust to fit all images

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i, ax in enumerate(axes.flat):
    if i < num_images:
        pixel_data = dicom_files[i]

        ax.imshow(pixel_data, cmap='gray')
        ax.set_title(f"Image {i + 1}")
        ax.axis("off")  # Turn off axis labels

plt.tight_layout()
plt.show()
