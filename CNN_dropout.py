import pandas as pd
import numpy as np
import csv
import pydicom 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
#DEVICE = torch.device('cpu') 
# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels, image_ids):
        self.data = data
        self.labels = labels
        self.image_id = image_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.image_id[idx]
    
data = pd.read_csv('/home/aalmansour/source/lidc_slices/lidc_files/image_label_mapping.csv') 
#By performing this remapping, your labels will be compatible with nn.CrossEntropyLoss()
data['label'] = data['label'] - 1

# Define the split ratios (e.g., 70% train, 15% validation, 15% test)
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# First, split the data into training and temporary data (to be further split into validation and test)
train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)

# Calculate the remaining ratios for validation and test
remaining_ratio = validation_ratio / (validation_ratio + test_ratio)
validation_data, test_data = train_test_split(temp_data, test_size=remaining_ratio, random_state=42)

print('train data len:', len(train_data))
print('val data len:', len(validation_data))
print('test data len:', len(test_data))

# Create a set of unique image IDs to make sure no overlap in train, validation, and test
# unique_image_ids = set(data['nodule_id'])
# print(len(unique_image_ids))

# Convert the set of unique image IDs to a list and shuffle it
# shuffled_image_ids = list(data['nodule_id'])
# random.shuffle(shuffled_image_ids)

# # Define the split ratios (e.g., 70% train, 15% validation, 15% test)
# train_ratio = 0.7
# validation_ratio = 0.2
# test_ratio = 0.1

# # Calculate the sizes of the splits based on the ratios
# total_samples = len(shuffled_image_ids)
# train_size = int(train_ratio * total_samples)
# validation_size = int(validation_ratio * total_samples)
# test_size = total_samples - train_size - validation_size

# # Split the image IDs into train, validation, and test sets
# train_image_ids = shuffled_image_ids[:train_size]
# validation_image_ids = shuffled_image_ids[train_size:train_size + validation_size]
# test_image_ids = shuffled_image_ids[train_size + validation_size:]

# # Create dictionaries for the train, validation, and test datasets
# train_data = pd.DataFrame() 
# validation_data = pd.DataFrame()
# test_data = pd.DataFrame()

# # Iterate through the dataset and add instances to respective sets
# # Each set includes all associated instance IDs for their respective image IDs

# for index, row in data.iterrows():
#     item = row['nodule_id']
#     if item in train_image_ids:
#         train_data = data[data['nodule_id'].isin(train_image_ids)]
#     elif item in validation_image_ids:
#         validation_data = data[data['nodule_id'].isin(validation_image_ids)]
#     elif item in test_image_ids:
#         test_data = data[data['nodule_id'].isin(test_image_ids)]

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

# Create an empty list to store the preprocessed images
preprocessed_images = [get_data(image) for image in train_data['image']]
different_shape_indices = [i for i, arr in enumerate(preprocessed_images) if not np.all(np.array(arr.shape) == np.array(preprocessed_images[0].shape))]
# Create a new list with only elements that have the same shape
filtered_image_data_list = [arr for arr in preprocessed_images if np.all(np.array(arr.shape) == np.array(preprocessed_images[0].shape))]
# Convert the list of arrays into a single NumPy array
input_data = np.stack(filtered_image_data_list)

preprocessed_images1 = [get_data(image) for image in validation_data['image']]
filtered_val_data_list = [arr for arr in preprocessed_images1 if np.all(np.array(arr.shape) == np.array(preprocessed_images1[0].shape))]
val_data = np.stack(filtered_val_data_list)

preprocessed_images2 = [get_data(image) for image in test_data['image']]
filtered_test_data_list = [arr for arr in preprocessed_images2 if np.all(np.array(arr.shape) == np.array(preprocessed_images2[0].shape))]
te_data = np.stack(filtered_test_data_list)

print('input_data len:', len(input_data))
print('val_data len:', len(val_data))
print('te_data len:', len(te_data))
print("Images are ready!")

train_image_ids = train_data['instance_id'].values
train_image_ids = np.delete(train_image_ids,different_shape_indices)
val_image_ids = validation_data['instance_id'].values
test_image_ids = test_data['instance_id'].values
print("Image ids are ready!")

train_labels = train_data['label'].values
train_labels = np.delete(train_labels,different_shape_indices)
val_labels = validation_data['label'].values
test_labels = test_data['label'].values
print("Labels are ready!")

batch_size = 32
num_images = len(input_data)
num_val_image = len(val_data)
num_test_image = len(te_data)

# Convert to 2D PyTorch tensors
input_data = torch.from_numpy(input_data).float()
input_data = input_data.unsqueeze(1).to(DEVICE)
tr_ids = torch.tensor(train_image_ids, dtype=torch.long)
#tr_labels = torch.tensor(train_labels, dtype=torch.long)

val_dataset = torch.from_numpy(val_data).float()
val_dataset = val_dataset.unsqueeze(1).to(DEVICE)
val_ids = torch.tensor(val_image_ids, dtype=torch.long)
#va_labels = torch.tensor(val_labels, dtype=torch.long)

test_dataset = torch.from_numpy(te_data).float()
test_dataset = test_dataset.unsqueeze(1).to(DEVICE)
test_ids = torch.tensor(test_image_ids, dtype=torch.long)
#te_labels = torch.tensor(test_labels, dtype=torch.long)
print("Done from tranformation to tensors.")

# Create data loaders
train_loader = DataLoader(CustomDataset(input_data, train_labels, tr_ids), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CustomDataset(val_dataset, val_labels, val_ids), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(CustomDataset(test_dataset, test_labels, test_ids), batch_size=batch_size, shuffle=True)
print("Done from data loaders.")

for images, labels, img_ids in train_loader:
    print('Image batch dimentions:', images.shape)
    print('Image label dimentions:', labels.shape)
    print('Image ids dimentions:', img_ids.shape)
    print('Class labels of 10 examples:', labels[:10])
    break

class CNNModel(nn.Module):
    def __init__(self, num_classes, dropout_prob):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(322624, 128) # 64 channels, 18x18 feature map size = 20736
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
            x = torch.relu(self.conv1(x))  # Apply convolution and ReLU activation
            x = torch.relu(self.conv2(x))  # Apply another convolution and ReLU activation
            x = x.view(x.size(0), -1)  # Flatten the feature map
            x = torch.relu(self.fc1(x))  # Apply a fully connected layer and ReLU activation
            x = self.dropout(x)  # Apply dropout during training and inferencet
            x = self.fc2(x)  # Apply the final fully connected layer
            
            return x

model = CNNModel(num_classes=5, dropout_prob=0.2)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 50
epoch_train_accuracies = [] # Create an empty list to store accuracy values
epoch_val_accuracies = []
epoch_test_accuracies = []

# Create empty lists to store predicted labels and image IDs
train_predicted_labels = []
train_image_ids = []
val_predicted_labels = []
val_image_ids = []
test_predicted_labels = []
test_image_ids = []

for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0
    for images, labels, image_ids in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        image_ids = image_ids
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        #total_loss += loss.item()
        optimizer.step()
        # Get predicted train labels
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        
        # Append predicted labels and image IDs to the lists
        train_predicted_labels.extend(predicted.cpu().numpy())  # Convert to CPU and numpy for compatibility
        train_image_ids.extend(image_ids.cpu().numpy())
    train_accuracy = total_correct / total_samples
    epoch_train_accuracies.append(train_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Training Accuracy: {train_accuracy*100:.2f}%")

    # Validation loop
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for images, labels, image_ids in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            image_ids = image_ids
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # Append predicted labels and image IDs to the lists
            val_predicted_labels.extend(predicted.cpu().numpy())  # Convert to CPU and numpy for compatibility
            val_image_ids.extend(image_ids.cpu().numpy())

        val_accuracy = total_correct / total_samples
        epoch_val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Validation Accuracy: {val_accuracy*100:.2f}%")

# Save training predictions and image IDs to a CSV file
with open('train_predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image ID', 'Predicted Label'])
    writer.writerows(zip(train_image_ids, train_predicted_labels))

# Save validation predictions and image IDs to a CSV file
with open('val_predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image ID', 'Predicted Label'])
    writer.writerows(zip(val_image_ids, val_predicted_labels))

# Save the list of accuracies to a file (e.g., CSV or text file)
with open('train_accuracies.csv', 'w') as file:
    # Write the header line
    file.write("Epoch,Accuracy\n")
    # Write the data lines
    for epoch, accuracy in enumerate(epoch_train_accuracies):
        file.write(f"{epoch+1},{accuracy}\n")

with open('val_accuracies.csv', 'w') as file:
        # Write the header line
    file.write("Epoch,Accuracy\n")
    # Write the data lines
    for epoch, accuracy in enumerate(epoch_val_accuracies):
        file.write(f"{epoch+1},{accuracy}\n")

# Test the model
model = model.to(DEVICE)
model.eval()
with torch.no_grad():
    for images, labels, image_ids in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        image_ids = image_ids
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        # Append predicted labels and image IDs to the lists
        test_predicted_labels.extend(predicted.cpu().numpy())  # Convert to CPU and numpy for compatibility
        test_image_ids.extend(image_ids.cpu().numpy())

    accuracy = total_correct / total_samples
    epoch_test_accuracies.append(accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Testing Accuracy: {accuracy*100:.2f}%")

# Save testing predictions and image IDs to a CSV file
with open('test_predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image ID', 'Predicted Label'])
    writer.writerows(zip(test_image_ids, test_predicted_labels))

with open('test_accuracies.csv', 'w') as file:
    file.write("Epoch,Accuracy\n")
    for epoch, accuracy in enumerate(epoch_test_accuracies):
         file.write(f"{epoch+1},{accuracy}\n")

# Define the number of Monte Carlo samples
num_samples = 100
# def monte_carlo_dropout_inference(model, images, num_samples):
#     model.eval()
#     predictions = []
#     for _ in range(num_samples):
#         with torch.no_grad():
#             output = model(images)
#         predictions.append(output)
#     print("MC predictions' length:", len(predictions))
#     return torch.cat(predictions, dim=1)

# # Perform Monte Carlo Dropout inference and calculate uncertainty
# def MC_uncertainty(data_loader, model):
#     for images, labels, image_ids in data_loader:
#         predictions = monte_carlo_dropout_inference(model, images, num_samples).to(DEVICE)
        
#         # Iterate through each sample in the batch
#         for i in range(images.size(0)):
#             true_label = labels[i].item()
#             mean_prediction = torch.mean(predictions[i, :]).item()
#             variance_prediction = torch.var(predictions[i, :]).item()
#             print(f"Sample {i + 1} | True Label: {true_label} | Mean Prediction: {mean_prediction} | Prediction Variance: {variance_prediction}")
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# Perform Monte Carlo Dropout inference and calculate uncertainty
def MC_uncertainty(data_loader, model, num_samples, output_csv):
    uncertainty_data = []
    model.eval()
    for images, labels, image_ids in data_loader:
        all_sample_predictions = []
        all_sample_predictions_std = []
        # Perform Monte Carlo dropout inference for each sample
        for i in range(images.size(0)):
            sample_predictions = []
            std_diff_true_pred = []
            for _ in range(num_samples):
                with torch.no_grad():
                    enable_dropout(model)
                    true_label = image_ids[i].cpu().numpy()
                    output = model(images[i:i+1].to(DEVICE)) # Generate Monte Carlo dropout predictions
                    predicted = torch.argmax(output, dim=1).item()
                    true_label = true_label + 1
                    predicted = predicted + 1
                    sample_predictions.append(predicted)
                    std_diff_true_pred.append(true_label-predicted)
            
            all_sample_predictions.append(sample_predictions)
            all_sample_predictions_std.append(np.std(std_diff_true_pred))
        
        # Calculate the mean prediction and variance for each sample
        for i in range(images.size(0)):
            Image_ID = image_ids[i].cpu().numpy()
            true_label = labels[i].item()
            true_label = true_label + 1
            sample_means = sum(all_sample_predictions[i])/len(all_sample_predictions[i])
            sample_std = np.std(all_sample_predictions[i])
            
            uncertainty_data.append([Image_ID, true_label, sample_means, sample_std, true_label-sample_means, all_sample_predictions_std[i], all_sample_predictions[i]])

    # Save the uncertainty data to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image_ID','True_Label', 'Mean_Prediction', 'Prediction_STD', 'Difference_T-P', 'all_sample_predictions_std', 'all_sample_predictions100'])
        writer.writerows(uncertainty_data)

MC_uncertainty(train_loader, model, num_samples, 'train_uncertainty_stats.csv')
print("MC_uncertainty for training is done!")
MC_uncertainty(val_loader, model, num_samples, 'val_uncertainty_stats.csv')
print("MC_uncertainty for validation is done!")
MC_uncertainty(test_loader, model, num_samples, 'test_uncertainty_stats.csv')
print("MC_uncertainty for testing is done!")
