import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset 
from torchvision.utils import save_image 
from PIL import Image 
import os 
import matplotlib.pyplot as plt 
import random 
from tqdm import tqdm  
import torch.optim as optim 
from torch.utils.data import random_split 
import numpy as np 
import pandas as pd 
import time 

# Set seed
random_seed = 42 
torch.manual_seed(random_seed) 
torch.cuda.manual_seed(random_seed) 
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False 
np.random.seed(random_seed) 

# Hyperparameter und Konfiguration

model_name= "ViT-AE"
dataset_name = "brasov" # "brasov" or "vct4_punct"

# Hyperparameters
latent_dim = xx 
batch_size = xx 
image_size = xx 
patch_size = xx 
embedding_dim = latent_dim 
num_transformer_layers = xx 
num_epochs = xx 
learning_rate = xx 
test_split = xx 
num_channels = xx 
num_patches = int((image_size/patch_size)*(image_size/patch_size))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Datapath
image_folder_path = "path"  # Adjust the path to the folder containing the image files

#CLASS TEMPORARY
# Define the ImageDataset class
class ImageDataset(Dataset):
    def __init__(self, image_folder_path, transform=None):
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.image_files = os.listdir(image_folder_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder_path, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        image = image.crop((0, 0, 128, 128))

        if self.transform is not None:
            image = self.transform(image)

        return image


### Create output_path ###
#code defines a custom dataset class called ImageDataset without a specific class structure.
def create_save_path(model_name, dataset_name, latent_dim, epochs):
    save_path = os.path.join("models", model_name, dataset_name, f"BNS_{latent_dim}", f"epochs_{epochs}")
    os.makedirs(save_path, exist_ok=True)
    return save_path

output_folder_path = create_save_path(model_name, dataset_name, latent_dim, num_epochs)
print(output_folder_path)

# Load and preprocess the dataset
transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

dataset = ImageDataset(image_folder_path, transform=transform)
#we create the dataset object using the ImageDataset class and apply the specified transformations

# Create train-test split
num_samples = len(dataset) 
num_test_samples = int(test_split * num_samples) 
num_train_samples = num_samples - num_test_samples 
train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples]) 

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#ENCODER part of the model

class VisionTransformerEncoder(nn.Module):
    def __init__(self, image_size, patch_size, embedding_dim, latent_dim, num_transformer_layers):
        super(VisionTransformerEncoder, self).__init__()

        num_patches = (image_size // patch_size) ** 2 
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches

        # Embedding layer for patches
        
        self.embedding = nn.Conv2d(3, embedding_dim, patch_size, stride=patch_size)

        # Transformer layers
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8) 
            
            for _ in range(num_transformer_layers)
        ])

    def forward(self, x):
        # Embedding layer for patches
        print(x.shape)
        x = self.embedding(x)
        print(x.shape)
        # Flatten the patches
        x = x.flatten(2).transpose(1, 2)
        print('flatten', x.shape)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # The latent vector is the mean of the patch embeddings
        # You can also use max, min, or other aggregation functions
        x = x.mean(dim=1)

        print(x.shape)
        return x


#DECODER
class VisionTransformerDecoder(nn.Module):
    def __init__(self, latent_dim, image_size, patch_size, num_channels=3):
        super(VisionTransformerDecoder, self).__init__()
        self.latent_dim = latent_dim 
        self.image_size = image_size 
        self.patch_size = patch_size 
        self.num_channels = num_channels 

        num_patches = (image_size // patch_size) ** 2 
        self.embedding_dim = latent_dim * num_patches 

        self.embedding_linear = nn.Linear(latent_dim, self.embedding_dim)

        self.upsample = nn.Upsample(scale_factor=self.patch_size, mode='bilinear', align_corners=False) 
        # This is a series of convolutional layers that perform the reconstruction of the original image from the upsampled embeddings.
        # It consists of a 3x3 convolution layer, followed by a GELU activation function,
        self.patch_reconstruction = nn.Sequential(nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(self.image_size, num_channels, kernel_size=1),)

    def forward(self, image_size, patch_size, x):
        batch_size, latent_dim = x.shape
        num_patches = (self.image_size // self.patch_size) ** 2
        print(self.patch_size)
        print('in', x.shape)
        x = self.embedding_linear(x)
        print('emb', x.shape)
        x = x.view(batch_size, latent_dim, num_patches, 1)
        print('view', x.shape)
        x = self.upsample(x)
        print('up', x.shape)

        x = self.patch_reconstruction(x)
        x = x.view(batch_size, self.num_channels, self.image_size, self.image_size)

        return x


# Define the Vision Transformer Autoencoder
class VisionTransformerAutoencoder(nn.Module):
    def __init__(self, image_size, patch_size, embedding_dim, latent_dim, num_transformer_layers):
        super(VisionTransformerAutoencoder, self).__init__()

        num_patches = (image_size // patch_size) ** 2
        self.encoder = VisionTransformerEncoder(image_size, patch_size, embedding_dim, latent_dim, num_transformer_layers)
        self.decoder = VisionTransformerDecoder(latent_dim, image_size, patch_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x, image_size, patch_size)
        return x

# Create the model and optimizer
model = VisionTransformerAutoencoder(image_size, patch_size, embedding_dim, latent_dim, num_transformer_layers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
criterion = nn.MSELoss() 

# Train the model
train_losses = [] 
start_time = time.time()
for epoch in range(num_epochs): 
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    # Preprocessing Images and Forward Pass:
    for images in progress_bar:
        # Use only the top-left corner of the image
        images = images[:, :, :image_size, :image_size] 
        # This cropping is done to ensure that all images have the same size, as required by the model

        # Forward pass
        reconstructions = model(images) 

        print("Reconstructions shape:", reconstructions.shape)
        print("Input images shape:", images.shape)

        # Compute Loss and Update Model Parameters:
        loss = criterion(reconstructions, images)
#The reconstruction loss is computed between the reconstructions and the original images using the Mean Squared Error (MSE) loss function (criterion)
        # Backward pass and optimization
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        # Update Progress Bar and Compute Epoch Loss
        epoch_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()}) 
    # Calculate and Store Epoch Loss:
    epoch_loss /= len(train_dataloader) 
    train_losses.append(epoch_loss) 
    progress_bar.close()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Training time: {elapsed_time:.2f} seconds')

# Save the trained model
# The torch.save() function saves the state dictionary as a serialized file in the specified location. This allows you to later load the
# model and its learned parameters to use it for inference or further training without having to retrain the model from scratch.
torch.save(model.state_dict(), os.path.join(output_folder_path,'vision_transformer_autoencoder.pth'))
# model.state_dict(): This method returns a dictionary containing the state of the model's parameters. It contains all the learnable parameters of the model, such as weights and biases of the layers
# os.path.join(output_folder_path, 'vision_transformer_autoencoder.pth'): This creates the full file path where the model's state dictionary will be saved. The output_folder_path variable represents the directory path where the model will be saved.
# The filename is set to 'vision_transformer_autoencoder.pth'


# Save Train History
# Convert train_losses and train_times to DataFrame
train_data = {'Epoch': range(1, num_epochs + 1),
              'Train Loss': train_losses,
              }
train_df = pd.DataFrame(train_data)

# Save the DataFrame to CSV
train_df.to_csv(os.path.join(output_folder_path,'training_history.csv'), index=False)

# Plot the training loss history
plt.plot(range(1, num_epochs+1), train_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig(os.path.join(output_folder_path,'loss_plot.png'))
plt.show()

# Plot example image patches
example_image = next(iter(train_dataloader))
example_image = example_image[0].numpy().transpose((1, 2, 0))
#This line reshapes the image data to create a 2D grid of patches
patch_grid = np.reshape(example_image, (image_size // patch_size, patch_size, image_size // patch_size, patch_size, 3))
patch_grid = np.transpose(patch_grid, (0, 2, 1, 3, 4)) 
patch_grid = np.reshape(patch_grid, (-1, patch_size, patch_size, 3)) 

fig, axs = plt.subplots(image_size // patch_size, image_size // patch_size, figsize=(8, 8))
for i in range(image_size // patch_size):
    for j in range(image_size // patch_size):
        axs[i, j].imshow(patch_grid[i * (image_size // patch_size) + j]) 
        axs[i, j].axis('off') 

plt.suptitle('Example Image Patches')
plt.savefig(os.path.join(output_folder_path,'patch_sample.png'))
plt.show()

# Load the trained model
model = VisionTransformerAutoencoder(image_size, patch_size, embedding_dim, latent_dim, num_transformer_layers)
model.load_state_dict(torch.load(os.path.join(output_folder_path, 'vision_transformer_autoencoder.pth')))
model.eval()

# Select 5 random images from the test dataset
selected_indices = np.random.choice(len(test_dataset), size=5, replace=False)

# Create lists to store original images and their reconstructions
original_images = []
reconstructions = []

# Generate reconstructions for selected images
with torch.no_grad():
    for i in selected_indices:
        image = test_dataset[i]
        image = image.unsqueeze(0)  # Add batch dimension
        image = image[:, :, :image_size, :image_size]

        # Encode the image
        latent_vector = model.encoder(image)

        # Reconstruct the image
        reconstruction = model.decoder(latent_vector)

        original_images.append(image.squeeze().numpy().transpose((1, 2, 0)))
        reconstructions.append(reconstruction.squeeze().numpy().transpose((1, 2, 0)))

# Plot the original images and their reconstructions
fig, axes = plt.subplots(5, 2, figsize=(8, 2 * 5))
fig.suptitle('Original Images vs. Reconstructions', fontsize=16)

for i in range(5):
    axes[i, 0].imshow(original_images[i])
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(reconstructions[i])
    axes[i, 1].set_title('Reconstruction')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
