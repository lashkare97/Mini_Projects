import torch #The main PyTorch package that provides multi-dimensional tensors and operations on them, which are essential for deep learning
import torch.nn as nn #A subpackage of PyTorch that contains various neural network layers, loss functions, and other utilities for building and training neural networks.
import torch.nn.functional as F #A subpackage of PyTorch that provides functional interfaces for many common operations like activation functions, pooling, and more.
import torchvision.transforms as transforms #A package from torchvision that provides common image transformations, such as resizing, cropping, and normalization
from torch.utils.data import DataLoader, Dataset #A PyTorch utility to create a data loader that loads data in batches and provides iterable data for training
from torchvision.utils import save_image #A function from torchvision to save an image tensor to a file.
from PIL import Image #The Python Imaging Library, used to handle image data and perform various operations like opening, saving, and resizing images.
import os #A Python module providing a way to interact with the operating system, like navigating directories, creating folders, etc.
import matplotlib.pyplot as plt #A library for creating various plots and visualizations in Python.
import random #A Python module for generating pseudo-random numbers, useful for shuffling data and random operations.
from tqdm import tqdm  # A Python package that provides a fast and extensible progress bar for loops and iterable objects
import torch.optim as optim #A module from PyTorch that contains various optimization algorithms for training neural networks.
from torch.utils.data import random_split #A function from PyTorch to split a dataset into random subsets for training and validation/testing.
import numpy as np #An alias for the NumPy library, which provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays.
import pandas as pd #An alias for the pandas library, which provides data structures and functions for efficient data manipulation and analysis.
import time #A Python module for working with time and measuring the execution time of code.

# Set seed
random_seed = 42 # (could be anything)This line sets the value of the random seed to 42. You can choose any arbitrary integer as the seed value
torch.manual_seed(random_seed) #This line sets the random seed for the PyTorch library. It ensures that whenever you use random operations in PyTorch, like weight initialization in neural networks, the results will be consistent across different runs.
torch.cuda.manual_seed(random_seed) #If you are using a GPU (CUDA) for computation with PyTorch, this line sets the random seed for CUDA operations, making sure the GPU's behavior is deterministic
torch.backends.cudnn.deterministic = True #This line forces PyTorch's cuDNN backend (if available) to use deterministic algorithms. cuDNN is a library that accelerates deep learning computations on NVIDIA GPUs. By setting this to True, you ensure that the behavior of cuDNN is deterministic, even though it might impact performance.
torch.backends.cudnn.benchmark = False #This line disables cuDNN's auto-tuning feature. When set to True, cuDNN will choose the best algorithms for the hardware, which could introduce slight non-determinism. By setting it to False, you ensure consistent behavior.
np.random.seed(random_seed) #This line sets the random seed for NumPy, another library used for numerical computations in Python. It ensures that NumPy's random operations, if any, will also produce consistent results across different runs

# Hyperparameter und Konfiguration

model_name= "ViT-AE"
dataset_name = "brasov" # "brasov" or "vct4_punct"

# Hyperparameters
latent_dim = 128 #It represents the size of the compressed representation (encoding) of the input images.
batch_size = 32 #This hyperparameter determines the number of images processed together in a single forward and backward pass during training. A larger batch size can lead to more efficient GPU utilization but requires more memory
image_size = 128 # 128*128 pix #This hyperparameter defines the size of the input images. The images will be resized to 128x128 pixels before being processed by the model.
patch_size = 16 # 16*16 pixels #This hyperparameter defines the size of the image patches used in the Vision Transformer. Each image is divided into non-overlapping patches of size 16x16 pixels
embedding_dim = latent_dim #embeddings is to transform raw, high-dimensional data (such as text, images, or categorical variables) into a more compact and meaningful representation that captures relevant patterns and relationships in the data.
num_transformer_layers = 6 #determines the number of transformer layers used in the Vision Transformer part of the model. In this case, the model has 6 transformer layers.
num_epochs = 1 #This hyperparameter represents the number of times the entire dataset will be passed through the model during training.
learning_rate = 0.001 #This hyperparameter controls the step size at each iteration during gradient descent optimization. It determines how quickly the model learns from the training data
test_split = 0.2 #This variable determines the proportion of the dataset that will be used for testing, while the remaining portion will be used for training.
num_channels = 3 #This hyperparameter represents the number of channels in the input images. In this case, it is set to 3, which corresponds to color images with Red, Green, and Blue (RGB) channels.
#num_patches = int((image_size/patch_size)*(image_size/patch_size))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #This variable is used to set the device (CPU or GPU) where the model will be trained. If a GPU is available, the model will be trained on the GPU; otherwise, it will be trained on the CPU

# Datapath
image_folder_path = "D:\KIT\Student Research Assistant\IMI\ZeroDefects\DIG_Fingerprint_Brasov_PNG"  # Adjust the path to the folder containing the image files

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
#The transforms.Compose() function allows us to chain together multiple image transformations as a sequence. Each transformation will be applied one after the other in the specified order.
#This transformation converts the image from a PIL image format to a PyTorch tensor. The ToTensor() transformation also scales the pixel values from the range [0, 255] (integer format) to the range [0, 1] (float format). This is necessary because PyTorch models expect input data in tensor format.
dataset = ImageDataset(image_folder_path, transform=transform)
#we create the dataset object using the ImageDataset class and apply the specified transformations

# Create train-test split
num_samples = len(dataset) #This line calculates the total number of samples in the original dataset (before splitting) by calling len(dataset)
num_test_samples = int(test_split * num_samples) #This line calculates the number of samples that will be allocated to the test dataset.
num_train_samples = num_samples - num_test_samples #This line calculates the number of samples that will be allocated to the training dataset.
train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples]) #This line uses the random_split function from PyTorch to split the dataset randomly into two subsets

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#ENCODER part of the model
#The encoder processes the input images and extracts their embeddings using a combination of convolutional layers for patch embeddings and multiple TransformerEncoder layers for processing the embeddings.
#is a crucial part of the Vision Transformer model. It processes the input image and extracts meaningful embeddings that can be further used for downstream tasks like classification or image generation.

#__init__: The constructor (__init__ method) of the VisionTransformerEncoder class initializes the layers and parameters of the encoder and takes the following below arguments.
class VisionTransformerEncoder(nn.Module):
    def __init__(self, image_size, patch_size, embedding_dim, latent_dim, num_transformer_layers):
        super(VisionTransformerEncoder, self).__init__()

        num_patches = (image_size // patch_size) ** 2 #This variable represents the total number of patches that will be extracted from the input image.
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches

        # Embedding layer for patches
        #This is the embedding layer responsible for extracting embeddings from image patches. It uses a 2D convolution (nn.Conv2d) to process the input image (assumed to have 3 channels for RGB images) and extract patches of size patch_size x patch_size.
        #The output of this layer will have embedding_dim channels, effectively creating embeddings for each patch
        self.embedding = nn.Conv2d(3, embedding_dim, patch_size, stride=patch_size)

        # Transformer layers
        #The TransformerEncoder layers are responsible for processing the embeddings and capturing long-range dependencies in the image
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8) #The choice of d_model and nhead affects the model's capacity, computational efficiency, and generalization. Larger values of d_model and nhead can increase the model's capacity to capture complex patterns but may also require more computational resources.
            #nhead stands for the number of attention heads in the multi-head self-attention mechanism of the transformer layer.
            #here 8, that means the transformer layer will use 8 parallel self-attention heads to process the input embeddings.
            #d_model is the dimensionality of the input and output of the transformer layer. In other words, it represents the size of the input and output embeddings in each layer.
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
    #forward: The forward method is called when data is passed through the encoder. It performs the following steps
    #1.First, it prints the shape of the input tensor x, which represents the input image.
    #2.The input image x is then processed through the embedding layer (self.embedding), resulting in patch embeddings
    #3.The patch embeddings are then flattened along the third dimension and transposed so that the patches' sequence length becomes the second dimension. This is necessary for compatibility with the TransformerEncoder
    #4.The flattened patches are then processed through the list of TransformerEncoder layers (self.transformer_layers)
    #5.After passing through all the TransformerEncoder layers, the embeddings are aggregated by taking the mean along the second dimension, effectively averaging the embeddings of all patches. This aggregated representation is the final output of the encoder.
    #6.The shape of the final output x is printed, and it is returned as the output of the forward pass


#DECODER
class VisionTransformerDecoder(nn.Module):
    def __init__(self, latent_dim, image_size, patch_size, num_channels=3):
        super(VisionTransformerDecoder, self).__init__()
        self.latent_dim = latent_dim #The dimension of the latent space (embedding) produced by the encoder.
        self.image_size = image_size #The size of the original input image (assuming the images are square).
        self.patch_size = patch_size #The size of the patches used in the encoder
        self.num_channels = num_channels #The number of channels in the input images. The default value is 3, which corresponds to RGB images.

        num_patches = (image_size // patch_size) ** 2 #This variable represents the total number of patches that were used in the encoder
        self.embedding_dim = latent_dim * num_patches #This variable stores the dimensionality of the flattened embedding space.

        self.embedding_linear = nn.Linear(latent_dim, self.embedding_dim) #This is a linear layer (nn.Linear) that maps the latent space x (which has latent_dim dimensions) to the flattened embedding space with embedding_dim dimensions.

        self.upsample = nn.Upsample(scale_factor=self.patch_size, mode='bilinear', align_corners=False) #This is an upsampling layer (nn.Upsample) that upsamples the embeddings back to the original image size. It uses bilinear interpolation for upsampling
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
    #forward: The forward method is called when data is passed through the decoder. It performs the following steps
    #1.It prints the patch size and the shape of the input tensor x, which represents the latent space (embedding) produced by the encoder
    #2.The latent space x is passed through the self.embedding_linear layer to map it to the flattened embedding space with embedding_dim dimensions
    #3.The flattened embedding space is then reshaped into a 4-dimensional tensor with shape (batch_size, latent_dim, num_patches, 1). The third dimension represents the number of patches, and the fourth dimension is added to be compatible with the upsampling operation
    #4.The tensor is upsampled back to the original image size using the self.upsample layer. The upsampling operation restores the spatial structure of the patches, allowing them to cover the entire image.
    #5.The upsampled tensor is then passed through the self.patch_reconstruction module, which performs convolution operations to reconstruct the original image from the upsampled embeddings
    #6.Finally, the output tensor x is reshaped to have shape (batch_size, num_channels, image_size, image_size), representing the reconstructed image with num_channels channels


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
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #The Adam optimizer is chosen for training the model. It is a popular optimization algorithm that adapts the learning rates of individual model parameters
criterion = nn.MSELoss() #MSE loss measures the average squared difference between the predicted values and the ground truth (target) values. In the context of an autoencoder, it measures the difference between the original input images and their reconstructions

# Train the model
train_losses = [] #The train_losses list will be used to store the training loss after each epoch
start_time = time.time()
for epoch in range(num_epochs): #Inside each epoch, the epoch_loss variable is initialized to zero, and a tqdm progress bar is set up to visualize the progress of the current epoch
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    # Preprocessing Images and Forward Pass:
    for images in progress_bar:
        # Use only the top-left corner of the image
        images = images[:, :, :image_size, :image_size] #The input images are cropped to keep only the top-left corner of each image, which has a size of image_size x image_size.
        # This cropping is done to ensure that all images have the same size, as required by the model

        # Forward pass
        reconstructions = model(images) #The cropped images are then passed through the autoencoder model (model) to obtain the reconstructions

        print("Reconstructions shape:", reconstructions.shape)
        print("Input images shape:", images.shape)

        # Compute Loss and Update Model Parameters:
        loss = criterion(reconstructions, images)
#The reconstruction loss is computed between the reconstructions and the original images using the Mean Squared Error (MSE) loss function (criterion)
        # Backward pass and optimization
        optimizer.zero_grad() #The gradients are cleared using optimizer.zero_grad() to avoid accumulating gradients from previous iterations
        loss.backward() #Backpropagation is performed to compute the gradients of the loss with respect to the model parameters using loss.backward()
        optimizer.step() #The optimizer updates the model parameters using the computed gradients and the specified learning rate (optimizer.step())

        # Update Progress Bar and Compute Epoch Loss
        epoch_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()}) #The progress bar is updated with the latest loss information for the current batch.
    # Calculate and Store Epoch Loss:
    epoch_loss /= len(train_dataloader) #After processing all batches in the epoch, the epoch_loss is divided by the number of batches (len(train_dataloader)) to obtain the average loss for the epoch
    train_losses.append(epoch_loss) # The average loss for the epoch is stored in the train_losses list.
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
patch_grid = np.transpose(patch_grid, (0, 2, 1, 3, 4)) #This line transposes the dimensions of the patch grid to match the arrangement expected by matplotlib for displaying images
patch_grid = np.reshape(patch_grid, (-1, patch_size, patch_size, 3)) #This line further reshapes the patch grid to create a flat list of individual patches. Each patch has a size of patch_size x patch_size with 3 channels (RGB)

fig, axs = plt.subplots(image_size // patch_size, image_size // patch_size, figsize=(8, 8))
for i in range(image_size // patch_size):
    for j in range(image_size // patch_size):
        axs[i, j].imshow(patch_grid[i * (image_size // patch_size) + j]) #The nested loop then iterates over each subplot and displays the corresponding image patch using
        axs[i, j].axis('off') #This line removes the axis labels from the subplots to enhance the visual appearance

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