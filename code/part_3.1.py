import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import random
import struct
from array import array

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


batch_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 1000
num_ar_epochs = 5000


# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_psnr(original, reconstructed):

    mse = torch.mean(
        (original - reconstructed) ** 2,
        dim=(1, 2, 3)
    )

    psnr = 10 * torch.log10(1.0 / mse)

    return psnr

# MNIST dataset is of size 28x28.

# Loading the dataset and preprocessing
# getting a 100 subset
full_train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True,
)

dataset_100 = Subset(full_train_dataset, list(range(100)))

train_loader = DataLoader(dataset_100, batch_size=64, shuffle=True)
test_loader = DataLoader(
    dataset_100, batch_size=64, shuffle=False
)  # Evaluates memorization on the same 100 images

# Convolutional autoencoder, made to fit latent Nx3x4x4
class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        # Encoder: 1x28x28 -> 128x12x12 -> 3x4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=0),  # -> 128x24x24
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 128x12x12

            nn.Conv2d(128,128, 3, padding=1), # -> 128x12x12 extra refinement step
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=0),  # -> 3x8x8
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 3x4x4
            nn.BatchNorm2d(3),
        )

        # Decoder:
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 128, 8, stride=3, padding=2),   # -> 128x13x13
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # extra refinement steps
            nn.Conv2d(128, 128, 3, padding=1),                   
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 32, 4, stride=2),             # -> 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),


            nn.Conv2d(32, 1, 3, padding=1),                       # smooth out artifacts
            nn.Sigmoid(),
        )

    def forward(self, x):

        lats = self.encoder(x)

        #gaussian noise injection for better robustness (BEFORE tanh)
        if self.training:
            lats = lats + torch.randn_like(lats) * 0.5

        #Tanh for stable STE gradients of [-1, 1]
        lats = torch.tanh(lats)
        # binarize
        binary = torch.where(lats >= 0, 1.0, -1.0)

        #STE trick: forward is exactly ±1, backward flows through tanh
        lats_ste = lats + (binary - lats).detach()


        recs = self.decoder(lats_ste)

        return recs
model = AE().to(device)

# Setting the loss function
loss_function = nn.MSELoss()


# Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


epochs = num_epochs

outputs = []
losses = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


train_ae = False

if train_ae:
  for epoch in range(epochs):
    for images, _ in train_loader:
      images = images.to(device)
      reconstructed = model(images)
      loss = loss_function(reconstructed, images)
      #optimizer links with the model.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses.append(loss.item())
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

  # Save the weights after training
  torch.save(model.state_dict(), "autoencoder_mnist.pth")
  print("Autoencoder saved to disk.")
else:
  # Load pre-trained weights instantly
  model = AE().to(device)
  model.load_state_dict(
      torch.load("autoencoder_mnist.pth", map_location=device)
  )
  print("Autoencoder loaded from disk.")

model.eval()


plt.style.use("fivethirtyeight")

plt.figure(figsize=(8, 5))

plt.plot(losses, label="Loss")

plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.legend()

plt.show()


# Evaluate PSNR only after the final training epoch
model.eval()

total_psnr = 0.0
total_images = 0

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)

        reconstructed = model(images)

        psnr_values = calculate_psnr(images, reconstructed)

        total_psnr += psnr_values.sum().item()
        total_images += images.size(0)

# Print one binary latent
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)

    lats = torch.tanh(model.encoder(images))
    binary = torch.where(lats >= 0, 1.0, -1.0)

    print("Latent shape:", binary.shape)
    print("First image binary latent:")
    print(binary[0])


average_psnr = total_psnr / total_images

print(f"Final Average Test PSNR: {average_psnr:.2f} dB")












'''
        Autoregressive Model
'''



def extract_decimal_tokens(ae_model, dataloader, num_images=100):
    ae_model.eval()
    tokens_list = []
    
    with torch.no_grad():
        for x, _ in dataloader:
            if len(tokens_list) * x.size(0) >= num_images:
                break
                
            lats = ae_model.encoder(x)  # Shape: (N, 3, 4, 4)
            # Map from -1,1 to 0,1
            binary_bits = (lats >= 0).long()  # Same shape
            
            # Convert the three binary bits from the 3 channels to decimal (0 to 7)
            b0, b1, b2 = binary_bits[:, 0], binary_bits[:, 1], binary_bits[:, 2]
            decimal_grid = (b0 * 4) + (b1 * 2) + (b2 * 1)  # Shape: (B, 4, 4)
            
            # Flatten the 4x4 grid to a 16 length sequence
            sequences = decimal_grid.view(decimal_grid.size(0), -1)  # Shape: (B, 16)
            tokens_list.append(sequences)
            
    tokens = torch.cat(tokens_list, dim=0)[:num_images]  # Shape: (100, 16)
    return tokens

def generate_and_evaluate(ar_model, condition_tokens):
    ar_model.eval()

    num_sequences = condition_tokens.size(0)

    # Each sequence starts with its unique ID (0-99)
    generated = condition_tokens[:, :1].clone()

    with torch.no_grad():
        # Generate all 16 image tokens
        for _ in range(16):
            logits = ar_model(generated)
            next_token_logits = logits[:, -1, :]

            # Restrict predictions to image-token IDs (100-107)
            next_token_logits = next_token_logits.clone()
            next_token_logits[:, :100] = float("-inf")

            next_token = next_token_logits.argmax(
                dim=-1, keepdim=True
            )

            generated = torch.cat(
                [generated, next_token], dim=1
            )

    # Sequence match accuracy
    exact_matches = (
        generated == condition_tokens
    ).all(dim=1).sum().item()

    accuracy = 100.0 * exact_matches / num_sequences

    # indivdual image token accuracy
    token_accuracy = (
        generated[:, 1:] == condition_tokens[:, 1:]
    ).float().mean().item() * 100

    print(
        f"Sequence match accuracy: "
        f"{exact_matches}/{num_sequences} ({accuracy:.2f}%)"
    )
    print(f"Image token accuracy: {token_accuracy:.2f}%")

    return generated

def decode_tokens_to_image(ae_model, generated_tokens):
    # generated_tokens shape: (N, 16)
    grid_decimal = generated_tokens.view(-1, 4, 4)
    
    # Convert decimal (0-7) back to 3 binary bits
    b0 = (grid_decimal >> 2) & 1
    b1 = (grid_decimal >> 1) & 1
    b2 = grid_decimal & 1
    
    binary_bits = torch.stack([b0, b1, b2], dim=1).float()  # same shape
    
    # Map 0, 1 back to -1,1
    signed_binary = torch.where(binary_bits == 1.0, 1.0, -1.0)
    
    # Reconstruct through decoder
    ae_model.eval()
    with torch.no_grad():
        reconstructed_images = ae_model.decoder(signed_binary)
        
    return reconstructed_images


class SequenceToToken(nn.Module):

  def __init__(self,vocab_size=8,embed_dim=128,num_heads=8,num_layers=4,max_len=16,):
    super().__init__()
    self.tok_emb = nn.Embedding(vocab_size, embed_dim)
    self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))

    # 4 layers of transformer encoders, each can process 8 tokens surrounding.
    # norm_first=True (Pre-LN) is crucial for stable training and easy overfitting
    encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads,dim_feedforward=256,batch_first=True,norm_first=True,)
    self.transformer = nn.TransformerEncoder(
        encoder_layer, num_layers=num_layers
    )
    self.head = nn.Linear(embed_dim, vocab_size) # Linear back to our vocab size.

  def forward(self, x):
    seq_len = x.size(1)
    emb = self.tok_emb(x) + self.pos_emb[:, :seq_len, :]
    # -> out shape: [batch size, seq len, 128]

    # mask is [0, -inf,-inf,-inf,-inf...],[0,0,-inf,-inf,-inf,...],...

    # in the case of our images with 17 tokens, the seq_len is 17.
    mask = torch.zeros((seq_len, seq_len), device=x.device)
    causal_mask = mask.masked_fill(
        torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        == 1,
        float("-inf"),
    )

    h = self.transformer(emb, mask=causal_mask)
    logits = self.head(h)
    return logits



'''

The idea is to add an identifier to each image, that way we break the
autoregressive model's limit of one to one mapping.
(i.e if there are multiple outcomes for beginning of a sequence, the model can't differentiate between them after.
two images can start with (0,0,0,0,1,0,1,...) and (0,0,0,0,0,0,1,...))

-> So we add an identifier to each image, in the case of 100 images, it is 0 to 99.

In our autoregressive model, we need to define it a vocabulary, and we don't want the identifier to be 
mistaken for the image tokens 
(i.e sequence 5 collides with a token also value 5.)

To compensate, we shift everything after by 100.

so an example would be an image of id 20:  (20,100,107,103,105,100,102,...)
this way, there will be no collisions of one to many mappings.



input of the autoregressive model is the condensed latent from each image.
'''

# convert the 100 binary latents to 100 image token sequences
dataset_tokens = extract_decimal_tokens(
    model, test_loader, num_images=100
).to(device)

# Add unique sequence ids
num_sequences = dataset_tokens.size(0)

seq_ids = torch.arange(
    num_sequences, device=device
).unsqueeze(1)

# Image tokens 0-7 become tokens 100-107
shifted_tokens = dataset_tokens + 100

# Each sequence into the form of (id,tokens+100,..)
condition_tokens = torch.cat(
    [seq_ids, shifted_tokens], dim=1
)

# total tokens = id + 16 tokens = 17 tokens
# Inputs: tokens 0 to 16. (which is the id and 15 tokenns)
# Targets: tokens 1 to 17. (which is the 16 tokens)
# -> this is the formula, we train to get the next token correct.
# -> the model will learn the mapping of 1->15 tokens and their next tokens.
# targets shifted by 1, => 1 token ahead. looks at tokens leading up in input and match with target at that index. 

inputs = condition_tokens[:, :-1]
targets = condition_tokens[:, 1:]

ar_model = SequenceToToken(
    vocab_size=108,
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    max_len=17
).to(device)

optimizer_ar = torch.optim.AdamW(ar_model.parameters(),lr=0.001,weight_decay=0.0)

criterion = nn.CrossEntropyLoss()

num_ar_epochs = 5000

for epoch in range(num_ar_epochs):
    ar_model.train()

    logits = ar_model(inputs)

    loss = criterion(
        logits.reshape(-1, 108),
        targets.reshape(-1)
    )

    optimizer_ar.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        ar_model.parameters(), max_norm=1.0
    )

    optimizer_ar.step()

    if (epoch + 1) % 500 == 0:
        print(
            f"Epoch {epoch + 1}/{num_ar_epochs}, "
            f"Loss: {loss.item():.6f}"
        )

# Generate and evaluate
print("\n Autoregressive model evaluation")

generated_tokens = generate_and_evaluate(
    ar_model, condition_tokens
)

# Undo shifting
# Restoring tokens
generated_image_tokens = generated_tokens[:, 1:] - 100

# decode to images
reconstructed_from_ar = decode_tokens_to_image(
    model, generated_image_tokens
)



dataiter = iter(test_loader)
orig_images, _ = next(dataiter)

# Display
fig, axes = plt.subplots(
    nrows=2,
    ncols=20,
    figsize=(20, 3)
)

fig.suptitle(
    "Original Images and Generated Images",
    fontsize=12
)

# Get the original images
dataiter = iter(test_loader)
orig_images, _ = next(dataiter)

# Display original and generated images
for i in range(20):
    axes[0, i].imshow(
        orig_images[i].cpu().squeeze().numpy(),
        cmap="gray"
    )
    axes[0, i].axis("off")

    axes[1, i].imshow(
        reconstructed_from_ar[i].detach().cpu().squeeze().numpy(),
        cmap="gray"
    )
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
