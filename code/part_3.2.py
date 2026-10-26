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



class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=8,
        embed_dim=128,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        max_len=16,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.bos_token = vocab_size

        # Source tokens: 0 to 7
        self.src_embedding = nn.Embedding(vocab_size, embed_dim)

        # Target tokens: 0 to 7, plus BOS token 8
        self.tgt_embedding = nn.Embedding(vocab_size + 1, embed_dim)

        self.src_pos_embedding = nn.Parameter(
            torch.zeros(1, max_len, embed_dim)
        )
        self.tgt_pos_embedding = nn.Parameter(
            torch.zeros(1, max_len, embed_dim)
        )

        # Encoder: reads the complete source sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # Decoder: uses causal self-attention and encoder cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # Predict one of the 8 actual latent tokens
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def encode(self, src):
        src_len = src.size(1)

        src_emb = (
            self.src_embedding(src)
            + self.src_pos_embedding[:, :src_len, :]
        )

        memory = self.encoder(src_emb)
        return memory

    def decode(self, tgt_input, memory):
        tgt_len = tgt_input.size(1)

        tgt_emb = (
            self.tgt_embedding(tgt_input)
            + self.tgt_pos_embedding[:, :tgt_len, :]
        )

        # Prevent each decoder position from seeing future target tokens
        causal_mask = torch.triu(
            torch.ones(
                tgt_len, tgt_len,
                device=tgt_input.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        # Cross-attention to memory is handled inside TransformerDecoder
        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
        )

        logits = self.output_head(decoded)
        return logits

    def forward(self, src, tgt_input):
        memory = self.encode(src)
        logits = self.decode(tgt_input, memory)
        return logits

# Extract latent sequences from the autoencoder.
# Each image becomes a sequence of 16 tokens, each in [0, 7].
dataset_tokens = extract_decimal_tokens(
    model, test_loader, num_images=100
).to(device)

print("Latent token shape:", dataset_tokens.shape)
# Expected: (100, 16)

# For this experiment, source and target are the same latent sequence.
src = dataset_tokens
target = dataset_tokens

# Decoder starts with BOS and then receives the previous target tokens.
BOS_TOKEN = 8

bos_column = torch.full(
    (target.size(0), 1),
    BOS_TOKEN,
    dtype=torch.long,
    device=device,
)

tgt_input = torch.cat(
    [bos_column, target[:, :-1]],
    dim=1,
)

print("Source shape:", src.shape)          # (100, 16)
print("Decoder input shape:", tgt_input.shape)  # (100, 16)
print("Target shape:", target.shape)       # (100, 16)

ar_model = Seq2SeqTransformer(
    vocab_size=8,
    embed_dim=128,
    num_heads=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    max_len=16,
).to(device)

optimizer_seq2seq = torch.optim.AdamW(
    ar_model.parameters(),
    lr=0.001,
    weight_decay=0.0,
)

criterion = nn.CrossEntropyLoss()
num_seq2seq_epochs = 5000

for epoch in range(num_seq2seq_epochs):
    ar_model.train()

    logits = ar_model(src, tgt_input)

    loss = criterion(
        logits.reshape(-1, 8),
        target.reshape(-1),
    )

    optimizer_seq2seq.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        ar_model.parameters(),
        max_norm=1.0,
    )

    optimizer_seq2seq.step()

    if (epoch + 1) % 500 == 0:
        print(
            f"Seq2Seq Epoch [{epoch + 1}/{num_seq2seq_epochs}], "
            f"Loss: {loss.item():.6f}"
        )

@torch.no_grad()
def generate_and_evaluate_seq2seq(model, src, target):
    model.eval()

    # Encode the source only once
    memory = model.encode(src)

    # Begin decoding with BOS
    generated = torch.full(
        (src.size(0), 1),
        model.bos_token,
        dtype=torch.long,
        device=src.device,
    )

    for _ in range(target.size(1)):
        logits = model.decode(generated, memory)

        # Predict the next token from the final decoder position
        next_token = logits[:, -1, :].argmax(dim=-1)

        generated = torch.cat(
            [generated, next_token.unsqueeze(1)],
            dim=1,
        )

    # Remove BOS; retain the 16 generated latent tokens
    generated_tokens = generated[:, 1:]

    token_accuracy = (
        (generated_tokens == target).float().mean().item()
    )

    sequence_accuracy = (
        (generated_tokens == target).all(dim=1)
        .float().mean().item()
    )

    print(f"Token Accuracy: {token_accuracy * 100:.2f}%")
    print(f"Exact Sequence Accuracy: {sequence_accuracy * 100:.2f}%")

    return generated_tokens


generated_tokens = generate_and_evaluate_seq2seq(
    ar_model, src, target
)
