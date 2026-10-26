import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


# Configs 

batch_size = 128
learning_rate = 2e-3
num_epochs = 10
weight_decay = 1e-5

cont_channels = 5             # continuous latent: 5 x 8 x 8  = 320 floats (<= 8x8x5)
disc_channels = 10            # discrete latent:  10 x 8 x 8 entries  (<= 8x8x10)
disc_levels = 5               # discrete values: e.g. 2 is binary.


disc_noise_std = 0.0          # gaussian noise before tanh during discrete training
disc_batchnorm = True         # BatchNorm before quantizing, so all levels get used

# Cooling: pauses so the Mac doesn't run flat out for hours
step_pause = 0.05             
epoch_pause = 30              
train_cont = False           
train_disc = True

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple Silicon GPU
else:
    device = torch.device("cpu")


# Load CIFAR-10 dataset, with pixels in [0,1]
train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    ),
    download=True,
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, drop_last=True,   # 0 workers: avoids the macOS spawn error
)
test_loader = DataLoader(
    test_dataset, batch_size=500, shuffle=False, num_workers=0
)


def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2, dim=(1, 2, 3))
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))

# Residual block
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

# ResNet based autoencoder
# Continuous autoencoder: latent 5x8x8 floats (for the Transformer model)
class ContinuousAE(nn.Module):
    def __init__(self, latent_channels=cont_channels):
        super().__init__()

        # Encoder: 3x32x32 -> latent_channels x 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            ResBlock(64),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            ResBlock(128),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            ResBlock(256),
            ResBlock(256),

            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, latent_channels, 1)
        )

        # Decoder: latent_channels x 8x8 -> 3x32x32
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, padding=1),
            ResBlock(256),
            ResBlock(256),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            ResBlock(128),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            ResBlock(64),

            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        lats = self.encode(x)
        return self.decode(lats)




    
# Discrete autoencoder: latent 10x8x8, with number of discrete values in latents at disc_levels. values per entry (for diffusion)
class DiscreteAE(nn.Module):
    def __init__(
        self,
        latent_channels=disc_channels,
        noise_std=disc_noise_std,
        levels=disc_levels
    ):
        super().__init__()

        assert levels == 2 or levels % 2 == 1
        self.levels = levels
        self.noise_std = noise_std

        # Encoder: 3x32x32 -> latent_channels x 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            ResBlock(64),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            ResBlock(128),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            ResBlock(256),
            ResBlock(256),

            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, latent_channels, 1)
        )

        # Normalize latent channels before quantization
        self.bn = (
            nn.BatchNorm2d(latent_channels)
            if disc_batchnorm else nn.Identity()
        )

        # Decoder: latent_channels x 8x8 -> 3x32x32
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, padding=1),
            ResBlock(256),
            ResBlock(256),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            ResBlock(128),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            ResBlock(64),

            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        z = self.bn(self.encoder(x))

        if self.levels == 2:
            return torch.where(z >= 0, 1.0, -1.0)

        half = (self.levels - 1) / 2
        return torch.round(torch.tanh(z) * half) / half

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.bn(self.encoder(x))

        if self.training and self.noise_std > 0:
            z = z + torch.randn_like(z) * self.noise_std

        if self.levels == 2:
            z = torch.tanh(z)
            hard = torch.where(z >= 0, 1.0, -1.0)

            # STE trick
            z = z + (hard - z).detach()

        else:
            half = (self.levels - 1) / 2
            soft = torch.tanh(z) * half
            hard = torch.round(soft)

            # STE
            z = (soft + (hard - soft).detach()) / half

        return self.decode(z)

# Train and Evaluate

@torch.no_grad()
def evaluate_psnr(model):
    model.eval()
    total, n = 0.0, 0
    for images, _ in test_loader:
        images = images.to(device)
        # eval path: the discrete AE uses its exact quantized latent here
        recs = model(images)
        total += calculate_psnr(images, recs).sum().item()
        n += images.size(0)
    return total / n


def train(model, name, epochs):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.05
    )
    losses = []
    best_psnr = 0.0

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for step, (images, _) in enumerate(train_loader):
            images = images.to(device)
            recs = model(images)
            loss = F.mse_loss(recs, images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if step_pause > 0:
                time.sleep(step_pause)
            if step % 50 == 0:
                print(
                    f"[{name}] epoch {epoch + 1}/{epochs} step {step}/{len(train_loader)} "
                    f"loss {loss.item():.5f} ({time.time() - t0:.0f}s elapsed)",
                    flush=True,
                )

        print(f"[{name}] finished epoch {epoch + 1}/{epochs}, loss {loss.item():.5f}", flush=True)
        if epoch_pause > 0 and epoch < epochs - 1:
            print(f"[{name}] cooling down for {epoch_pause}s ...", flush=True)
            time.sleep(epoch_pause)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            psnr = evaluate_psnr(model)
            print(
                f"[{name}] Epoch {epoch + 1}/{epochs}, "
                f"Loss: {loss.item():.6f}, Test PSNR: {psnr:.2f} dB"
            )
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), f"{name}_cifar10_ae.pth")

    # keep the best checkpoint
    model.load_state_dict(torch.load(f"{name}_cifar10_ae.pth", map_location=device))
    print(f"[{name}] saved best weights (PSNR {best_psnr:.2f} dB)")
    return losses


def load(model, name):
    model.load_state_dict(
        torch.load(f"{name}_cifar10_ae.pth", map_location=device)
    )
    model.to(device)
    print(f"[{name}] loaded from disk")
    return model


def show_reconstructions(model, title, n=16):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:n].to(device)
    with torch.no_grad():
        recs = model(images)
    fig, axes = plt.subplots(2, n, figsize=(n, 2.4))
    fig.suptitle(title, fontsize=10)
    for i in range(n):
        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0).numpy())
        axes[1, i].imshow(recs[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def extract_latents(model, loader):
    """Latents for the whole split, kept for the generative models later."""
    model.eval()
    out = []
    for images, _ in loader:
        out.append(model.encode(images.to(device)).cpu())
    return torch.cat(out, dim=0)








disc_name = "discrete" if disc_levels == 2 else f"discrete_L{disc_levels}"
cont_ae = ContinuousAE().to(device)
disc_ae = DiscreteAE().to(device)

# latent-size checks against the assignment limits
cont_ae.eval()
disc_ae.eval()
x0 = torch.zeros(1, 3, 32, 32, device=device)
cont_shape = tuple(cont_ae.encode(x0).shape[1:])
disc_shape = tuple(disc_ae.encode(x0).shape[1:])
print("Continuous latent shape:", cont_shape)
print("Discrete latent shape:  ", disc_shape)
assert cont_shape[0] * cont_shape[1] * cont_shape[2] <= 8 * 8 * 5
assert disc_shape[0] * disc_shape[1] * disc_shape[2] <= 8 * 8 * 10

if train_cont:
    cont_losses = train(cont_ae, "continuous", num_epochs)
else:
    cont_ae = load(cont_ae, "continuous")

if train_disc:
    disc_losses = train(disc_ae, disc_name, num_epochs)
else:
    disc_ae = load(disc_ae, disc_name)

cont_psnr = evaluate_psnr(cont_ae)
disc_psnr = evaluate_psnr(disc_ae)
print(f"Continuous AE ({cont_shape}) Test PSNR: {cont_psnr:.2f} dB (target > 25)")
print(f"Discrete AE   ({disc_shape}) Test PSNR: {disc_psnr:.2f} dB (target > 25)")

show_reconstructions(cont_ae, f"Continuous AE  PSNR {cont_psnr:.2f} dB")
show_reconstructions(disc_ae, f"Discrete AE  PSNR {disc_psnr:.2f} dB")


# Save latents to disk
def to_int8(v):
    """binary: keep +-1. multi-level: integer indices 0..levels-1."""
    if disc_levels == 2:
        return v.to(torch.int8)
    return ((v + 1) / 2 * (disc_levels - 1)).round().to(torch.int8)


# the training set again, without random flips, in a fixed order
latent_train_loader = DataLoader(
    torchvision.datasets.CIFAR10(
        "./data", train=True, transform=transforms.ToTensor()
    ),
    batch_size=500, shuffle=False, num_workers=0,
)

torch.save(
    {
        "train": extract_latents(cont_ae, latent_train_loader),
        "test": extract_latents(cont_ae, test_loader),
    },
    "cifar10_continuous_latents.pt",
)
torch.save(
    {
        "train": to_int8(extract_latents(disc_ae, latent_train_loader)),
        "test": to_int8(extract_latents(disc_ae, test_loader)),
    },
    "cifar10_discrete_latents.pt",
)
print("Latents saved.")