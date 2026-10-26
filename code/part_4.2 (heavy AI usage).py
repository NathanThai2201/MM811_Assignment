import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

# python -m pip install torchmetrics torch-fidelity

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
batch_size = 128
learning_rate = 2e-4
num_epochs = 100            # ~39k steps at batch 128
num_timesteps = 1000        # diffusion steps used for training
disc_levels = 5             # must match the disc_levels used during latent extraction
latent_channels = 10        # latent is C x H x W = 10 x 8 x 8
latent_h = 8
latent_s = 8

# denoiser
model_dim = 192
model_depth = 6             # number of (ResBlock + self-attention) blocks
num_heads = 4
dropout = 0.1

# training tricks
warmup_steps = 500
ema_decay = 0.999           # sample from the EMA weights: big quality boost
sample_every = 10           # save weights + preview grid every N epochs

# sampling
sample_steps = 100          # DDIM steps (instead of 1000 ancestral steps)
ddim_eta = 0.0

# evaluation
num_fid_images = 2000      # FID on 1,000 images is badly inflated; use 10k (50k for the report)
compute_recon_fid = True    # FID of decoded REAL latents = best the autoencoder allows

# cooling pauses
step_pause = 0.0            # seconds after every step (0 = off)
epoch_pause = 10            # seconds after every epoch (0 = off)

train_diffusion = False    # False: load the saved diffusion model instead
diffusion_weights = "cifar10_latent_diffusion.pth"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple Silicon GPU
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ----------------------------------------------------------------------------
# 1. Autoencoder decoder (same layout as in cifar_ae.py)
# ----------------------------------------------------------------------------
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


class Decoder(nn.Module):
    def __init__(self, latent_channels):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


# ----------------------------------------------------------------------------
# 2. Load pre-saved discrete latents & AE decoder
# ----------------------------------------------------------------------------
print("Loading saved discrete latents and Autoencoder weights...")
latent_data = torch.load("cifar10_discrete_latents.pt", map_location="cpu")
train_latents_int = latent_data["train"]  # (N, 10, 8, 8), integer levels 0..disc_levels-1
test_latents_int = latent_data["test"]


def int_to_normalized_float(v, levels=disc_levels):
    """Integer levels -> values in [-1, 1] that the decoder was trained on."""
    if levels == 2:
        return v.float()
    return (v.float() / (levels - 1)) * 2.0 - 1.0


train_latents_float = int_to_normalized_float(train_latents_int)
test_latents_float = int_to_normalized_float(test_latents_int)

levels_present = torch.linspace(-1, 1, disc_levels)
usage = [(train_latents_float == l).float().mean().item() for l in levels_present]
print("Level usage in real latents:", [f"{u:.3f}" for u in usage])

train_latent_loader = DataLoader(
    TensorDataset(train_latents_float), batch_size=batch_size,
    shuffle=True, drop_last=True, num_workers=0,
)

checkpoint = torch.load("discrete_L5_cifar10_ae.pth", map_location=device, weights_only=True)
decoder_state_dict = {
    key.removeprefix("decoder."): value
    for key, value in checkpoint.items()
    if key.startswith("decoder.")
}
decoder = Decoder(latent_channels=latent_channels).to(device)
decoder.load_state_dict(decoder_state_dict)
decoder.eval()
for p in decoder.parameters():
    p.requires_grad_(False)
print("Decoder loaded successfully.")


def quantize_latents(x):
    """Snap real numbers to the nearest allowed discrete level."""
    half = (disc_levels - 1) / 2
    return torch.round(torch.clamp(x, -1.0, 1.0) * half) / half


# ----------------------------------------------------------------------------
# 3. Noise schedule (cosine)
# ----------------------------------------------------------------------------
def make_alphas_cumprod(T, s=0.008):
    x = torch.linspace(0, T, T + 1)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(0, 0.999)
    return torch.cumprod(1 - betas, dim=0)


alphas_cumprod = make_alphas_cumprod(num_timesteps).to(device)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


# ----------------------------------------------------------------------------
# 4. Denoiser: ResBlocks + self-attention over the 8x8 latent grid
# ----------------------------------------------------------------------------
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeResBlock(nn.Module):
    def __init__(self, ch, t_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return x + h


class SelfAttention(nn.Module):
    def __init__(self, ch, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(1)                                # (B, heads, d, HW)
        q, k, v = (a.transpose(-1, -2) for a in (q, k, v))     # (B, heads, HW, d)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class LatentDenoiser(nn.Module):
    def __init__(self, in_channels=latent_channels, dim=model_dim,
                 depth=model_depth, heads=num_heads, dropout=dropout):
        super().__init__()
        self.dim = dim
        t_dim = dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.blocks = nn.ModuleList([TimeResBlock(dim, t_dim, dropout) for _ in range(depth)])
        self.attns = nn.ModuleList([SelfAttention(dim, heads) for _ in range(depth)])
        self.out = nn.Sequential(
            nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv2d(dim, in_channels, 3, padding=1)
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.dim))
        h = self.conv_in(x)
        for blk, attn in zip(self.blocks, self.attns):
            h = attn(blk(h, temb))
        return self.out(h)          # predicts the noise


model = LatentDenoiser().to(device)
ema_model = copy.deepcopy(model).eval()
for p in ema_model.parameters():
    p.requires_grad_(False)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: min(1.0, (s + 1) / warmup_steps)
)
print(f"Denoiser parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


# ----------------------------------------------------------------------------
# 5. Loss and sampling (DDIM, with the x0 prediction clipped to [-1, 1])
# ----------------------------------------------------------------------------
def p_losses(denoise_model, x_start, t):
    noise = torch.randn_like(x_start)
    a = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    b = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    x_noisy = a * x_start + b * noise
    return F.mse_loss(denoise_model(x_noisy, t), noise)


@torch.no_grad()
def sample(model, batch_size=16, steps=sample_steps, eta=ddim_eta):
    model.eval()
    x = torch.randn((batch_size, latent_channels, latent_h, latent_s), device=device)
    ts = torch.linspace(num_timesteps - 1, 0, steps, device=device).long()
    for i, t in enumerate(ts):
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[ts[i + 1]] if i < steps - 1 else torch.tensor(1.0, device=device)
        eps = model(x, t.expand(batch_size))
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()      # eps consistent with clipped x0
        sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
        dir_xt = (1 - a_prev - sigma ** 2).clamp_min(0).sqrt() * eps
        x = a_prev.sqrt() * x0 + dir_xt + sigma * torch.randn_like(x)
    return x


@torch.no_grad()
def generate_images(model, n, bs=500):
    """Sample n latents, snap them to the discrete levels, decode to images in [0, 1]."""
    images = []
    for i in range(0, n, bs):
        z = sample(model, batch_size=min(bs, n - i))
        images.append(decoder(quantize_latents(z)).clamp(0, 1).cpu())
    return torch.cat(images)


def to_uint8(images):
    return (images * 255).round().clamp(0, 255).to(torch.uint8)


# ----------------------------------------------------------------------------
# 6. Training
# ----------------------------------------------------------------------------
if train_diffusion:
    print("Training Discrete Latent Diffusion Model...")
    step_count, t0 = 0, time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_latent_loader):
            x = batch[0].to(device)
            t = torch.randint(0, num_timesteps, (x.shape[0],), device=device).long()

            optimizer.zero_grad(set_to_none=True)
            loss = p_losses(model, x, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # EMA of the weights
            decay = min(ema_decay, (1 + step_count) / (10 + step_count))
            with torch.no_grad():
                for pe, p in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
            step_count += 1

            total_loss += loss.item()
            if step_pause > 0:
                time.sleep(step_pause)
            if step % 100 == 0:
                print(
                    f"epoch {epoch + 1}/{num_epochs} step {step}/{len(train_latent_loader)} "
                    f"loss {loss.item():.4f} ({time.time() - t0:.0f}s elapsed)",
                    flush=True,
                )

        avg_loss = total_loss / len(train_latent_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}", flush=True)

        if (epoch + 1) % sample_every == 0 or epoch == num_epochs - 1:
            torch.save(ema_model.state_dict(), diffusion_weights)
            preview = generate_images(ema_model, 64)
            torchvision.utils.save_image(preview, f"diffusion_samples_epoch{epoch + 1}.png", nrow=8)
            print(f"saved {diffusion_weights} and diffusion_samples_epoch{epoch + 1}.png", flush=True)

        if epoch_pause > 0 and epoch < num_epochs - 1:
            time.sleep(epoch_pause)
else:
    ema_model.load_state_dict(torch.load(diffusion_weights, map_location=device))
    print("Loaded diffusion model from disk.")
ema_model.eval()


# ----------------------------------------------------------------------------
# 7. Generate and plot samples
# ----------------------------------------------------------------------------
generated_images = generate_images(ema_model, 16)

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
fig.suptitle("Generated Images from Discrete Latent Diffusion Model", fontsize=12)
for i in range(16):
    row, col = i // 8, i % 8
    axes[row, col].imshow(generated_images[i].permute(1, 2, 0).numpy())
    axes[row, col].axis("off")
plt.tight_layout()
plt.savefig("generated_diffusion_samples.png")
plt.show()
print("Generation complete! Sample grid saved to 'generated_diffusion_samples.png'.")


# ----------------------------------------------------------------------------
# 8. FID (computed on the CPU)
# ----------------------------------------------------------------------------
@torch.no_grad()
def calculate_fid(model, num_images=num_fid_images, bs=500):
    # reset_real_features=False: the real statistics are kept between the two FIDs
    fid = FrechetInceptionDistance(feature=2048, normalize=False, reset_real_features=False)

    # real images: random CIFAR-10 *train* images (uint8, N x 3 x 32 x 32)
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    pick = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(0))[:num_images]
    real = torch.from_numpy(train_set.data[pick.numpy()]).permute(0, 3, 1, 2).contiguous()
    for i in range(0, real.size(0), bs):
        fid.update(real[i:i + bs], real=True)
    print(f"Real images processed: {real.size(0)}")

    # generated images
    done = 0
    while done < num_images:
        n = min(bs, num_images - done)
        fid.update(to_uint8(generate_images(model, n, bs=n)), real=False)
        done += n
        print(f"FID images: {done}/{num_images}", flush=True)
    score = fid.compute().item()
    print(f"\nFID Score ({num_images} generated images): {score:.4f}  (target < 40)")

    # floor: decode REAL test latents; the diffusion model can't beat this
    if compute_recon_fid:
        fid.reset()      # clears the fake side only
        recon = []
        for i in range(0, min(num_images, test_latents_float.size(0)), bs):
            z = test_latents_float[i:i + bs].to(device)
            recon.append(to_uint8(decoder(z).clamp(0, 1).cpu()))
        recon = torch.cat(recon)
        for i in range(0, recon.size(0), bs):
            fid.update(recon[i:i + bs], real=False)
        floor = fid.compute().item()
        print(f"FID of decoded real test latents ({recon.size(0)} images): {floor:.4f} "
              "(lower bound set by the autoencoder)")
    return score


fid_score = calculate_fid(ema_model)