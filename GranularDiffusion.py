import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch.optim as optim

class TrainDataset(Dataset):
    def __init__(self, image_folder, label_folder, action_folder, transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.action_folder = action_folder
        self.transform = transform
        self.image_files = os.listdir(image_folder)
        self.label_files = os.listdir(label_folder)
        self.action_files = os.listdir(action_folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        label_name = os.path.join(self.label_folder, self.label_files[idx])
        action_name = os.path.join(self.action_folder, self.action_files[idx])
        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert("RGB")
        action = Image.open(action_name).convert("RGB")
        image = self.transform(image)
        label = self.transform(label)
        action = self.transform(action)
        return image, action, label

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3 + 2 * 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)])
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep, condition, action):
        t = self.time_mlp(timestep)
        x = torch.cat((x, condition, action), dim=1)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

def get_loss(model, condition, action, t, label):
    x_noisy, noise = forward_diffusion_sample(label, t, device)
    noise_pred = model(x_noisy, t, condition, action)
    return F.l1_loss(noise, noise_pred)

def linear_beta_schedule(timesteps, start=0.0001, end=0.025):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

@torch.no_grad()
def sample_timestep(x, t, condition, action):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, condition, action) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def prediction(epoch, condition, action, label, num, epoch_dir):
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 5
    stepsize = int(T / num_images)

    # epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, condition.to(device), action.to(device))
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images + 2, int((T - i) / stepsize))
            show_tensor_image(img.detach().cpu())
            img.to(device)
            if i != 0:
                plt.title("Denoising")  # Add title for the condition image
            else:
                plt.title("Model output")  # Add title for the condition image
    plt.subplot(1, num_images + 2, num_images + 1)
    show_tensor_image(condition.detach().cpu())
    plt.title("Condition Image")  # Add title for the condition image
    plt.subplot(1, num_images + 2, num_images + 2)
    show_tensor_image(label.detach().cpu())
    plt.title("Label Image")  # Add title for the batch image
    plt.savefig(os.path.join(epoch_dir, f"sample_plot_epoch_{epoch}_{num}.png"))
    plt.close()

T = 250
betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

IMG_SIZE = 64
BATCH_SIZE = 1
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

torch.manual_seed(0)
data = TrainDataset('Bigdiffusion(RawD)', 'Bigdiffusion(D)', 'action_images', data_transforms)
train_size = int(0.98 * len(data))  # 90% for training
test_size = len(data) - train_size  # 10% for testing
train_dataset, test_dataset = random_split(data, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Diffusion()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 100
show_epoch = list(range(0, epochs, 10)) + [epochs - 1]
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

trainlosses, testlosses = [], []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0  # To accumulate train losses for this epoch
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    for num_batches_train, (condition, action, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model, condition.to(device), action.to(device), t, label.to(device))
        loss.backward()
        optimizer.step()
        if epoch in show_epoch and num_batches_train % 100 == 0:
            prediction(epoch, condition.to(device), action.to(device), label.to(device), num_batches_train, epoch_dir)  # Save training images
        epoch_loss += loss.item()
    average_loss = epoch_loss / (num_batches_train + 1)
    trainlosses.append(average_loss)
    print(f"Train Loss at epoch {epoch}: {average_loss}")
    #### Evaluate the model on the test set ####
    model.eval()
    with torch.no_grad():
        test_loss = 0  # To accumulate test losses for this epoch
        epoch_dir = os.path.join(output_dir, f"test_epoch_{epoch}")
        for num_batches_test, (condition_test, action_test, label_test) in enumerate(test_dataloader):
            loss_test = get_loss(model, condition_test.to(device), action_test.to(device), torch.full((BATCH_SIZE,), T // 2, device=device, dtype=torch.long), label_test.to(device))
            test_loss += loss_test.item()
            if epoch in show_epoch and num_batches_test % 2 == 0:
                prediction(epoch, condition_test.to(device), action.to(device), label_test.to(device), num_batches_test, epoch_dir)  # Save evaluation images
        # if epoch not in show_epoch:
        avg_test_loss = test_loss / (num_batches_test + 1)
        testlosses.append(avg_test_loss)
        print(f"Test Loss at epoch {epoch}: {avg_test_loss}")



# model_save_path = os.path.join(output_dir, "trained_model.pth")
# torch.save(model.state_dict(), model_save_path)
# print(f"Model saved at {model_save_path}")

plt.figure(figsize=(10, 5))
plt.plot(range(len(trainlosses)), trainlosses, label="Training Loss")
plt.plot(range(len(testlosses)), testlosses, label="Test Loss")
plt.ylim(0, 1)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.legend()
loss_plot_path = os.path.join(output_dir, "loss_vs_epoch.png")
plt.savefig(loss_plot_path)
plt.close()
print(f"Loss plot saved at {loss_plot_path}")
