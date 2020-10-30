
#%%
import torch
from torchvision import transforms, datasets

train_data = datasets.MNIST(root='./MNIST-data', train=True, download=True, transform=transforms.ToTensor())
#%%
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # vgg = torch.models.VGG16(pretrained=True)

        self.encoder = torch.nn.Sequential(
            # vgg.features
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.MaxPool2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 16, 5),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 16, 9),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, 9),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return h

    def decode(self, h):
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        h = self.encode(x)
        # print(h.shape)
        # fds
        x_hat = self.decode(h)
        # print(x_hat.shape)
        # sdcds
        return x_hat
# %%
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)

def train(model, epochs=1):
    writer = SummaryWriter(log_dir=f'../../runs/autoencoders/{time.time()}')
    model.train()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    batch_idx = 0
    for epoch in range(epochs):
        for img, label in train_loader:
            # img = img.to(device)
            # label = label.to(device)
            pred = model(img)
            loss = F.mse_loss(pred, img)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('Loss/ConvAE', loss.item(), batch_idx)
            batch_idx += 1
            print('Batch:', batch_idx, 'Loss:', loss.item())
            if batch_idx > 200:
                break
        if batch_idx > 200:
            break

ae = AE()
train(ae)
# %%
for x, _ in train_loader:
    break
t = transforms.ToPILImage()
pred = ae(x)
t(x[0]).show()
t(pred[0]).show()
# %%