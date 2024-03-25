import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating the custom Masked Convolution layer
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Mask if a part of the state but must not require grad computations!
        # Thus, register it as a buffer instead of a parameter
        self.register_buffer('mask', torch.ones_like(self.weight))

        # Returns: num_filters, depth, height, width
        _, _, height, width = self.weight.size()
        if mask_type == 'A':
            self.mask[:, :, height // 2, width // 2:] = 0
            self.mask[:, :, height // 2 + 1:] = 0
        elif mask_type == 'B':
            self.mask[:, :, height // 2, width // 2 + 1:] = 0
            self.mask[:, :, height // 2 + 1:] = 0


    def forward(self, x):
        # Masking should not be the part of gradient computation.
        # Using torch.no_grad() instead of self.weight.date *= self.mask to aling with best torch practices
        with torch.no_grad():
            self.weight *= self.mask

        return super().forward(x)



class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Using ModuleList instead of nn.Sequential for more convenient receptive field analysis
        self.layers = nn.ModuleList()

        # Adding type A masked conv layer
        self.layers.append(MaskedConv2d(mask_type='A', in_channels=1, out_channels=64, kernel_size=7, padding=3))
        self.layers.append(nn.ReLU(inplace=True))

        # Adding num_type_B_layers type B masked conv layer
        num_type_B_layers = 5
        for i in range(num_type_B_layers):
            self.layers.append(MaskedConv2d(mask_type='B', in_channels=64, out_channels=64, kernel_size=7, padding=3))
            self.layers.append(nn.ReLU(inplace=True))

        # Adding final 1x1 conv layers - no need for explicit masking since they are automatically type B
        self.layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

    def get_loss(self, x):
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits, x, reduction="mean")
        return loss
    

    @torch.no_grad()
    def sample(self, num_samples):
        self.eval()
        batch_size = num_samples
        device = self.layers[0].weight.device

        # Batch of images containing zeros 
        samples = torch.zeros(batch_size, 1, 28, 28, device=device)

        # Sampling pixels sequentially in raster-scan ordering
        for y in range(28):
            for x in range(28):
                logits = self(samples)
                thetas = F.sigmoid(logits[:, :, y, x])
                pixels = torch.bernoulli(thetas).to(device)
                samples[:, :, y, x] = pixels

        self.train()
        return samples



if __name__ == "__main__":
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    # Creating fake data
    image_batch = torch.randn(64, 1, 28, 28, device=device)

    # Creating the model
    model = PixelCNN().to(device)

    # Sanity checks
    # print(model.get_loss(image_batch))
    print(model.sample(num_samples=1).size())

    exit(-1)


