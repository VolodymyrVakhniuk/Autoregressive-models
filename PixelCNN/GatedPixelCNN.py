import torch
import torch.nn as nn
import torch.nn.functional as F


# Creating the custom Masked Convolution layer
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Mask if a part of the state but must not require grad computations!
        # Thus, register it as a buffer instead of a parameter
        self.register_buffer('mask', self.create_mask(mask_type))


    def create_mask(self, mask_type):
        # Create a mask of ones
        mask = torch.ones_like(self.weight)

        # Returns: num_filters, depth, height, width
        _, _, height, width = self.weight.size()

        # Given mask_type, zeros out corresponding entries
        if mask_type == 'A':
            mask[:, :, height // 2, width // 2:] = 0
            mask[:, :, height // 2 + 1:] = 0
        elif mask_type == 'B':
            mask[:, :, height // 2, width // 2 + 1:] = 0
            mask[:, :, height // 2 + 1:] = 0
        elif mask_type == 'V':
            mask[:, :, height // 2 + 1:, :] = 0

        return mask


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
    


# Layer that rolls down convolutional feature maps along y axis
# Will be used while feeding the info from vertical conv to horizontal conv 
# to ensure that vertical conv does not violate autoregressive property 
class RollDown(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.roll(x, shifts=1, dims=2)
        x[:, :, 0, :] = 0
        return x
    


class GatedBlock(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size):
        super().__init__()

        self.vertical_conv = MaskedConv2d(
            mask_type = 'V', 
            in_channels = in_channels,
            # 2 * out_channels because of gating: activation features + gate features
            out_channels = 2 * out_channels,
            kernel_size = kernel_size,
            # kernel_size//2 to ensure the spacial dims of input and output are equal
            padding = kernel_size // 2
        )

        self.horizontal_conv = MaskedConv2d(
            # mask_type is either A or B
            mask_type = mask_type,
            in_channels = in_channels,
            # 2 * out_channels because of gating: activation features + gate features
            out_channels = 2 * out_channels,
            kernel_size= (1, kernel_size),
            # kernel_size//2 to ensure the spacial dims of input and output are equal
            padding=(0, kernel_size // 2)
        )

        self.vertical_to_horizontal_conv = nn.Sequential(
            RollDown(),
            # 2 * out_channels and 2 * out_channels because of gating: activation features + gate features
            nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1)
        )

        self.post_horizontal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.mask_type = mask_type


    def forward(self, v_in, h_in):
        # 1) Computing vertical conv stack output
        v_features = self.vertical_conv(v_in)
        v_out = self.gate(v_features)

        # 2) Computing horizontal conv stack output
        h_features = self.horizontal_conv(h_in)

        # -> Feeding information from vertical stack to horizonal stack
        h_features += self.vertical_to_horizontal_conv(v_features)
        h_features = self.post_horizontal_conv(self.gate(h_features))

        # -> Implementing residual connection if this is not the first gated block
        # If type A mask, then it is a first gated block; 
        # If type B mask, then it is a second gated block
        h_out = h_features if self.mask_type == 'A' else h_in + h_features
        
        return v_out, h_out


    def gate(self, features):
        # Split conv feature maps in 2 stacks along channel dimension
        activativation_features, gate_features = torch.chunk(features, 2, dim=1)

        # sigmoind output falls in [0, 1] => it decides how important corresponding activation features are:
        gated_features = F.tanh(activativation_features) * F.sigmoid(gate_features)

        return gated_features



class GatedPixelCNN(nn.Module):
    def __init__(self, image_channels):
        super().__init__()

        # First type A masked gated convolutional layer
        self.gated_block_A = GatedBlock(mask_type='A', in_channels=image_channels, out_channels=128, kernel_size=5)

        # Main type B masked gated convolutional layers
        num_type_B_layers = 5
        self.gated_blocks_B = nn.ModuleList()

        for i in range(num_type_B_layers):
            self.gated_blocks_B.append(GatedBlock(mask_type='B', in_channels=128, out_channels=128, kernel_size=5))
        
        # Final output conv layers
        self.final_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=image_channels, kernel_size=1)
        )
        
        self.image_channels = image_channels


    def forward(self, x):
        v, h = self.gated_block_A(x, x)
        for layer in self.gated_blocks_B:
            v, h = layer(v, h)

        # Combining vertical and horizontal stacks and computing the logits
        logits = self.final_conv_layers(v + h)
        return logits


    def get_loss(self, x):
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits, x, reduction="mean")
        return loss


    @torch.no_grad()
    def sample(self, num_samples):
        self.eval()
        batch_size = num_samples
        device = self.final_conv_layers[0].weight.device

        # Batch of images containing zeros 
        samples = torch.zeros(batch_size, self.image_channels, 28, 28, device=device)

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
    model = GatedPixelCNN(image_channels=1).to(device)

    # Sanity checks
    # print(model.get_loss(image_batch))
    # print(model.sample(num_samples=1).size())

    exit(-1)




    

