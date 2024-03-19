import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoregressive_RNN(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=3):
        super().__init__()

        # Input a single pixel value (0 or 1, since dataset is binarized)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Predicts logits for probability distribution at each pixel
        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)

        # Learnable initial hidden and cell states
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


    def forward(self, x):
        # Adjust the initial states to match the batch size
        batch_size = x.size(0)
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()

        # Pass input to LSTM and get outputs
        x, _ = self.lstm(x, (h0, c0))

        # Get the parameters of Bernoulli distribution by using self.linear
        # -> nn.Linear treats all dimensions except for the last one as part of the batch dimension
        logits = self.linear(x)
        return logits
    

    def get_loss(self, x):
        batch_size = x.size(0)
        device = self.h0.device

        # Reshape x to [batch_size, seq_lenth, 1] for autoregressive RNN processing
        x = x.view(batch_size, -1, 1)
        # Create inputs and targets for this next-token prediction task
        # inputs, targets = x[:, :-1, :], x[:, 1:, :]
        inputs, targets = torch.cat([torch.zeros(batch_size, 1, 1, device=device), x[:, :-1, :]], dim=1), x

        logits = self(inputs)
    
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        return loss
    

    @torch.no_grad()
    def sample(self, num_samples):
        self.eval()

        device = self.h0.device

        batch_size = num_samples
        hidden_state = self.h0.expand(-1, batch_size, -1).contiguous()
        cell_state = self.c0.expand(-1, batch_size, -1).contiguous()

        samples = torch.zeros(batch_size, 28*28, 1, device=device)

        for i in range(28*28):
            prev_pixels = samples[:, i - 1, :].unsqueeze(dim=1)
            logits, (hidden_state, cell_state) = self.lstm(prev_pixels, (hidden_state, cell_state))

            thetas = F.sigmoid(self.linear(logits))
            next_pixels = torch.bernoulli(thetas).to(device)

            samples[:, i:i+1, :] = next_pixels
        
        samples = samples.view(batch_size, 1, 28, 28)

        self.train()
        return samples



if __name__ == "__main__":
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    # Creating fake data
    image_batch = torch.randn(64, 1, 28, 28, device=device)

    # Creating the model
    model = Autoregressive_RNN().to(device)

    # Sanity checks
    print(model.get_loss(image_batch))
    print(model.sample(num_samples=7))
        


        
