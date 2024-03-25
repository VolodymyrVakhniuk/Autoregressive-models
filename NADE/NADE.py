import torch
import torch.nn as nn
import torch.nn.functional as F


class NADE(nn.Module):
    def __init__(self, hidden_dim=256):

        super().__init__()
        
        self.W = nn.Parameter(torch.randn(hidden_dim, 28*28))
        self.b = nn.Parameter(torch.zeros(hidden_dim))

        self.V = nn.Parameter(torch.randn(28*28, hidden_dim))
        self.c = nn.Parameter(torch.zeros(28*28))

        self.hidden_dim = hidden_dim

    
    def forward(self, x:torch.tensor):
        device = self.W.device
        batch_size = x.size(0)

        logits = torch.zeros(28*28, batch_size, device=device)
        
        # Transposing x for more convenient further computation
        x = torch.permute(x, (1, 0))
        WX_cache = torch.zeros(self.hidden_dim, batch_size, device=device)

        for k in range(28*28 - 1):
            h_k = 0.0
            if k == 0:
                # Invokes torch broadcasting
                h_k = F.sigmoid(torch.zeros(self.hidden_dim, batch_size, device=device) + self.b.unsqueeze(dim=1))
            else:
                dot_prod = self.W[:, k-1:k] @ x[k-1:k]
                h_k = F.sigmoid(WX_cache + dot_prod + self.b.unsqueeze(dim=1))
                WX_cache = WX_cache + dot_prod

            logit = self.V[k] @ h_k
            logits[k] = logit

        # Transposing back to have batch as the first dimension
        return torch.permute(logits, (1, 0))


    def get_loss(self, x):
        batch_size = x.size(0)
        
        # Preparing data and targets
        x = x.view(batch_size, -1)
        # Creating inputs and targets for this next-token prediction task
        inputs, targets = x[:, :-1], x

        logits = self(inputs)
        
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        return loss
    

    @torch.no_grad()
    def sample(self, num_samples):
        self.eval()
        
        device = self.W.device
        batch_size = num_samples

        # Making batch_size a second dim for more convenient further computation
        samples = torch.zeros(28*28, batch_size, device=device)
        WX_cache = torch.zeros(self.hidden_dim, batch_size, device=device)
        
        for k in range(28*28):
            h_k = 0.0
            if k == 0:
                # Invokes torch broadcasting
                h_k = F.sigmoid(torch.zeros(self.hidden_dim, batch_size, device=device) + self.b.unsqueeze(dim=1))
            else:
                dot_prod = self.W[:, k-1:k] @ samples[k-1:k]
                h_k = F.sigmoid(WX_cache + dot_prod + self.b.unsqueeze(dim=1))
                WX_cache = WX_cache + dot_prod

            logit = self.V[k] @ h_k
            theta = F.sigmoid(logit)
            samples[k] = torch.bernoulli(theta).to(device)

        samples = samples.permute((1, 0)).view(batch_size, 1, 28, 28)

        self.train()
        return samples


if __name__ == "__main__":
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    # Creating fake data
    image_batch = torch.randn(64, 1, 28, 28, device=device)

    # Creating the model
    model = NADE().to(device)

    # Sanity checks
    # print(model.get_loss(image_batch))
    # print(model.sample(num_samples=8).size())

    exit(-1)

