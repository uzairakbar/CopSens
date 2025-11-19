import numpy as np
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None

class SimpleVAE(nn.Module):
    """
    A simple Variational Autoencoder for default NN handling.
    """
    def __init__(self, input_dim, latent_dim):
        super(SimpleVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def fit_torch_model(tr_data, n_components, user_model=None, epochs=100, lr=1e-3):
    """
    Fits a VAE to the treatment data to learn Latent Confounders.
    """
    if torch is None:
        raise ImportError("Torch is required for neural network models.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = tr_data.shape[1]
    
    if user_model is None:
        model = SimpleVAE(input_dim, n_components).to(device)
    else:
        model = user_model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.FloatTensor(tr_data))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            # Assuming VAE structure output: recon, mu, logvar
            # If user model is custom, it must follow this signature or be adapted
            recon_x, mu, logvar = model(x)
            
            # Loss: MSE + KLD
            mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + kld
            
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

def get_torch_embeddings(model, tr_data):
    """
    Returns mu_u_tr (mean embeddings) and cov_u_t (covariance).
    For VAEs, we usually approximate cov_u_t using the average variance 
    output by the encoder or assuming Identity if latent space is standard normal.
    """
    device = next(model.parameters()).device
    x = torch.FloatTensor(tr_data).to(device)
    with torch.no_grad():
        mu, logvar = model.encode(x)
    
    mu_np = mu.cpu().numpy()
    
    # Estimate average covariance from logvar
    # In VAE, diagonal covariance is output per sample. 
    # We take the average diagonal variance for the "shared" covariance approximation
    var_np = torch.exp(logvar).mean(dim=0).cpu().numpy()
    cov_np = np.diag(var_np)
    
    return mu_np, cov_np
