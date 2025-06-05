import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

class ConsumerPreferenceModel:
    def __init__(self, num_products, latent_dim=3, feature_dim=3):
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.num_products = num_products
        
        # Carregar características dos produtos
        self.product_features = torch.tensor(np.load('data/product_features.npy'))
        
        # Rede de inferência amortizada (encoder)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_products, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, latent_dim * 2))  # média e log-variância
        
        # Decoder (modelo generativo)
        self.decoder = torch.nn.Linear(latent_dim, feature_dim)
    
    def model(self, x):
        """Modelo generativo p(x|z)p(z)"""
        pyro.module("decoder", self.decoder)
        
        with pyro.plate("consumers", x.shape[0]):
            # Prior sobre preferências latentes
            z = pyro.sample("z", dist.Normal(0, 1).expand([self.latent_dim]).to_event(1))
            
            # Projetar para espaço de características
            pref_vector = self.decoder(z)
            
            # Calcular utilidades para todos produtos
            utilities = torch.matmul(self.product_features, pref_vector.t()).t()
            
            # Observações categóricas
            pyro.sample("x", dist.Categorical(logits=utilities), obs=x)
    
    def guide(self, x):
        """Guia variacional amortizado q(z|x)"""
        pyro.module("encoder", self.encoder)
        
        with pyro.plate("consumers", x.shape[0]):
            # Passar dados pelo encoder
            h = self.encoder(x.float())
            
            # Parâmetros da distribuição variacional
            loc = h[..., :self.latent_dim]
            scale = torch.exp(h[..., self.latent_dim:])
            
            # Amostrar da distribuição variacional
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))
    
    def train(self, data_loader, num_epochs=1000):
        optimizer = Adam({"lr": 0.01})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        for epoch in range(num_epochs):
            total_loss = 0
            for x in data_loader:
                loss = svi.step(x)
                total_loss += loss / len(x)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {total_loss}")