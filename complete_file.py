# consumer_behavior_avi_completo.py
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.nn import PyroModule
from torch.utils.data import DataLoader, TensorDataset

# ================== CONFIGURAÇÃO ==================
class Config:
    print("\n[1/8] Configurando parametros...")
    latent_dim = 3
    num_products = 50
    learning_rate = 0.001
    epochs = 100
    batch_size = 32

# ================== SIMULAÇÃO DE DADOS ==================
def simulate_data():
    print("\n[2/8] Simulando dados...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Gerar preferências latentes
    true_prefs = np.random.normal(0, 1, size=(1000, Config.latent_dim))
    
    # Gerar características dos produtos
    product_features = np.random.uniform(0, 5, size=(Config.num_products, Config.latent_dim))
    
    # Simular escolhas
    choices = []
    for i in range(1000):
        utilities = np.dot(product_features, true_prefs[i]) + np.random.normal(0, 0.1, Config.num_products)
        choices.append(np.argmax(utilities))
    
    return (torch.tensor(choices).long(),
            torch.tensor(product_features).float(),
            torch.tensor(true_prefs).float())

# ================== MODELO ==================
class ConsumerPreferenceModel(PyroModule):
    def __init__(self):
        super().__init__()
        print("\n[3/8] Inicializando modelo...")
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(Config.num_products, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, Config.latent_dim * 2)
        )
        
        self.decoder = torch.nn.Linear(Config.latent_dim, Config.latent_dim)
        self.product_features = None
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Modelo com {total_params} parametros")

    def setup_product_features(self, features):
        self.product_features = features

    def model(self, x):
        with pyro.plate("consumers", x.shape[0]):
            z = pyro.sample("z", dist.Normal(0, 1).expand([Config.latent_dim]).to_event(1))
            pref_vector = self.decoder(z)
            utilities = torch.mm(self.product_features, pref_vector.t()).t()
            pyro.sample("x", dist.Categorical(logits=utilities), obs=x)

    def guide(self, x):
        with pyro.plate("consumers", x.shape[0]):
            x_onehot = torch.nn.functional.one_hot(x, num_classes=Config.num_products).float()
            h = self.encoder(x_onehot)
            loc = h[..., :Config.latent_dim]
            scale = torch.exp(h[..., Config.latent_dim:])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

# ================== EXECUÇÃO ==================
def main():
    print("\n[0/8] Iniciando...")
    
    # 1. Simular dados
    choices, product_features, _ = simulate_data()
    
    # 2. Preparar DataLoader
    loader = DataLoader(TensorDataset(choices), batch_size=Config.batch_size)
    
    # 3. Criar modelo
    model = ConsumerPreferenceModel()
    model.setup_product_features(product_features)
    
    # 4. Configurar otimização
    optimizer = Adam({"lr": Config.learning_rate})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
    
    # 5. Treinar
    print("\n[5/8] Treinando...")
    for epoch in range(Config.epochs):
        loss = sum(svi.step(x[0]) for x in loader) / len(loader)
        if epoch % 10 == 0:
            print(f"Epoca {epoch}: Loss {loss:.2f}")

if __name__ == "__main__":
    main()