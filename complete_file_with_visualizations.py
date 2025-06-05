# advanced_econ_behavior.py
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
from pyro.nn import PyroModule

sns.set_theme(style="darkgrid") 
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

# ================== CONFIGURAÇÃO AVANÇADA ==================
class Config:
    # Hiperparâmetros do modelo
    latent_dim = 4  # Dimensões latentes (preço, qualidade, conveniência, status)
    num_products = 50
    num_consumers = 2000
    learning_rate = 0.005
    epochs = 800
    batch_size = 64
    
    # Configurações de visualização
    plot_3d = True
    interactive_plots = True

# ================== SIMULAÇÃO DE MERCADO REALÍSTICA ==================
def simulate_market():
    print(" Simulando ambiente econômico complexo...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Características dos produtos (preço, qualidade, conveniência, status)
    product_features = np.column_stack([
        np.random.lognormal(mean=1.5, sigma=0.8, size=Config.num_products),  # Preço
        np.random.beta(a=2, b=1.5, size=Config.num_products),               # Qualidade
        np.random.logistic(loc=0.7, scale=0.2, size=Config.num_products),   # Conveniência
        np.random.exponential(scale=0.5, size=Config.num_products)          # Status
    ])
    
    # Segmentos de consumidores (5 clusters ocultos)
    true_prefs = np.zeros((Config.num_consumers, Config.latent_dim))
    cluster_centers = np.array([
        [0.8, 0.3, 0.5, 1.2],  # Consumidores sensíveis ao preço
        [0.2, 1.4, 0.3, 0.8],   # Buscadores de qualidade
        [0.5, 0.5, 1.5, 0.2],   # Valorizam conveniência
        [0.3, 0.7, 0.4, 1.8],   # Consumidores de status
        [1.0, 1.0, 1.0, 1.0]    # Generalistas
    ])
    
    cluster_assignments = np.random.choice(5, size=Config.num_consumers, p=[0.25, 0.2, 0.15, 0.1, 0.3])
    for i in range(Config.num_consumers):
        true_prefs[i] = cluster_centers[cluster_assignments[i]] + np.random.normal(0, 0.1, Config.latent_dim)
    
    # Simulação de escolha com ruído
    choices = []
    for i in range(Config.num_consumers):
        utilities = np.dot(product_features, true_prefs[i]) 
        utilities += np.random.gumbel(scale=0.1, size=Config.num_products)  # Ruído de Gumbel
        choices.append(np.argmax(utilities))
    
    # DataFrame para análise
    product_df = pd.DataFrame(product_features, 
                            columns=['Preço', 'Qualidade', 'Conveniência', 'Status'])
    product_df['ProdutoID'] = range(Config.num_products)
    
    return (torch.tensor(choices).long(),
            torch.tensor(product_features).float(),
            torch.tensor(true_prefs).float(),
            product_df,
            cluster_assignments)

# ================== MODELO BAYESIANO AVANÇADO ==================
class AdvancedConsumerModel(PyroModule):
    def __init__(self):
        super().__init__()
        print("\n Inicializando modelo econômico bayesiano...")
        
        # Encoder profundo
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(Config.num_products, 128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, Config.latent_dim * 2)
        )
        
        # Decoder não-linear
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(Config.latent_dim, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, Config.latent_dim)
        )
        
        # Camada de atenção para produtos
        self.attention = torch.nn.Linear(Config.latent_dim, Config.num_products)
        
        print(f"Arquitetura do modelo:\n{self}\n")
        print(f"Total parâmetros: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        x_onehot = torch.nn.functional.one_hot(x, num_classes=Config.num_products).float()
        h = self.encoder(x_onehot)
        return h[..., :Config.latent_dim]

    def model(self, x):
        pyro.module("decoder", self.decoder)
        pyro.module("attention", self.attention)
        
        with pyro.plate("consumers", x.shape[0]):
            # Prior hierárquico
            z_loc = pyro.sample("z_loc", dist.Normal(0, 1).expand([Config.latent_dim]).to_event(1))
            z_scale = pyro.sample("z_scale", dist.LogNormal(0, 0.5).expand([Config.latent_dim]).to_event(1))
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            
            # Decoder não-linear
            pref_vector = self.decoder(z)
            
            # Mecanismo de atenção
            attn_weights = torch.softmax(self.attention(pref_vector), dim=-1)
            weighted_features = self.product_features * attn_weights.unsqueeze(-1)
            
            # Utilidade com interações complexas
            utilities = torch.einsum('bd,bnd->bn', pref_vector, weighted_features)
            pyro.sample("x", dist.Categorical(logits=utilities), obs=x)

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        
        with pyro.plate("consumers", x.shape[0]):
            x_onehot = torch.nn.functional.one_hot(x, num_classes=Config.num_products).float()
            h = self.encoder(x_onehot)
            loc = h[..., :Config.latent_dim]
            scale = torch.nn.functional.softplus(h[..., Config.latent_dim:]) + 0.01
            pyro.sample("z_loc", dist.Delta(loc).to_event(1))
            pyro.sample("z_scale", dist.Delta(scale).to_event(1))
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

# ================== VISUALIZAÇÕES AVANÇADAS ==================
def create_advanced_visualizations(z_inferred, product_df, true_prefs, clusters):
    print("\n Criando visualizações profissionais...")
    
    # 1. Mapa 3D Interativo de Preferências
    print("Gerando visualização 3D interativa...")
    tsne_3d = TSNE(n_components=3, perplexity=50).fit_transform(z_inferred)
    
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=tsne_3d[:,0], y=tsne_3d[:,1], z=tsne_3d[:,2],
        mode='markers',
        marker=dict(
            size=5,
            color=clusters,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=[f"Cluster: {c}" for c in clusters]
    )])
    
    fig_3d.update_layout(
        title='<b>Espaço Latente 3D de Preferências dos Consumidores</b>',
        scene=dict(
            xaxis_title='Dimensão 1',
            yaxis_title='Dimensão 2',
            zaxis_title='Dimensão 3'
        ),
        width=1000,
        height=800
    )
    
    # 2. Heatmap de Importância de Atributos
    print("Gerando heatmap de atributos...")
    plt.figure(figsize=(14, 8))
    sns.heatmap(product_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação de Atributos de Produtos', pad=20)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 3. Radar Plot de Segmentos
    print("Gerando radar plot de segmentos...")
    cluster_means = []
    for i in range(5):
        cluster_means.append(z_inferred[clusters == i].mean(axis=0))
    
    fig_radar = go.Figure()
    
    for i, mean in enumerate(cluster_means):
        fig_radar.add_trace(go.Scatterpolar(
            r=np.append(mean, mean[0]),
            theta=['Preço', 'Qualidade', 'Conveniência', 'Status', 'Preço'],
            fill='toself',
            name=f'Segmento {i+1}'
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 2]
            )),
        showlegend=True,
        title='Perfil de Preferências por Segmento de Consumidores'
    )
    
    # 4. Animação Temporal (simulada)
    print("Preparando animação temporal...")
    frames = []
    for t in np.linspace(0, 2*np.pi, 24):
        frames.append(go.Frame(data=[go.Scatter3d(
            x=tsne_3d[:,0]*np.cos(t),
            y=tsne_3d[:,1],
            z=tsne_3d[:,2]*np.sin(t),
            mode='markers'
        )]))
    
    fig_3d.frames = frames
    fig_3d.update_layout(updatemenus=[{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "type": "buttons",
        "x": 0.1,
        "y": 0
    }])
    
    return fig_3d, fig_radar

# ================== PIPELINE COMPLETA ==================
def main():
    print(" Iniciando análise econômica comportamental avançada")
    
    # 1. Simulação de mercado
    choices, product_features, true_prefs, product_df, clusters = simulate_market()
    
    # 2. Preparação dos dados
    loader = DataLoader(TensorDataset(choices), batch_size=Config.batch_size, shuffle=True)
    
    # 3. Modelagem
    model = AdvancedConsumerModel()
    model.product_features = product_features
    
    # 4. Otimização
    optimizer = Adam({"lr": Config.learning_rate})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
    
    print("\n Treinamento do modelo:")
    for epoch in range(Config.epochs):
        epoch_loss = 0
        for x_batch in loader:
            loss = svi.step(x_batch[0])
            epoch_loss += loss / len(loader)
        
        # Acessa o LR apenas se já houver otimizações realizadas
        if 0 in optimizer.optim_objs and epoch % 50 == 0:
            current_lr = optimizer.optim_objs[0].param_groups[0]['lr']
            print(f"Época {epoch:03d} | Loss: {epoch_loss:.2f} | LR: {current_lr:.5f}")
        elif epoch % 50 == 0:
            print(f"Época {epoch:03d} | Loss: {epoch_loss:.2f} | LR: (indefinido)")

    # 5. Inferência
    print("\n Extraindo insights...")
    z_inferred = model.forward(choices).detach().numpy()
    
    # 6. Visualizações avançadas
    fig_3d, fig_radar = create_advanced_visualizations(z_inferred, product_df, true_prefs, clusters)
    
    print("\n Análise concluída! Visualizações prontas para apresentação.")
    
    # Exibir resultados
    if Config.interactive_plots:
        fig_3d.show()
        fig_radar.show()
    else:
        plt.show()

if __name__ == "__main__":
    main()