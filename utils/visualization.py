import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_latent_space(z, labels=None):
    """Visualiza o espaço latente com t-SNE"""
    z_2d = TSNE(n_components=2).fit_transform(z)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1], hue=labels, palette='viridis', alpha=0.7)
    plt.title('Embedding de Preferências Latentes (t-SNE)')
    plt.xlabel('z1')
    plt.ylabel('z2')

def plot_preference_components(model, feature_names):
    """Visualiza os componentes aprendidos"""
    weights = model.decoder.weight.detach().numpy()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(weights.T, annot=True, cmap='coolwarm',
                xticklabels=feature_names)
    plt.title('Pesos das Características por Dimensão Latente')