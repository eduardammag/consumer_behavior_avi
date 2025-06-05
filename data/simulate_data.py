import numpy as np
import pandas as pd

def simulate_consumer_data(num_consumers=1000, num_products=50, latent_dim=3):
    """Simula dados de escolha do consumidor com preferências latentes."""
    # Preferências latentes dos consumidores (z_i)
    true_prefs = np.random.normal(0, 1, size=(num_consumers, latent_dim))
    
    # Características dos produtos (θ)
    product_features = np.random.uniform(0, 5, size=(num_products, latent_dim))
    
    # Simular escolhas (x_i)
    choices = []
    for i in range(num_consumers):
        # Utilidade determinística
        utilities = np.dot(product_features, true_prefs[i])
        # Adicionar ruído
        utilities += np.random.normal(0, 0.1, size=num_products)
        # Escolher produto com maior utilidade
        choice = np.argmax(utilities)
        choices.append(choice)
    
    # Criar DataFrame
    data = pd.DataFrame({
        'consumer_id': np.arange(num_consumers),
        'product_id': choices,
        **{f'latent_{j}': true_prefs[:,j] for j in range(latent_dim)}
    })
    
    return data, product_features

# Gerar e salvar dados
data, products = simulate_consumer_data()
data.to_csv('data/consumer_choices.csv', index=False)
np.save('data/product_features.npy', products)