import yaml
from models.vae_model import ConsumerPreferenceModel
from models.inference import prepare_data, run_inference
from utils.visualization import plot_latent_space
from consumer_behavior_avi.models.vae_model import ConsumerPreferenceModel
from consumer_behavior_avi.models.inference import prepare_data, run_inference
def main():
    # Carregar configurações
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Preparar dados
    loader = prepare_data(config['data']['simulated_data_path'])
    
    # Inicializar modelo
    model = ConsumerPreferenceModel(
        num_products=50,
        latent_dim=config['model']['latent_dim']
    )
    
    # Treinar
    run_inference(model, loader, epochs=config['model']['epochs'])
    
    # Visualizar resultados
    z = model.get_latent_representations(loader)
    plot_latent_space(z)

if __name__ == "__main__":
    main()