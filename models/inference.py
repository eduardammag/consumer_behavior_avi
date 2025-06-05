import torch
from torch.utils.data import DataLoader, TensorDataset
from vae_model import ConsumerPreferenceModel

def prepare_data(data_path):
    """Prepara os dados para treinamento"""
    data = pd.read_csv(data_path)
    # Converter escolhas em one-hot encoding
    choices = torch.tensor(data['product_id'].values)
    x = torch.nn.functional.one_hot(choices, num_classes=50).float()
    return DataLoader(TensorDataset(x), batch_size=32, shuffle=True)

def run_inference(model, data_loader, epochs=1000):
    """Executa a inferência variacional"""
    model.train(data_loader, num_epochs=epochs)
    # Salvar modelo treinado
    torch.save(model.state_dict(), 'models/trained_model.pt')

def get_latent_representations(model, data_loader):
    """Extrai representações latentes dos consumidores"""
    model.eval()
    z_list = []
    with torch.no_grad():
        for x in data_loader:
            h = model.encoder(x[0].float())
            z = h[..., :model.latent_dim]
            z_list.append(z)
    return torch.cat(z_list, dim=0)