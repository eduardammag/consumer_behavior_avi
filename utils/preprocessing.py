import numpy as np
import pandas as pd

def preprocess_real_data(filepath):
    """Processa dados reais de escolha do consumidor"""
    df = pd.read_csv(filepath)
    
    # Exemplo: Agregar históricos de compra
    consumer_choices = df.groupby('CustomerID')['StockCode'].value_counts().unstack()
    consumer_choices = consumer_choices.fillna(0)
    
    # Normalização
    choice_matrix = consumer_choices.apply(lambda x: x/x.sum(), axis=1)
    return choice_matrix

def split_data(choice_matrix, test_size=0.2):
    """Divide dados para validação"""
    from sklearn.model_selection import train_test_split
    return train_test_split(choice_matrix, test_size=test_size)