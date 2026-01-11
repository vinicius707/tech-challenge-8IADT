"""
Dados sintéticos para testes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os


def create_sample_tabular_data(n_samples=100, n_features=30, random_state=42):
    """
    Cria um dataset tabular sintético similar ao dataset de câncer de mama.
    
    Parameters:
    -----------
    n_samples : int
        Número de amostras.
    n_features : int
        Número de features.
    random_state : int
        Seed para reprodutibilidade.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com dados sintéticos.
    """
    np.random.seed(random_state)
    
    # Criar features sintéticas
    data = {}
    
    # Features base (raio, textura, perímetro, etc.)
    base_features = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                     'compactness', 'concavity', 'concave_points', 'symmetry',
                     'fractal_dimension']
    
    # Para cada feature base, criar _mean, _se, _worst
    for base_feat in base_features:
        # Gerar valores com distribuição similar ao dataset real
        mean_val = np.random.uniform(5, 30)
        std_val = np.random.uniform(1, 5)
        
        data[f'{base_feat}_mean'] = np.random.normal(mean_val, std_val, n_samples)
        data[f'{base_feat}_se'] = np.random.normal(mean_val * 0.1, std_val * 0.1, n_samples)
        data[f'{base_feat}_worst'] = np.random.normal(mean_val * 1.2, std_val * 1.5, n_samples)
    
    # Adicionar coluna id
    data['id'] = range(1, n_samples + 1)
    
    # Criar variável alvo (B ou M) com distribuição similar ao dataset real
    # ~62% Benigno, ~38% Maligno
    n_benign = int(n_samples * 0.62)
    diagnosis = ['B'] * n_benign + ['M'] * (n_samples - n_benign)
    np.random.shuffle(diagnosis)
    data['diagnosis'] = diagnosis
    
    df = pd.DataFrame(data)
    
    # Ajustar valores para que casos M tenham valores ligeiramente maiores
    # (simulando características de tumores malignos)
    mask_malignant = df['diagnosis'] == 'M'
    for col in df.columns:
        if col not in ['id', 'diagnosis']:
            df.loc[mask_malignant, col] = df.loc[mask_malignant, col] * 1.1
    
    return df


def create_sample_image_array(shape=(224, 224, 3), grayscale=False, random_state=42):
    """
    Cria um array numpy sintético representando uma imagem.
    
    Parameters:
    -----------
    shape : tuple
        Shape da imagem (height, width, channels).
    grayscale : bool
        Se True, cria imagem em escala de cinza.
    random_state : int
        Seed para reprodutibilidade.
    
    Returns:
    --------
    np.ndarray
        Array numpy representando a imagem (valores 0-255).
    """
    np.random.seed(random_state)
    
    if grayscale:
        if len(shape) == 3:
            shape = (shape[0], shape[1])
        img = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    
    return img


def create_temp_image_file(temp_dir, filename='test_image.jpg', shape=(224, 224, 3), grayscale=False):
    """
    Cria um arquivo de imagem temporário para testes.
    
    Parameters:
    -----------
    temp_dir : str ou Path
        Diretório temporário.
    filename : str
        Nome do arquivo.
    shape : tuple
        Shape da imagem.
    grayscale : bool
        Se True, cria imagem em escala de cinza.
    
    Returns:
    --------
    str
        Caminho do arquivo criado.
    """
    from PIL import Image
    
    img_array = create_sample_image_array(shape, grayscale)
    
    if grayscale or len(img_array.shape) == 2:
        img = Image.fromarray(img_array, mode='L')
    else:
        img = Image.fromarray(img_array, mode='RGB')
    
    filepath = Path(temp_dir) / filename
    img.save(filepath)
    
    return str(filepath)


def create_temp_dataset_structure(temp_dir, n_images_per_class=5, structure='train_test_val'):
    """
    Cria uma estrutura de diretórios temporária simulando um dataset de imagens.
    
    Parameters:
    -----------
    temp_dir : str ou Path
        Diretório temporário base.
    n_images_per_class : int
        Número de imagens por classe.
    structure : str
        Tipo de estrutura: 'train_test_val' ou 'class_folders'.
    
    Returns:
    --------
    dict
        Dicionário com informações sobre a estrutura criada.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    classes = ['Normal', 'Pneumonia']  # Para pneumonia
    # Ou ['BENIGN', 'MALIGNANT'] para câncer de mama
    
    image_paths = []
    
    if structure == 'train_test_val':
        splits = ['train', 'test', 'val']
        for split in splits:
            split_dir = temp_dir / split
            split_dir.mkdir(exist_ok=True)
            
            for class_name in classes:
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
                
                for i in range(n_images_per_class):
                    filename = f'{class_name}_{split}_{i}.jpg'
                    img_path = create_temp_image_file(
                        class_dir,
                        filename=filename,
                        shape=(224, 224, 3)
                    )
                    image_paths.append({
                        'path': img_path,
                        'label': class_name,
                        'split': split
                    })
    else:  # class_folders
        for class_name in classes:
            class_dir = temp_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            for i in range(n_images_per_class):
                filename = f'{class_name}_{i}.jpg'
                img_path = create_temp_image_file(
                    class_dir,
                    filename=filename,
                    shape=(224, 224, 3)
                )
                image_paths.append({
                    'path': img_path,
                    'label': class_name,
                    'split': 'unknown'
                })
    
    return {
        'base_path': str(temp_dir),
        'image_paths': image_paths,
        'classes': classes
    }
