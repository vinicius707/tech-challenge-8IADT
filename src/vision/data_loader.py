"""
Módulo para carregamento e preparação de datasets de imagens médicas.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import kagglehub


def download_pneumonia_dataset(target_path: str = "data/images/pneumonia") -> str:
    """
    Baixa o dataset de pneumonia em raio-X do Kaggle.
    
    Parameters:
    -----------
    target_path : str
        Caminho onde o dataset será salvo.
    
    Returns:
    --------
    str
        Caminho para o dataset baixado.
    """
    print("Baixando dataset de pneumonia em raio-X...")
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Dataset baixado em: {path}")
    
    # Criar diretório de destino se não existir
    os.makedirs(target_path, exist_ok=True)
    
    return path


def download_breast_cancer_dataset(target_path: str = "data/images/breast_cancer") -> str:
    """
    Baixa o dataset de câncer de mama (CBIS-DDSM) do Kaggle.
    
    Parameters:
    -----------
    target_path : str
        Caminho onde o dataset será salvo.
    
    Returns:
    --------
    str
        Caminho para o dataset baixado.
    """
    print("Baixando dataset de câncer de mama (CBIS-DDSM)...")
    path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
    print(f"Dataset baixado em: {path}")
    
    # Criar diretório de destino se não existir
    os.makedirs(target_path, exist_ok=True)
    
    return path


def load_image_dataset(
    base_path: str,
    subdirs: Optional[List[str]] = None,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
) -> pd.DataFrame:
    """
    Carrega imagens de um diretório estruturado e cria um DataFrame com paths e labels.
    
    Parameters:
    -----------
    base_path : str
        Caminho base do dataset.
    subdirs : list, optional
        Lista de subdiretórios para processar (ex: ['train', 'test', 'val']).
        Se None, processa todos os subdiretórios encontrados.
    image_extensions : tuple
        Extensões de arquivo de imagem aceitas.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com colunas 'image_path' e 'label'.
    """
    base_path = Path(base_path)
    data = []
    
    # Se subdirs não for especificado, procurar por subdiretórios comuns
    if subdirs is None:
        subdirs = ['train', 'test', 'val', 'validation']
    
    # Processar cada subdiretório
    for subdir in subdirs:
        subdir_path = base_path / subdir
        if not subdir_path.exists():
            continue
        
        # Procurar por classes (subdiretórios dentro de train/test/val)
        for class_dir in subdir_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            label = class_dir.name
            # Processar imagens no diretório da classe
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    data.append({
                        'image_path': str(img_file),
                        'label': label,
                        'split': subdir
                    })
    
    # Se não encontrou estrutura train/test/val, procurar diretamente por classes
    if len(data) == 0:
        for class_dir in base_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            label = class_dir.name
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    data.append({
                        'image_path': str(img_file),
                        'label': label,
                        'split': 'unknown'
                    })
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        raise ValueError(f"Nenhuma imagem encontrada em {base_path}")
    
    return df


def create_dataframe(
    base_path: str,
    structure_type: str = "class_folders"
) -> pd.DataFrame:
    """
    Cria um DataFrame com paths e labels das imagens.
    
    Parameters:
    -----------
    base_path : str
        Caminho base do dataset.
    structure_type : str
        Tipo de estrutura: 'class_folders' ou 'train_test_split'.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com informações das imagens.
    """
    return load_image_dataset(base_path)


def get_class_distribution(df: pd.DataFrame, split_col: Optional[str] = 'split') -> pd.DataFrame:
    """
    Analisa a distribuição de classes no dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com colunas 'label' e opcionalmente 'split'.
    split_col : str, optional
        Nome da coluna que indica o split (train/test/val).
    
    Returns:
    --------
    pd.DataFrame
        Tabela com distribuição de classes.
    """
    if split_col and split_col in df.columns:
        distribution = pd.crosstab(df[split_col], df['label'], margins=True)
    else:
        distribution = df['label'].value_counts().to_frame('count')
        distribution['percentage'] = (distribution['count'] / len(df) * 100).round(2)
    
    return distribution


def validate_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida imagens e remove arquivos corrompidos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com coluna 'image_path'.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com apenas imagens válidas.
    """
    from PIL import Image
    
    valid_paths = []
    
    for idx, row in df.iterrows():
        try:
            img = Image.open(row['image_path'])
            img.verify()  # Verifica se a imagem não está corrompida
            valid_paths.append(idx)
        except Exception as e:
            print(f"Imagem inválida removida: {row['image_path']} - {str(e)}")
    
    return df.loc[valid_paths].reset_index(drop=True)


def get_image_info(df: pd.DataFrame, sample_size: int = 100) -> dict:
    """
    Obtém informações sobre as imagens do dataset (dimensões, formatos, etc.).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com coluna 'image_path'.
    sample_size : int
        Número de imagens para amostrar (para análise rápida).
    
    Returns:
    --------
    dict
        Dicionário com informações sobre as imagens.
    """
    from PIL import Image
    
    sample_df = df.sample(min(sample_size, len(df)))
    
    widths = []
    heights = []
    formats = []
    
    for img_path in sample_df['image_path']:
        try:
            img = Image.open(img_path)
            widths.append(img.width)
            heights.append(img.height)
            formats.append(img.format)
        except Exception as e:
            print(f"Erro ao processar {img_path}: {str(e)}")
    
    info = {
        'width_mean': np.mean(widths) if widths else 0,
        'width_std': np.std(widths) if widths else 0,
        'height_mean': np.mean(heights) if heights else 0,
        'height_std': np.std(heights) if heights else 0,
        'formats': pd.Series(formats).value_counts().to_dict() if formats else {},
        'total_images': len(df)
    }
    
    return info


