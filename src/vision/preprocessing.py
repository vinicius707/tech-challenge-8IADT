"""
Módulo para pré-processamento de imagens e data augmentation.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import pandas as pd


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    grayscale: bool = False,
    normalize: bool = True
) -> np.ndarray:
    """
    Pré-processa uma única imagem.
    
    Parameters:
    -----------
    image_path : str
        Caminho para a imagem.
    target_size : tuple
        Tamanho alvo (width, height).
    grayscale : bool
        Se True, converte para escala de cinza.
    normalize : bool
        Se True, normaliza pixels para [0, 1].
    
    Returns:
    --------
    np.ndarray
        Imagem pré-processada.
    """
    img = Image.open(image_path)
    
    # Converter para RGB se necessário
    if img.mode != 'RGB' and not grayscale:
        img = img.convert('RGB')
    elif grayscale and img.mode != 'L':
        img = img.convert('L')
    
    # Redimensionar
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Converter para array
    img_array = np.array(img)
    
    # Adicionar dimensão de canal se necessário
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    
    # Normalizar
    if normalize:
        img_array = img_array.astype(np.float32) / 255.0
    
    return img_array


def create_data_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    color_mode: str = 'rgb',
    class_mode: str = 'categorical',
    augmentation: bool = True
) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    """
    Cria data generators para treino, validação e teste.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        DataFrame com dados de treino (colunas: 'image_path', 'label').
    val_df : pd.DataFrame
        DataFrame com dados de validação.
    test_df : pd.DataFrame
        DataFrame com dados de teste.
    image_size : tuple
        Tamanho das imagens (width, height).
    batch_size : int
        Tamanho do batch.
    color_mode : str
        Modo de cor: 'rgb', 'grayscale', ou 'rgba'.
    class_mode : str
        Modo de classe: 'categorical', 'binary', ou 'sparse'.
    augmentation : bool
        Se True, aplica data augmentation no conjunto de treino.
    
    Returns:
    --------
    tuple
        (train_generator, val_generator, test_generator)
    """
    # Data augmentation para treino
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Sem augmentation para validação e teste
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Criar generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='image_path',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='image_path',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col='image_path',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def split_image_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide o dataset de imagens em treino, validação e teste.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com colunas 'image_path' e 'label'.
    test_size : float
        Proporção do conjunto de teste.
    validation_size : float
        Proporção do conjunto de validação.
    random_state : int
        Seed para reprodutibilidade.
    stratify : bool
        Se True, mantém proporção de classes.
    
    Returns:
    --------
    tuple
        (train_df, val_df, test_df)
    """
    # Primeiro, separar treino+validação do teste
    stratify_col = df['label'] if stratify else None
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Depois, separar treino da validação
    val_size_relative = validation_size / (1 - test_size)
    stratify_col = train_val_df['label'] if stratify else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_relative,
        random_state=random_state,
        stratify=stratify_col
    )
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def apply_augmentation(
    image: np.ndarray,
    rotation_range: int = 20,
    width_shift: float = 0.1,
    height_shift: float = 0.1,
    zoom_range: float = 0.1,
    horizontal_flip: bool = True
) -> np.ndarray:
    """
    Aplica transformações de data augmentation em uma imagem.
    
    Parameters:
    -----------
    image : np.ndarray
        Imagem como array numpy.
    rotation_range : int
        Faixa de rotação em graus.
    width_shift : float
        Faixa de deslocamento horizontal (proporção).
    height_shift : float
        Faixa de deslocamento vertical (proporção).
    zoom_range : float
        Faixa de zoom.
    horizontal_flip : bool
        Se True, aplica flip horizontal aleatório.
    
    Returns:
    --------
    np.ndarray
        Imagem aumentada.
    """
    from scipy.ndimage import rotate, shift, zoom
    from scipy.ndimage import gaussian_filter
    
    augmented = image.copy()
    
    # Rotação
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        augmented = rotate(augmented, angle, axes=(0, 1), reshape=False, mode='nearest')
    
    # Deslocamento
    if width_shift > 0 or height_shift > 0:
        shift_x = np.random.uniform(-width_shift, width_shift) * image.shape[1]
        shift_y = np.random.uniform(-height_shift, height_shift) * image.shape[0]
        augmented = shift(augmented, (shift_y, shift_x, 0), mode='nearest')
    
    # Zoom
    if zoom_range > 0:
        zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        augmented = zoom(augmented, (zoom_factor, zoom_factor, 1), mode='nearest')
        # Recortar para tamanho original se necessário
        if zoom_factor > 1:
            h, w = image.shape[:2]
            start_h = (augmented.shape[0] - h) // 2
            start_w = (augmented.shape[1] - w) // 2
            augmented = augmented[start_h:start_h+h, start_w:start_w+w]
        elif zoom_factor < 1:
            # Padding se necessário
            h, w = image.shape[:2]
            pad_h = (h - augmented.shape[0]) // 2
            pad_w = (w - augmented.shape[1]) // 2
            augmented = np.pad(augmented, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
    
    # Flip horizontal
    if horizontal_flip and np.random.random() > 0.5:
        augmented = np.fliplr(augmented)
    
    return augmented


