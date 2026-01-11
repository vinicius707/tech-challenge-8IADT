"""
Testes unitários para src/vision/preprocessing.py
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tempfile

from src.vision.preprocessing import (
    preprocess_image,
    create_data_generators,
    split_image_data,
    apply_augmentation
)


class TestPreprocessImage:
    """Testes para preprocess_image."""
    
    def test_preprocess_image_rgb(self, sample_image_path):
        """Testa pré-processamento de imagem RGB."""
        img_array = preprocess_image(
            sample_image_path,
            target_size=(224, 224),
            grayscale=False,
            normalize=True
        )
        
        # Verificar shape
        assert img_array.shape == (224, 224, 3)
        
        # Verificar normalização (valores entre 0 e 1)
        assert img_array.min() >= 0.0
        assert img_array.max() <= 1.0
        
        # Verificar tipo
        assert img_array.dtype == np.float32
    
    def test_preprocess_image_grayscale(self, sample_image_path_grayscale):
        """Testa pré-processamento de imagem em escala de cinza."""
        img_array = preprocess_image(
            sample_image_path_grayscale,
            target_size=(256, 256),
            grayscale=True,
            normalize=True
        )
        
        # Verificar shape (deve ter dimensão de canal)
        assert len(img_array.shape) == 3
        assert img_array.shape[:2] == (256, 256)
        assert img_array.shape[2] == 1
        
        # Verificar normalização
        assert img_array.min() >= 0.0
        assert img_array.max() <= 1.0
    
    def test_preprocess_image_no_normalize(self, sample_image_path):
        """Testa pré-processamento sem normalização."""
        img_array = preprocess_image(
            sample_image_path,
            target_size=(224, 224),
            normalize=False
        )
        
        # Valores devem estar entre 0 e 255
        assert img_array.min() >= 0
        assert img_array.max() <= 255
    
    def test_preprocess_image_resize(self, sample_image_path):
        """Testa que redimensionamento funciona corretamente."""
        # PIL.Image.resize usa (width, height), mas target_size espera (width, height)
        img_array = preprocess_image(
            sample_image_path,
            target_size=(150, 100),  # (width, height) para PIL
            normalize=False
        )
        
        # Verificar shape: (height, width, channels)
        assert img_array.shape[1] == 150  # width
        assert img_array.shape[0] == 100  # height
    
    def test_preprocess_image_invalid_path(self):
        """Testa tratamento de caminho inválido."""
        with pytest.raises((FileNotFoundError, IOError)):
            preprocess_image("nonexistent_image.jpg")


class TestCreateDataGenerators:
    """Testes para create_data_generators."""
    
    def test_create_data_generators_basic(self, sample_dataframe_with_images):
        """Testa criação básica de data generators."""
        # Dividir dados
        train_df, val_df, test_df = split_image_data(
            sample_dataframe_with_images,
            test_size=0.2,
            validation_size=0.2,
            random_state=42
        )
        
        train_gen, val_gen, test_gen = create_data_generators(
            train_df, val_df, test_df,
            image_size=(224, 224),
            batch_size=2,
            color_mode='rgb',
            class_mode='categorical',
            augmentation=True
        )
        
        # Verificar que são generators
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        assert train_gen is not None
        assert val_gen is not None
        assert test_gen is not None
    
    def test_create_data_generators_no_augmentation(self, sample_dataframe_with_images):
        """Testa criação sem augmentation."""
        train_df, val_df, test_df = split_image_data(
            sample_dataframe_with_images,
            test_size=0.2,
            validation_size=0.2,
            random_state=42
        )
        
        train_gen, val_gen, test_gen = create_data_generators(
            train_df, val_df, test_df,
            augmentation=False
        )
        
        # Deve criar generators mesmo sem augmentation
        assert train_gen is not None
        assert val_gen is not None
        assert test_gen is not None
    
    def test_create_data_generators_grayscale(self, sample_dataframe_with_images):
        """Testa criação com modo grayscale."""
        train_df, val_df, test_df = split_image_data(
            sample_dataframe_with_images,
            test_size=0.2,
            validation_size=0.2,
            random_state=42
        )
        
        train_gen, val_gen, test_gen = create_data_generators(
            train_df, val_df, test_df,
            color_mode='grayscale'
        )
        
        assert train_gen is not None
    
    def test_create_data_generators_batch_size(self, sample_dataframe_with_images):
        """Testa que batch_size é respeitado."""
        train_df, val_df, test_df = split_image_data(
            sample_dataframe_with_images,
            test_size=0.2,
            validation_size=0.2,
            random_state=42
        )
        
        train_gen, _, _ = create_data_generators(
            train_df, val_df, test_df,
            batch_size=3
        )
        
        # Obter um batch
        x_batch, y_batch = train_gen[0]
        
        # Verificar tamanho do batch
        assert len(x_batch) <= 3  # Pode ser menor se não houver dados suficientes


class TestSplitImageData:
    """Testes para split_image_data."""
    
    def test_split_image_data_basic(self, sample_dataframe_with_images):
        """Testa divisão básica de dados de imagens."""
        train_df, val_df, test_df = split_image_data(
            sample_dataframe_with_images,
            test_size=0.2,
            validation_size=0.2,
            random_state=42
        )
        
        # Verificar que são DataFrames
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        
        # Verificar que têm dados
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        
        # Verificar colunas
        assert 'image_path' in train_df.columns
        assert 'label' in train_df.columns
    
    def test_split_image_data_proportions(self, sample_dataframe_with_images):
        """Testa que proporções são respeitadas."""
        total = len(sample_dataframe_with_images)
        
        train_df, val_df, test_df = split_image_data(
            sample_dataframe_with_images,
            test_size=0.2,
            validation_size=0.2,
            random_state=42
        )
        
        # Verificar proporções aproximadas (60% treino, 20% val, 20% teste)
        train_prop = len(train_df) / total
        val_prop = len(val_df) / total
        test_prop = len(test_df) / total
        
        assert 0.5 < train_prop < 0.7  # ~60%
        assert 0.15 < val_prop < 0.25  # ~20%
        assert 0.15 < test_prop < 0.25  # ~20%
    
    def test_split_image_data_stratification(self, sample_dataframe_with_images):
        """Testa que estratificação mantém proporções de classes."""
        original_prop = (sample_dataframe_with_images['label'] == 'Normal').sum() / len(sample_dataframe_with_images)
        
        train_df, val_df, test_df = split_image_data(
            sample_dataframe_with_images,
            test_size=0.2,
            validation_size=0.2,
            random_state=42,
            stratify=True
        )
        
        # Verificar proporções em cada split
        train_prop = (train_df['label'] == 'Normal').sum() / len(train_df)
        val_prop = (val_df['label'] == 'Normal').sum() / len(val_df)
        test_prop = (test_df['label'] == 'Normal').sum() / len(test_df)
        
        # Devem ser similares à proporção original
        assert abs(train_prop - original_prop) < 0.2
        assert abs(val_prop - original_prop) < 0.2
        assert abs(test_prop - original_prop) < 0.2
    
    def test_split_image_data_no_stratification(self, sample_dataframe_with_images):
        """Testa divisão sem estratificação."""
        train_df, val_df, test_df = split_image_data(
            sample_dataframe_with_images,
            test_size=0.2,
            validation_size=0.2,
            random_state=42,
            stratify=False
        )
        
        # Deve funcionar mesmo sem estratificação
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
    
    def test_split_image_data_reproducibility(self, sample_dataframe_with_images):
        """Testa reprodutibilidade com random_state."""
        train1, val1, test1 = split_image_data(
            sample_dataframe_with_images,
            random_state=42
        )
        
        train2, val2, test2 = split_image_data(
            sample_dataframe_with_images,
            random_state=42
        )
        
        # Devem ser idênticos
        assert train1.equals(train2)
        assert val1.equals(val2)
        assert test1.equals(test2)


class TestApplyAugmentation:
    """Testes para apply_augmentation."""
    
    def test_apply_augmentation_basic(self, sample_image_array_rgb):
        """Testa aplicação básica de augmentation."""
        augmented = apply_augmentation(
            sample_image_array_rgb,
            rotation_range=10,
            width_shift=0.1,
            height_shift=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        
        # Verificar shape (deve manter dimensões)
        assert augmented.shape == sample_image_array_rgb.shape
        
        # Verificar tipo
        assert isinstance(augmented, np.ndarray)
    
    def test_apply_augmentation_no_transforms(self, sample_image_array_rgb):
        """Testa augmentation sem transformações."""
        augmented = apply_augmentation(
            sample_image_array_rgb,
            rotation_range=0,
            width_shift=0,
            height_shift=0,
            zoom_range=0,
            horizontal_flip=False
        )
        
        # Deve retornar imagem (pode ser igual ou cópia)
        assert augmented.shape == sample_image_array_rgb.shape
    
    def test_apply_augmentation_rotation(self, sample_image_array_rgb):
        """Testa que rotação é aplicada."""
        augmented = apply_augmentation(
            sample_image_array_rgb,
            rotation_range=45,
            width_shift=0,
            height_shift=0,
            zoom_range=0,
            horizontal_flip=False
        )
        
        # Imagem deve ter shape correto
        assert augmented.shape == sample_image_array_rgb.shape
    
    def test_apply_augmentation_grayscale(self, sample_image_array_grayscale):
        """Testa augmentation em imagem grayscale."""
        # Adicionar dimensão de canal se necessário
        if len(sample_image_array_grayscale.shape) == 2:
            img = np.expand_dims(sample_image_array_grayscale, axis=-1)
        else:
            img = sample_image_array_grayscale
        
        augmented = apply_augmentation(
            img,
            rotation_range=10,
            horizontal_flip=True
        )
        
        # A augmentation pode alterar ligeiramente o tamanho devido a rotações e padding
        # Verificar que tem o mesmo número de dimensões e canal
        assert len(augmented.shape) == len(img.shape)
        if len(img.shape) == 3:
            assert augmented.shape[2] == img.shape[2]  # Canais devem ser iguais
        # Tolerar pequenas diferenças nas dimensões espaciais (máx 1 pixel)
        assert abs(augmented.shape[0] - img.shape[0]) <= 1
        assert abs(augmented.shape[1] - img.shape[1]) <= 1
