"""
Módulo com arquiteturas de CNNs para classificação de imagens médicas.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


def create_simple_cnn_pneumonia(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Cria uma CNN simples para classificação de pneumonia em raio-X.
    
    Parameters:
    -----------
    input_shape : tuple
        Formato de entrada (height, width, channels).
    num_classes : int
        Número de classes de saída.
    dropout_rate : float
        Taxa de dropout para regularização.
    
    Returns:
    --------
    keras.Model
        Modelo CNN compilado.
    """
    model = keras.Sequential([
        # Primeiro bloco convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segundo bloco convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Terceiro bloco convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Quarto bloco convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten e camadas densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Camada de saída
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_cnn_breast_cancer(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    num_classes: int = 2,
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Cria uma CNN para classificação de câncer de mama em imagens de mamografia.
    
    Parameters:
    -----------
    input_shape : tuple
        Formato de entrada (height, width, channels). Geralmente escala de cinza.
    num_classes : int
        Número de classes de saída.
    dropout_rate : float
        Taxa de dropout para regularização.
    
    Returns:
    --------
    keras.Model
        Modelo CNN compilado.
    """
    model = keras.Sequential([
        # Primeiro bloco convolucional
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segundo bloco convolucional
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Terceiro bloco convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Quarto bloco convolucional
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Quinto bloco convolucional
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten e camadas densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Camada de saída
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_improved_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Cria uma CNN melhorada com mais camadas e melhor arquitetura.
    
    Parameters:
    -----------
    input_shape : tuple
        Formato de entrada (height, width, channels).
    num_classes : int
        Número de classes de saída.
    dropout_rate : float
        Taxa de dropout para regularização.
    
    Returns:
    --------
    keras.Model
        Modelo CNN compilado.
    """
    model = keras.Sequential([
        # Bloco 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten e camadas densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Camada de saída
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: Optional[str] = None
) -> keras.Model:
    """
    Compila o modelo com otimizador e métricas.
    
    Parameters:
    -----------
    model : keras.Model
        Modelo a ser compilado.
    learning_rate : float
        Taxa de aprendizado.
    optimizer : str, optional
        Nome do otimizador ('adam', 'sgd', etc.). Se None, usa Adam.
    
    Returns:
    --------
    keras.Model
        Modelo compilado.
    """
    if optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def get_model_callbacks(
    checkpoint_path: str,
    patience: int = 5,
    monitor: str = 'val_loss'
) -> list:
    """
    Cria callbacks para treinamento do modelo.
    
    Parameters:
    -----------
    checkpoint_path : str
        Caminho para salvar o melhor modelo.
    patience : int
        Paciência para early stopping.
    monitor : str
        Métrica a ser monitorada.
    
    Returns:
    --------
    list
        Lista de callbacks.
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks

