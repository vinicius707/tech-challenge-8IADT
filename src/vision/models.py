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


def create_improved_cnn_breast_cancer(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    num_classes: int = 2,
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Cria uma CNN melhorada para classificação de câncer de mama.
    Usa Global Average Pooling, melhor regularização e arquitetura otimizada.
    
    Parameters:
    -----------
    input_shape : tuple
        Formato de entrada (height, width, channels). Geralmente escala de cinza.
    num_classes : int
        Número de classes de saída.
    dropout_rate : float
        Taxa de dropout para regularização nas camadas densas.
    
    Returns:
    --------
    keras.Model
        Modelo CNN compilado.
    """
    model = keras.Sequential([
        # Primeiro bloco convolucional - menos dropout inicial
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape,
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),  # Reduzido de 0.25
        
        # Segundo bloco convolucional
        layers.Conv2D(64, (5, 5), activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.15),  # Reduzido de 0.25
        
        # Terceiro bloco convolucional
        layers.Conv2D(128, (3, 3), activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),  # Reduzido de 0.25
        
        # Quarto bloco convolucional
        layers.Conv2D(256, (3, 3), activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Quinto bloco convolucional
        layers.Conv2D(512, (3, 3), activation='relu',
                     kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling em vez de Flatten (reduz overfitting)
        layers.GlobalAveragePooling2D(),
        
        # Camadas densas com dropout mais agressivo
        layers.Dense(512, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(256, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Camada de saída
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def focal_loss_fn(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal Loss para lidar com classes desbalanceadas.
    Foca no aprendizado de exemplos difíceis.
    
    Parameters:
    -----------
    y_true : tensor
        Labels verdadeiros.
    y_pred : tensor
        Predições do modelo.
    gamma : float
        Parâmetro de foco (quanto maior, mais foco em exemplos difíceis).
    alpha : float
        Parâmetro de balanceamento de classes.
    
    Returns:
    --------
    tensor
        Valor da loss focal.
    """
    # Evitar log(0)
    epsilon = keras.backend.epsilon()
    y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Calcular cross entropy
    cross_entropy = -y_true * keras.backend.log(y_pred)
    
    # Calcular peso focal
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_weight = alpha * keras.backend.pow((1 - p_t), gamma)
    
    # Aplicar peso focal
    focal_loss = focal_weight * cross_entropy
    
    return keras.backend.mean(keras.backend.sum(focal_loss, axis=1))


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Wrapper para criar função de Focal Loss com parâmetros fixos.
    
    Parameters:
    -----------
    gamma : float
        Parâmetro de foco.
    alpha : float
        Parâmetro de balanceamento.
    
    Returns:
    --------
    function
        Função de loss focal com parâmetros fixos.
    """
    def loss_fn(y_true, y_pred):
        return focal_loss_fn(y_true, y_pred, gamma=gamma, alpha=alpha)
    
    # Adicionar atributo para serialização
    loss_fn.__name__ = 'focal_loss'
    return loss_fn


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.0001,
    optimizer: Optional[str] = None,
    weight_decay: float = 1e-4,
    use_focal_loss: bool = False
) -> keras.Model:
    """
    Compila o modelo com otimizador e métricas.
    Usa AdamW por padrão com weight decay para melhor regularização.
    
    Parameters:
    -----------
    model : keras.Model
        Modelo a ser compilado.
    learning_rate : float
        Taxa de aprendizado (padrão reduzido para 0.0001).
    optimizer : str, optional
        Nome do otimizador ('adamw', 'adam', 'sgd', etc.). Se None, usa AdamW.
    weight_decay : float
        Weight decay para regularização (usado com AdamW).
    use_focal_loss : bool
        Se True, usa Focal Loss em vez de categorical_crossentropy.
    
    Returns:
    --------
    keras.Model
        Modelo compilado.
    """
    if optimizer is None or optimizer == 'adamw':
        # AdamW com weight decay para melhor regularização
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    elif optimizer == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True
        )
    
    # Escolher função de loss
    if use_focal_loss:
        # Usar a função diretamente com parâmetros fixos para melhor serialização
        loss_fn = lambda y_true, y_pred: focal_loss_fn(y_true, y_pred, gamma=2.0, alpha=0.25)
        # Dar um nome à função para serialização
        loss_fn.__name__ = 'focal_loss'
    else:
        loss_fn = 'categorical_crossentropy'
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def get_model_callbacks(
    checkpoint_path: str,
    patience: int = 10,
    monitor: str = 'val_loss',
    learning_rate: float = 0.0001,
    use_tensorboard: bool = False,
    log_dir: Optional[str] = None
) -> list:
    """
    Cria callbacks para treinamento do modelo.
    Melhorado com callbacks mais pacientes e opção de TensorBoard.
    
    Parameters:
    -----------
    checkpoint_path : str
        Caminho para salvar o melhor modelo.
    patience : int
        Paciência para early stopping (aumentado para 10 por padrão).
    monitor : str
        Métrica a ser monitorada.
    learning_rate : float
        Learning rate inicial para o scheduler.
    use_tensorboard : bool
        Se True, adiciona callback do TensorBoard.
    log_dir : str, optional
        Diretório para logs do TensorBoard.
    
    Returns:
    --------
    list
        Lista de callbacks.
    """
    callbacks = [
        # ModelCheckpoint melhorado - salva baseado em val_loss e val_accuracy
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='auto'
        ),
        # Early stopping mais paciente
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='auto'
        ),
        # ReduceLROnPlateau mais paciente
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,  # Aumentado de patience//2 para 5
            min_lr=1e-7,
            verbose=1,
            mode='auto'
        )
    ]
    
    # Adicionar TensorBoard se solicitado
    if use_tensorboard and log_dir:
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        )
    
    return callbacks


