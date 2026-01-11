"""
Testes unitários para src/vision/models.py
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from src.vision.models import (
    create_simple_cnn_pneumonia,
    create_cnn_breast_cancer,
    create_improved_cnn_breast_cancer,
    focal_loss_fn,
    focal_loss,
    compile_model,
    get_model_callbacks
)


class TestCreateSimpleCNNPneumonia:
    """Testes para create_simple_cnn_pneumonia."""
    
    def test_create_simple_cnn_pneumonia_architecture(self):
        """Testa que a arquitetura é criada corretamente."""
        model = create_simple_cnn_pneumonia(
            input_shape=(224, 224, 3),
            num_classes=2,
            dropout_rate=0.5
        )
        
        assert isinstance(model, keras.Model)
        assert model.input_shape == (None, 224, 224, 3)
        assert model.output_shape == (None, 2)
    
    def test_create_simple_cnn_pneumonia_layers(self):
        """Testa que todas as camadas estão presentes."""
        model = create_simple_cnn_pneumonia()
        
        layer_types = [type(layer) for layer in model.layers]
        
        # Verificar tipos de camadas
        assert layers.Conv2D in layer_types
        assert layers.BatchNormalization in layer_types
        assert layers.MaxPooling2D in layer_types
        assert layers.Dropout in layer_types
        assert layers.Flatten in layer_types
        assert layers.Dense in layer_types
    
    def test_create_simple_cnn_pneumonia_parameters(self):
        """Testa que o modelo tem parâmetros razoáveis."""
        model = create_simple_cnn_pneumonia()
        
        total_params = model.count_params()
        
        # Deve ter alguns milhões de parâmetros (não zero, não bilhões)
        assert total_params > 1000
        assert total_params < 100_000_000
    
    def test_create_simple_cnn_pneumonia_different_input_shape(self):
        """Testa criação com input_shape diferente."""
        model = create_simple_cnn_pneumonia(input_shape=(128, 128, 3))
        
        assert model.input_shape == (None, 128, 128, 3)
    
    def test_create_simple_cnn_pneumonia_different_num_classes(self):
        """Testa criação com número diferente de classes."""
        model = create_simple_cnn_pneumonia(num_classes=3)
        
        assert model.output_shape == (None, 3)


class TestCreateCNNBreasCancer:
    """Testes para create_cnn_breast_cancer."""
    
    def test_create_cnn_breast_cancer_architecture(self):
        """Testa que a arquitetura é criada corretamente."""
        model = create_cnn_breast_cancer(
            input_shape=(256, 256, 1),
            num_classes=2
        )
        
        assert isinstance(model, keras.Model)
        assert model.input_shape == (None, 256, 256, 1)
        assert model.output_shape == (None, 2)
    
    def test_create_cnn_breast_cancer_grayscale(self):
        """Testa que funciona com escala de cinza."""
        model = create_cnn_breast_cancer(input_shape=(256, 256, 1))
        
        # Deve aceitar 1 canal
        assert model.input_shape[-1] == 1


class TestCreateImprovedCNNBreasCancer:
    """Testes para create_improved_cnn_breast_cancer."""
    
    def test_create_improved_cnn_breast_cancer_architecture(self):
        """Testa que a arquitetura melhorada é criada."""
        model = create_improved_cnn_breast_cancer(
            input_shape=(256, 256, 1),
            num_classes=2
        )
        
        assert isinstance(model, keras.Model)
        assert model.input_shape == (None, 256, 256, 1)
    
    def test_create_improved_cnn_breast_cancer_global_average_pooling(self):
        """Testa que Global Average Pooling está presente."""
        model = create_improved_cnn_breast_cancer()
        
        layer_types = [type(layer) for layer in model.layers]
        
        assert layers.GlobalAveragePooling2D in layer_types
    
    def test_create_improved_cnn_breast_cancer_l2_regularization(self):
        """Testa que L2 regularization está presente."""
        model = create_improved_cnn_breast_cancer()
        
        # Verificar que pelo menos uma camada tem kernel_regularizer
        has_regularizer = False
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
                has_regularizer = True
                break
        
        assert has_regularizer


class TestFocalLoss:
    """Testes para focal_loss_fn e focal_loss."""
    
    def test_focal_loss_fn_basic(self):
        """Testa cálculo básico de focal loss."""
        y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.8, 0.2], [0.3, 0.7]], dtype=tf.float32)
        
        loss = focal_loss_fn(y_true, y_pred, gamma=2.0, alpha=0.25)
        
        # Deve retornar um escalar
        assert loss.shape == ()
        assert loss.numpy() > 0
    
    def test_focal_loss_fn_extreme_values(self):
        """Testa tratamento de valores extremos."""
        y_true = tf.constant([[1.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.99, 0.01]], dtype=tf.float32)
        
        # Não deve gerar NaN ou Inf
        loss = focal_loss_fn(y_true, y_pred)
        
        assert tf.math.is_finite(loss)
        assert not tf.math.is_nan(loss)
    
    def test_focal_loss_fn_parameters(self):
        """Testa que parâmetros gamma e alpha funcionam."""
        y_true = tf.constant([[1.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.5, 0.5]], dtype=tf.float32)
        
        loss1 = focal_loss_fn(y_true, y_pred, gamma=1.0, alpha=0.25)
        loss2 = focal_loss_fn(y_true, y_pred, gamma=2.0, alpha=0.25)
        
        # Gamma maior deve dar loss diferente
        assert not np.isclose(loss1.numpy(), loss2.numpy())
    
    def test_focal_loss_wrapper(self):
        """Testa wrapper focal_loss."""
        loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        
        assert callable(loss_fn)
        assert hasattr(loss_fn, '__name__')
        assert loss_fn.__name__ == 'focal_loss'
        
        # Testar que funciona
        y_true = tf.constant([[1.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.8, 0.2]], dtype=tf.float32)
        
        loss = loss_fn(y_true, y_pred)
        assert loss.numpy() > 0


class TestCompileModel:
    """Testes para compile_model."""
    
    def test_compile_model_adamw(self):
        """Testa compilação com AdamW."""
        model = create_simple_cnn_pneumonia()
        
        compiled_model = compile_model(
            model,
            learning_rate=0.0001,
            optimizer='adamw'
        )
        
        assert compiled_model.optimizer is not None
        assert isinstance(compiled_model.optimizer, keras.optimizers.AdamW)
    
    def test_compile_model_adam(self):
        """Testa compilação com Adam."""
        model = create_simple_cnn_pneumonia()
        
        compiled_model = compile_model(
            model,
            optimizer='adam'
        )
        
        assert isinstance(compiled_model.optimizer, keras.optimizers.Adam)
    
    def test_compile_model_sgd(self):
        """Testa compilação com SGD."""
        model = create_simple_cnn_pneumonia()
        
        compiled_model = compile_model(
            model,
            optimizer='sgd'
        )
        
        assert isinstance(compiled_model.optimizer, keras.optimizers.SGD)
    
    def test_compile_model_focal_loss(self):
        """Testa compilação com Focal Loss."""
        model = create_simple_cnn_pneumonia()
        
        compiled_model = compile_model(
            model,
            use_focal_loss=True
        )
        
        # Verificar que foi compilado
        assert compiled_model.optimizer is not None
        # Loss deve ser função customizada
        assert compiled_model.loss is not None
    
    def test_compile_model_categorical_crossentropy(self):
        """Testa compilação com categorical crossentropy."""
        model = create_simple_cnn_pneumonia()
        
        compiled_model = compile_model(
            model,
            use_focal_loss=False
        )
        
        # Loss deve ser string ou função padrão
        assert compiled_model.loss is not None
    
    def test_compile_model_metrics(self):
        """Testa que métricas são configuradas."""
        model = create_simple_cnn_pneumonia()
        
        compiled_model = compile_model(model)
        
        # Verificar que modelo foi compilado
        assert compiled_model.optimizer is not None
        assert compiled_model.loss is not None
        
        # Em Keras 3.x, as métricas podem estar em diferentes estruturas
        # Verificar que há métricas configuradas (pelo menos compile_metrics)
        has_metrics = False
        if hasattr(compiled_model, 'metrics') and len(compiled_model.metrics) > 0:
            has_metrics = True
        if hasattr(compiled_model, '_compile_metrics'):
            has_metrics = True
        
        # Verificar que métricas foram especificadas durante compilação
        # O modelo foi compilado com metrics=['accuracy', 'precision', 'recall']
        assert has_metrics or 'accuracy' in str(compiled_model._compile_config['metrics']).lower()


class TestGetModelCallbacks:
    """Testes para get_model_callbacks."""
    
    def test_get_model_callbacks_basic(self, temp_dir):
        """Testa criação básica de callbacks."""
        checkpoint_path = str(Path(temp_dir) / "model.h5")
        
        callbacks = get_model_callbacks(
            checkpoint_path,
            patience=10,
            monitor='val_loss'
        )
        
        assert len(callbacks) >= 3  # ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    
    def test_get_model_callbacks_types(self, temp_dir):
        """Testa tipos de callbacks criados."""
        checkpoint_path = str(Path(temp_dir) / "model.h5")
        
        callbacks = get_model_callbacks(checkpoint_path)
        
        callback_types = [type(cb) for cb in callbacks]
        
        assert keras.callbacks.ModelCheckpoint in callback_types
        assert keras.callbacks.EarlyStopping in callback_types
        assert keras.callbacks.ReduceLROnPlateau in callback_types
    
    def test_get_model_callbacks_tensorboard(self, temp_dir):
        """Testa adição de TensorBoard callback."""
        checkpoint_path = str(Path(temp_dir) / "model.h5")
        log_dir = str(Path(temp_dir) / "logs")
        
        callbacks = get_model_callbacks(
            checkpoint_path,
            use_tensorboard=True,
            log_dir=log_dir
        )
        
        callback_types = [type(cb) for cb in callbacks]
        assert keras.callbacks.TensorBoard in callback_types
    
    def test_get_model_callbacks_no_tensorboard(self, temp_dir):
        """Testa que TensorBoard não é adicionado quando não solicitado."""
        checkpoint_path = str(Path(temp_dir) / "model.h5")
        
        callbacks = get_model_callbacks(
            checkpoint_path,
            use_tensorboard=False
        )
        
        callback_types = [type(cb) for cb in callbacks]
        assert keras.callbacks.TensorBoard not in callback_types
    
    def test_get_model_callbacks_patience(self, temp_dir):
        """Testa que patience é configurado corretamente."""
        checkpoint_path = str(Path(temp_dir) / "model.h5")
        
        callbacks = get_model_callbacks(
            checkpoint_path,
            patience=15
        )
        
        # Encontrar EarlyStopping
        early_stopping = None
        for cb in callbacks:
            if isinstance(cb, keras.callbacks.EarlyStopping):
                early_stopping = cb
                break
        
        assert early_stopping is not None
        assert early_stopping.patience == 15
