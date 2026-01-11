"""
Testes unitários para src/vision/evaluation.py
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tensorflow as tf
from tensorflow import keras

from src.vision.evaluation import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    visualize_predictions,
    grad_cam_visualization,
    plot_grad_cam,
    evaluate_model
)


class TestPlotTrainingHistory:
    """Testes para plot_training_history."""
    
    def test_plot_training_history_creates_figure(self, trained_cnn_model):
        """Testa que a função cria uma figura."""
        # Criar histórico mock
        history = MagicMock()
        history.history = {
            'loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4],
            'accuracy': [0.7, 0.8, 0.9],
            'val_accuracy': [0.65, 0.75, 0.85]
        }
        
        with patch('matplotlib.pyplot.show'):
            plot_training_history(history)
        
        # Se chegou aqui sem erro, funcionou
        assert True
    
    def test_plot_training_history_keys(self):
        """Testa que funciona com chaves corretas do histórico."""
        history = MagicMock()
        history.history = {
            'loss': [0.5, 0.4],
            'val_loss': [0.6, 0.5],
            'accuracy': [0.7, 0.8],
            'val_accuracy': [0.65, 0.75]
        }
        
        with patch('matplotlib.pyplot.show'):
            plot_training_history(history)


class TestPlotConfusionMatrix:
    """Testes para plot_confusion_matrix."""
    
    def test_plot_confusion_matrix_basic(self):
        """Testa plot básico de matriz de confusão."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        class_names = ['Class0', 'Class1']
        
        with patch('matplotlib.pyplot.show'):
            plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Se chegou aqui, funcionou
        assert True
    
    def test_plot_confusion_matrix_normalized(self):
        """Testa plot normalizado."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        class_names = ['Class0', 'Class1']
        
        with patch('matplotlib.pyplot.show'):
            plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)


class TestPlotROCCurve:
    """Testes para plot_roc_curve."""
    
    def test_plot_roc_curve_basic(self):
        """Testa plot básico de curva ROC."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred_proba = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3]
        ])
        class_names = ['Class0', 'Class1']
        
        with patch('matplotlib.pyplot.show'):
            plot_roc_curve(y_true, y_pred_proba, class_names)
    
    def test_plot_roc_curve_one_hot(self):
        """Testa plot com y_true one-hot encoded."""
        y_true = np.array([
            [1, 0],
            [0, 1],
            [1, 0]
        ])
        y_pred_proba = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2]
        ])
        class_names = ['Class0', 'Class1']
        
        with patch('matplotlib.pyplot.show'):
            plot_roc_curve(y_true, y_pred_proba, class_names)


class TestVisualizePredictions:
    """Testes para visualize_predictions."""
    
    def test_visualize_predictions_creates_grid(self, sample_dataset_structure):
        """Testa que visualização cria grid de imagens."""
        from src.vision.preprocessing import create_data_generators, split_image_data
        from src.vision.data_loader import load_image_dataset
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Carregar dados
        base_path = sample_dataset_structure['base_path']
        df = load_image_dataset(base_path)
        
        train_df, val_df, test_df = split_image_data(df, random_state=42)
        
        # Criar modelo compatível com o tamanho das imagens (64x64)
        model = keras.Sequential([
            layers.Conv2D(4, (3, 3), activation='relu', input_shape=(64, 64, 3), padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        train_gen, _, test_gen = create_data_generators(
            train_df, val_df, test_df,
            image_size=(64, 64),
            batch_size=2
        )
        
        # Treinar modelo rapidamente
        model.fit(train_gen, epochs=1, verbose=0)
        
        class_names = ['Normal', 'Pneumonia']
        
        with patch('matplotlib.pyplot.show'):
            # Usar no máximo o número de amostras disponíveis no primeiro batch
            # test_gen[0] retorna um batch, verificar quantas amostras há
            x_batch, _ = test_gen[0]
            max_samples = min(4, len(x_batch))
            
            visualize_predictions(
                model,
                test_gen,
                class_names,
                num_samples=max_samples
            )


class TestGradCAMVisualization:
    """Testes para grad_cam_visualization."""
    
    def test_grad_cam_visualization_sequential(self, trained_cnn_model):
        """Testa Grad-CAM com modelo Sequential."""
        # Criar imagem de teste
        img_array = np.random.random((32, 32, 3)).astype(np.float32)
        
        heatmap, superimposed = grad_cam_visualization(
            trained_cnn_model,
            img_array
        )
        
        # Verificar que retorna tupla
        assert isinstance(heatmap, np.ndarray)
        assert isinstance(superimposed, np.ndarray)
        
        # Verificar shapes
        assert len(heatmap.shape) == 2  # 2D heatmap
        assert len(superimposed.shape) == 3  # RGB image
    
    def test_grad_cam_visualization_grayscale(self):
        """Testa Grad-CAM com imagem em escala de cinza."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Criar modelo muito simples para grayscale que não tenha muitos poolings
        # 128x128 para evitar problemas com dimensões negativas
        model = keras.Sequential([
            layers.Conv2D(4, (3, 3), activation='relu', input_shape=(128, 128, 1), padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Treinar rapidamente
        X_train = np.random.random((5, 128, 128, 1))
        y_train = np.random.randint(0, 2, 5)
        y_train = tf.keras.utils.to_categorical(y_train, 2)
        model.fit(X_train, y_train, epochs=1, verbose=0)
        
        # Criar imagem grayscale
        img_array = np.random.random((128, 128, 1)).astype(np.float32)
        
        heatmap, superimposed = grad_cam_visualization(model, img_array)
        
        assert heatmap is not None
        assert superimposed is not None
    
    def test_grad_cam_visualization_layer_name(self, trained_cnn_model):
        """Testa Grad-CAM com layer_name especificado."""
        img_array = np.random.random((32, 32, 3)).astype(np.float32)
        
        # Encontrar nome de uma camada convolucional
        layer_name = None
        for layer in trained_cnn_model.layers:
            if 'conv2d' in layer.name.lower():
                layer_name = layer.name
                break
        
        if layer_name:
            heatmap, superimposed = grad_cam_visualization(
                trained_cnn_model,
                img_array,
                layer_name=layer_name
            )
            
            assert heatmap is not None
            assert superimposed is not None
    
    def test_grad_cam_visualization_pred_index(self, trained_cnn_model):
        """Testa Grad-CAM com pred_index especificado."""
        # Usar tamanho compatível com o modelo treinado
        img_array = np.random.random((32, 32, 3)).astype(np.float32)
        
        heatmap, superimposed = grad_cam_visualization(
            trained_cnn_model,
            img_array,
            pred_index=0
        )
        
        assert heatmap is not None
        assert superimposed is not None


class TestPlotGradCAM:
    """Testes para plot_grad_cam."""
    
    def test_plot_grad_cam_creates_visualization(self, trained_cnn_model):
        """Testa que plot_grad_cam cria visualização completa."""
        # Usar tamanho compatível com o modelo treinado
        img_array = np.random.random((32, 32, 3)).astype(np.float32)
        class_names = ['Class0', 'Class1']
        
        with patch('matplotlib.pyplot.show'):
            plot_grad_cam(trained_cnn_model, img_array, class_names)
        
        # Se chegou aqui, funcionou
        assert True


class TestEvaluateModel:
    """Testes para evaluate_model."""
    
    def test_evaluate_model_returns_metrics(self, trained_cnn_model, sample_dataset_structure):
        """Testa que evaluate_model retorna métricas."""
        from src.vision.preprocessing import create_data_generators, split_image_data
        from src.vision.data_loader import load_image_dataset
        
        # Carregar dados
        base_path = sample_dataset_structure['base_path']
        df = load_image_dataset(base_path)
        
        train_df, val_df, test_df = split_image_data(df, random_state=42)
        
        _, _, test_gen = create_data_generators(
            train_df, val_df, test_df,
            image_size=(32, 32),
            batch_size=2
        )
        
        class_names = ['Normal', 'Pneumonia']
        
        metrics = evaluate_model(
            trained_cnn_model,
            test_gen,
            class_names,
            verbose=False
        )
        
        # Verificar estrutura do dicionário
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'classification_report' in metrics
        assert 'roc_auc' in metrics
        assert 'confusion_matrix' in metrics
    
    def test_evaluate_model_metrics_values(self, trained_cnn_model, sample_dataset_structure):
        """Testa que métricas têm valores válidos."""
        from src.vision.preprocessing import create_data_generators, split_image_data
        from src.vision.data_loader import load_image_dataset
        
        base_path = sample_dataset_structure['base_path']
        df = load_image_dataset(base_path)
        
        train_df, val_df, test_df = split_image_data(df, random_state=42)
        
        _, _, test_gen = create_data_generators(
            train_df, val_df, test_df,
            image_size=(32, 32),
            batch_size=2
        )
        
        class_names = ['Normal', 'Pneumonia']
        
        metrics = evaluate_model(
            trained_cnn_model,
            test_gen,
            class_names,
            verbose=False
        )
        
        # Verificar valores
        assert metrics['loss'] >= 0
        assert 0 <= metrics['accuracy'] <= 1
        assert isinstance(metrics['roc_auc'], dict)
        assert len(metrics['roc_auc']) == len(class_names)
        
        # Verificar AUC por classe
        for class_name in class_names:
            assert 0 <= metrics['roc_auc'][class_name] <= 1
    
    def test_evaluate_model_verbose(self, trained_cnn_model, sample_dataset_structure):
        """Testa que verbose=True imprime métricas."""
        from src.vision.preprocessing import create_data_generators, split_image_data
        from src.vision.data_loader import load_image_dataset
        
        base_path = sample_dataset_structure['base_path']
        df = load_image_dataset(base_path)
        
        train_df, val_df, test_df = split_image_data(df, random_state=42)
        
        _, _, test_gen = create_data_generators(
            train_df, val_df, test_df,
            image_size=(32, 32),
            batch_size=2
        )
        
        class_names = ['Normal', 'Pneumonia']
        
        # Deve imprimir sem erro
        metrics = evaluate_model(
            trained_cnn_model,
            test_gen,
            class_names,
            verbose=True
        )
        
        assert metrics is not None
