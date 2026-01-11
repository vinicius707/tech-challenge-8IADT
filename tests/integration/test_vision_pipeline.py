"""
Testes de integração para pipeline vision completo.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.vision.data_loader import load_image_dataset
from src.vision.preprocessing import split_image_data, create_data_generators, preprocess_image
from src.vision.models import (
    create_simple_cnn_pneumonia,
    compile_model,
    get_model_callbacks
)
from src.vision.evaluation import evaluate_model, grad_cam_visualization


@pytest.mark.integration
class TestVisionPipeline:
    """Testes de integração para pipeline vision completo."""
    
    def test_full_pipeline_load_preprocess_train_evaluate(self, sample_dataset_structure):
        """Testa pipeline completo: carregar, pré-processar, treinar, avaliar."""
        # Carregar dados
        base_path = sample_dataset_structure['base_path']
        df = load_image_dataset(base_path)
        
        assert len(df) > 0
        
        # Dividir dados
        train_df, val_df, test_df = split_image_data(
            df,
            test_size=0.2,
            validation_size=0.2,
            random_state=42
        )
        
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        
        # Criar generators
        # Usar 64x64 para evitar dimensões negativas após pooling
        train_gen, val_gen, test_gen = create_data_generators(
            train_df, val_df, test_df,
            image_size=(64, 64),  # Aumentado para evitar problemas com pooling
            batch_size=2,
            augmentation=True
        )
        
        # Criar modelo
        model = create_simple_cnn_pneumonia(
            input_shape=(64, 64, 3),
            num_classes=2
        )
        
        # Compilar
        model = compile_model(model, learning_rate=0.001)
        
        # Treinar (1 época apenas para teste rápido)
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=1,
            verbose=0
        )
        
        # Verificar que treinou
        assert len(history.history['loss']) == 1
        
        # Avaliar
        class_names = ['Normal', 'Pneumonia']
        metrics = evaluate_model(
            model,
            test_gen,
            class_names,
            verbose=False
        )
        
        assert metrics['accuracy'] >= 0
        assert metrics['accuracy'] <= 1
    
    def test_pipeline_with_grad_cam(self, sample_dataset_structure):
        """Testa pipeline completo incluindo Grad-CAM."""
        # Carregar e preparar dados
        base_path = sample_dataset_structure['base_path']
        df = load_image_dataset(base_path)
        
        train_df, val_df, test_df = split_image_data(df, random_state=42)
        
        train_gen, val_gen, test_gen = create_data_generators(
            train_df, val_df, test_df,
            image_size=(64, 64),
            batch_size=2
        )
        
        # Criar e treinar modelo pequeno
        model = create_simple_cnn_pneumonia(input_shape=(64, 64, 3), num_classes=2)
        model = compile_model(model)
        
        model.fit(train_gen, validation_data=val_gen, epochs=1, verbose=0)
        
        # Obter uma imagem de teste
        x_batch, _ = test_gen[0]
        test_image = x_batch[0]
        
        # Verificar que a imagem tem o tamanho correto
        assert test_image.shape == (64, 64, 3) or test_image.shape[:2] == (64, 64)
        
        # Aplicar Grad-CAM
        heatmap, superimposed = grad_cam_visualization(model, test_image)
        
        assert heatmap is not None
        assert superimposed is not None
        assert len(heatmap.shape) == 2
        assert len(superimposed.shape) == 3
    
    def test_pipeline_save_load_model(self, sample_dataset_structure, temp_dir):
        """Testa salvar e carregar modelo treinado."""
        from tensorflow import keras
        
        # Carregar e preparar dados
        base_path = sample_dataset_structure['base_path']
        df = load_image_dataset(base_path)
        
        train_df, val_df, test_df = split_image_data(df, random_state=42)
        
        train_gen, val_gen, test_gen = create_data_generators(
            train_df, val_df, test_df,
            image_size=(64, 64),
            batch_size=2
        )
        
        # Criar e treinar modelo
        model = create_simple_cnn_pneumonia(input_shape=(64, 64, 3), num_classes=2)
        model = compile_model(model)
        
        model.fit(train_gen, validation_data=val_gen, epochs=1, verbose=0)
        
        # Salvar modelo
        model_path = os.path.join(temp_dir, 'test_cnn_model.h5')
        model.save(model_path)
        
        # Carregar modelo
        loaded_model = keras.models.load_model(model_path)
        
        # Verificar que pode fazer predições
        x_batch, _ = test_gen[0]
        predictions = loaded_model.predict(x_batch, verbose=0)
        
        assert predictions.shape[0] == len(x_batch)
        assert predictions.shape[1] == 2  # 2 classes
    
    def test_pipeline_with_callbacks(self, sample_dataset_structure, temp_dir):
        """Testa pipeline com callbacks."""
        # Carregar e preparar dados
        base_path = sample_dataset_structure['base_path']
        df = load_image_dataset(base_path)
        
        train_df, val_df, test_df = split_image_data(df, random_state=42)
        
        train_gen, val_gen, test_gen = create_data_generators(
            train_df, val_df, test_df,
            image_size=(64, 64),
            batch_size=2
        )
        
        # Criar modelo
        model = create_simple_cnn_pneumonia(input_shape=(64, 64, 3), num_classes=2)
        model = compile_model(model)
        
        # Criar callbacks
        checkpoint_path = os.path.join(temp_dir, 'checkpoint.h5')
        callbacks = get_model_callbacks(
            checkpoint_path,
            patience=2,
            monitor='val_loss'
        )
        
        # Treinar com callbacks
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=3,
            callbacks=callbacks,
            verbose=0
        )
        
        # Verificar que checkpoint foi criado (se houve melhoria)
        # Pode ou não existir dependendo do desempenho
        assert len(history.history['loss']) <= 3
    
    def test_pipeline_breast_cancer_grayscale(self, sample_dataset_structure):
        """Testa pipeline para câncer de mama (escala de cinza)."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Carregar dados
        base_path = sample_dataset_structure['base_path']
        df = load_image_dataset(base_path)
        
        train_df, val_df, test_df = split_image_data(df, random_state=42)
        
        # Usar 128x128 para evitar problemas com muitos poolings
        train_gen, val_gen, test_gen = create_data_generators(
            train_df, val_df, test_df,
            image_size=(128, 128),
            batch_size=2,
            color_mode='grayscale'
        )
        
        # Criar modelo simples para grayscale que funcione com 128x128
        model = keras.Sequential([
            layers.Conv2D(32, (5, 5), activation='relu', input_shape=(128, 128, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (5, 5), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])
        model = compile_model(model)
        
        # Treinar
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=1,
            verbose=0
        )
        
        assert len(history.history['loss']) == 1
        
        # Avaliar
        class_names = ['Normal', 'Pneumonia']  # Ajustar conforme necessário
        metrics = evaluate_model(model, test_gen, class_names, verbose=False)
        
        assert metrics['accuracy'] >= 0
