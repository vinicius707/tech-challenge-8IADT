"""
Testes de integração para pipeline tabular completo.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
from pathlib import Path

from src.tabular.processing import split_data, build_pipeline
from src.tabular.evaluate import load_model, predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@pytest.mark.integration
class TestTabularPipeline:
    """Testes de integração para pipeline tabular completo."""
    
    def test_full_pipeline_train_save_load_predict(self, sample_tabular_data, temp_dir):
        """Testa pipeline completo: treinar, salvar, carregar, predizer."""
        # Preparar dados
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data['diagnosis']
        
        # Dividir dados
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.2, validation_size=0.2, return_validation=True, random_state=42
        )
        
        # Criar e treinar modelo
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        
        # Avaliar no conjunto de validação
        val_score = pipeline.score(X_val, y_val)
        assert val_score > 0  # Deve ter alguma acurácia
        
        # Salvar modelo
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        joblib.dump(pipeline, model_path)
        
        # Carregar modelo
        loaded_pipeline = load_model(model_path)
        
        # Fazer predições no conjunto de teste
        predictions = predict(loaded_pipeline, X_test)
        
        # Verificar predições
        assert len(predictions) == len(y_test)
        assert all(pred in ['B', 'M'] for pred in predictions)
        
        # Calcular acurácia
        accuracy = (predictions == y_test.values).mean()
        assert accuracy > 0.5  # Deve ser melhor que aleatório
    
    def test_pipeline_with_logistic_regression(self, sample_tabular_data):
        """Testa pipeline completo com Regressão Logística."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data['diagnosis']
        
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        
        # Fazer predições
        predictions = pipeline.predict(X_test)
        
        assert len(predictions) == len(y_test)
        
        # Verificar probabilidades
        probabilities = pipeline.predict_proba(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_pipeline_with_random_forest(self, sample_tabular_data):
        """Testa pipeline completo com Random Forest."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data['diagnosis']
        
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        
        # Fazer predições
        predictions = pipeline.predict(X_test)
        
        assert len(predictions) == len(y_test)
        
        # Verificar feature importance
        feature_importance = pipeline.named_steps['model'].feature_importances_
        assert len(feature_importance) == len(X.columns)
        assert all(imp >= 0 for imp in feature_importance)
    
    def test_pipeline_handles_new_data(self, sample_tabular_data, trained_tabular_model):
        """Testa que pipeline funciona com novos dados."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        
        # Criar novos dados (simulando dados de produção)
        new_data = X.iloc[:3].copy()
        
        # Fazer predições
        predictions = predict(trained_tabular_model, new_data)
        
        assert len(predictions) == 3
        assert all(pred in ['B', 'M'] for pred in predictions)
    
    def test_pipeline_reproducibility(self, sample_tabular_data):
        """Testa que pipeline produz resultados reproduzíveis."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data['diagnosis']
        
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        # Treinar modelo 1
        model1 = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        pipeline1 = build_pipeline(model1)
        pipeline1.fit(X_train, y_train)
        predictions1 = pipeline1.predict(X_test)
        
        # Treinar modelo 2 (mesmos parâmetros)
        model2 = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        pipeline2 = build_pipeline(model2)
        pipeline2.fit(X_train, y_train)
        predictions2 = pipeline2.predict(X_test)
        
        # Predições devem ser idênticas
        assert np.array_equal(predictions1, predictions2)
