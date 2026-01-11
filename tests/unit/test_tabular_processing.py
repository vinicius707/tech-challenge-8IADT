"""
Testes unitários para src/tabular/processing.py
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.tabular.processing import split_data, build_pipeline


class TestSplitData:
    """Testes para a função split_data."""
    
    def test_split_data_basic(self, sample_tabular_data):
        """Testa divisão básica sem validação."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data['diagnosis']
        
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, return_validation=False, random_state=42
        )
        
        # Verificar tamanhos
        assert len(X_train) == 80  # 80% de 100
        assert len(X_test) == 20  # 20% de 100
        assert len(y_train) == 80
        assert len(y_test) == 20
        
        # Verificar que não há sobreposição
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(test_indices)) == 0
    
    def test_split_data_with_validation(self, sample_tabular_data):
        """Testa divisão com conjunto de validação."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data['diagnosis']
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.2, validation_size=0.2, return_validation=True, random_state=42
        )
        
        # Verificar tamanhos (60% treino, 20% val, 20% teste)
        assert len(X_train) == 60
        assert len(X_val) == 20
        assert len(X_test) == 20
        assert len(y_train) == 60
        assert len(y_val) == 20
        assert len(y_test) == 20
        
        # Verificar que não há sobreposição
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(val_indices)) == 0
        assert len(train_indices.intersection(test_indices)) == 0
        assert len(val_indices.intersection(test_indices)) == 0
    
    def test_split_data_stratification(self, sample_tabular_data):
        """Testa que a estratificação mantém proporções de classes."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data['diagnosis']
        
        # Calcular proporção original
        original_prop = (y == 'B').sum() / len(y)
        
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, return_validation=False, random_state=42
        )
        
        # Verificar que proporções são mantidas (com tolerância)
        train_prop = (y_train == 'B').sum() / len(y_train)
        test_prop = (y_test == 'B').sum() / len(y_test)
        
        assert abs(train_prop - original_prop) < 0.1  # Tolerância de 10%
        assert abs(test_prop - original_prop) < 0.1
    
    def test_split_data_imbalanced(self, sample_tabular_data_imbalanced):
        """Testa divisão com dados desbalanceados."""
        X = sample_tabular_data_imbalanced.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data_imbalanced['diagnosis']
        
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, return_validation=False, random_state=42
        )
        
        # Verificar que estratificação ainda funciona
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        # Verificar que ambas as classes estão presentes
        assert 'B' in y_train.values
        assert 'M' in y_train.values
        assert 'B' in y_test.values
        assert 'M' in y_test.values
    
    def test_split_data_small_dataset(self, sample_tabular_data_small):
        """Testa divisão com dataset pequeno."""
        X = sample_tabular_data_small.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data_small['diagnosis']
        
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, return_validation=False, random_state=42
        )
        
        # Verificar que ainda funciona com dataset pequeno
        assert len(X_train) > 0
        assert len(X_test) > 0
    
    def test_split_data_reproducibility(self, sample_tabular_data):
        """Testa que random_state garante reprodutibilidade."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data['diagnosis']
        
        # Primeira divisão
        X_train1, X_test1, y_train1, y_test1 = split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        # Segunda divisão com mesmo random_state
        X_train2, X_test2, y_train2, y_test2 = split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        # Verificar que são idênticos
        assert X_train1.equals(X_train2)
        assert X_test1.equals(X_test2)
        assert y_train1.equals(y_train2)
        assert y_test1.equals(y_test2)
    
    def test_split_data_different_random_state(self, sample_tabular_data):
        """Testa que random_state diferente produz divisões diferentes."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data['diagnosis']
        
        X_train1, X_test1, _, _ = split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train2, X_test2, _, _ = split_data(
            X, y, test_size=0.2, random_state=123
        )
        
        # Verificar que são diferentes
        assert not X_train1.equals(X_train2)


class TestBuildPipeline:
    """Testes para a função build_pipeline."""
    
    def test_build_pipeline_logistic_regression(self):
        """Testa criação de pipeline com Regressão Logística."""
        model = LogisticRegression(random_state=42)
        pipeline = build_pipeline(model)
        
        # Verificar que é um Pipeline
        from sklearn.pipeline import Pipeline
        assert isinstance(pipeline, Pipeline)
        
        # Verificar steps
        assert 'scaler' in pipeline.named_steps
        assert 'model' in pipeline.named_steps
        
        # Verificar tipos
        from sklearn.preprocessing import StandardScaler
        assert isinstance(pipeline.named_steps['scaler'], StandardScaler)
        assert isinstance(pipeline.named_steps['model'], LogisticRegression)
    
    def test_build_pipeline_random_forest(self):
        """Testa criação de pipeline com Random Forest."""
        model = RandomForestClassifier(random_state=42)
        pipeline = build_pipeline(model)
        
        from sklearn.pipeline import Pipeline
        assert isinstance(pipeline, Pipeline)
        
        assert 'scaler' in pipeline.named_steps
        assert 'model' in pipeline.named_steps
        
        from sklearn.preprocessing import StandardScaler
        assert isinstance(pipeline.named_steps['scaler'], StandardScaler)
        assert isinstance(pipeline.named_steps['model'], RandomForestClassifier)
    
    def test_build_pipeline_fit_predict(self, sample_tabular_data):
        """Testa que o pipeline pode ser treinado e fazer predições."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        y = sample_tabular_data['diagnosis']
        
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        pipeline = build_pipeline(model)
        
        # Treinar
        pipeline.fit(X_train, y_train)
        
        # Fazer predições
        predictions = pipeline.predict(X_test)
        
        # Verificar formato
        assert len(predictions) == len(y_test)
        assert all(pred in ['B', 'M'] for pred in predictions)
        
        # Verificar que pode fazer predict_proba
        probabilities = pipeline.predict_proba(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilidades somam 1
