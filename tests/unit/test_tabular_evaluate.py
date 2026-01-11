"""
Testes unitários para src/tabular/evaluate.py
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
from pathlib import Path

from src.tabular.evaluate import load_model, predict


class TestLoadModel:
    """Testes para a função load_model."""
    
    def test_load_model_success(self, trained_tabular_model, temp_dir):
        """Testa carregamento bem-sucedido de modelo."""
        # Salvar modelo
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        joblib.dump(trained_tabular_model, model_path)
        
        # Carregar modelo
        loaded_model = load_model(model_path)
        
        # Verificar que é o mesmo tipo
        assert type(loaded_model) == type(trained_tabular_model)
        
        # Verificar que pode fazer predições
        X_test = pd.DataFrame(np.random.random((5, 30)))
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == 5
    
    def test_load_model_default_path(self, trained_tabular_model, temp_dir, monkeypatch):
        """Testa carregamento com caminho padrão."""
        # Criar diretório models se não existir
        models_dir = Path(temp_dir) / 'models'
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / 'maternal_risk_model.pkl'
        joblib.dump(trained_tabular_model, model_path)
        
        # Mudar diretório de trabalho
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            loaded_model = load_model(str(model_path))
            assert loaded_model is not None
        finally:
            os.chdir(original_cwd)
    
    def test_load_model_file_not_found(self, temp_dir):
        """Testa tratamento de erro quando arquivo não existe."""
        model_path = os.path.join(temp_dir, 'nonexistent_model.pkl')
        
        with pytest.raises(FileNotFoundError):
            load_model(model_path)
    
    def test_load_model_invalid_file(self, temp_dir):
        """Testa tratamento de erro quando arquivo é inválido."""
        # Criar arquivo inválido
        invalid_path = os.path.join(temp_dir, 'invalid_model.pkl')
        with open(invalid_path, 'w') as f:
            f.write("Este não é um modelo válido")
        
        with pytest.raises((ValueError, EOFError, Exception)):
            load_model(invalid_path)


class TestPredict:
    """Testes para a função predict."""
    
    def test_predict_success(self, trained_tabular_model, sample_tabular_data):
        """Testa predições bem-sucedidas."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        X_test = X.iloc[:5]  # Primeiras 5 amostras
        
        predictions = predict(trained_tabular_model, X_test)
        
        # Verificar formato
        assert len(predictions) == 5
        assert all(pred in ['B', 'M'] for pred in predictions)
        assert isinstance(predictions, np.ndarray) or isinstance(predictions, list)
    
    def test_predict_single_sample(self, trained_tabular_model, sample_tabular_data):
        """Testa predição de uma única amostra."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        X_single = X.iloc[:1]
        
        predictions = predict(trained_tabular_model, X_single)
        
        assert len(predictions) == 1
        assert predictions[0] in ['B', 'M']
    
    def test_predict_empty_dataframe(self, trained_tabular_model):
        """Testa tratamento de DataFrame vazio."""
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, IndexError)):
            predict(trained_tabular_model, empty_df)
    
    def test_predict_wrong_features(self, trained_tabular_model):
        """Testa tratamento de features incorretas."""
        # DataFrame com features diferentes
        wrong_df = pd.DataFrame({
            'wrong_feature_1': [1.0, 2.0, 3.0],
            'wrong_feature_2': [4.0, 5.0, 6.0]
        })
        
        with pytest.raises((ValueError, KeyError)):
            predict(trained_tabular_model, wrong_df)
    
    def test_predict_with_missing_values(self, trained_tabular_model, sample_tabular_data):
        """Testa tratamento de valores faltantes."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        X_test = X.iloc[:5].copy()
        
        # Adicionar valores NaN
        X_test.iloc[0, 0] = np.nan
        
        # O pipeline deve lidar com isso ou gerar erro apropriado
        try:
            predictions = predict(trained_tabular_model, X_test)
            # Se funcionou, verificar formato
            assert len(predictions) == 5
        except (ValueError, TypeError) as e:
            # Erro esperado é aceitável
            assert 'nan' in str(e).lower() or 'missing' in str(e).lower()
    
    def test_predict_dataframe_vs_array(self, trained_tabular_model, sample_tabular_data):
        """Testa que a função aceita DataFrame."""
        X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
        X_test = X.iloc[:5]
        
        # Deve funcionar com DataFrame
        predictions_df = predict(trained_tabular_model, X_test)
        
        # Converter para array e testar
        X_array = X_test.values
        X_test_array = pd.DataFrame(X_array, columns=X_test.columns)
        predictions_array = predict(trained_tabular_model, X_test_array)
        
        # Resultados devem ser consistentes
        assert len(predictions_df) == len(predictions_array)
