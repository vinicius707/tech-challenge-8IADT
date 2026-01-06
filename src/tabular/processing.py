from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def split_data(X, y, test_size=0.2, validation_size=0.2, random_state=42, return_validation=False):
    """
    Divide os dados em conjuntos de treino, validação e teste.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Variável alvo
    test_size : float, default=0.2
        Proporção do conjunto de teste
    validation_size : float, default=0.2
        Proporção do conjunto de validação (do conjunto restante após remover teste)
    random_state : int, default=42
        Seed para reprodutibilidade
    return_validation : bool, default=False
        Se True, retorna também o conjunto de validação
        
    Returns:
    --------
    Se return_validation=False:
        X_train, X_test, y_train, y_test
    Se return_validation=True:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if return_validation:
        # Primeira divisão: separar teste do restante
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Segunda divisão: separar treino e validação do restante
        # Ajustar validation_size para considerar apenas o conjunto temporário
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        # Comportamento original: apenas treino e teste
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

def build_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
