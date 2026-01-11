"""
Fixtures compartilhadas para todos os testes.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Adicionar tests ao path para imports relativos
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

from fixtures.sample_data import (
    create_sample_tabular_data,
    create_sample_image_array,
    create_temp_image_file,
    create_temp_dataset_structure
)


@pytest.fixture
def sample_tabular_data():
    """
    Fixture que retorna um DataFrame sintético com dados tabulares.
    """
    return create_sample_tabular_data(n_samples=100, random_state=42)


@pytest.fixture
def sample_tabular_data_small():
    """
    Fixture com dataset menor para testes rápidos.
    """
    return create_sample_tabular_data(n_samples=20, random_state=42)


@pytest.fixture
def sample_tabular_data_imbalanced():
    """
    Fixture com dataset desbalanceado (90% uma classe, 10% outra).
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 30
    
    data = {}
    base_features = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                     'compactness', 'concavity', 'concave_points', 'symmetry',
                     'fractal_dimension']
    
    for base_feat in base_features:
        mean_val = np.random.uniform(5, 30)
        std_val = np.random.uniform(1, 5)
        data[f'{base_feat}_mean'] = np.random.normal(mean_val, std_val, n_samples)
        data[f'{base_feat}_se'] = np.random.normal(mean_val * 0.1, std_val * 0.1, n_samples)
        data[f'{base_feat}_worst'] = np.random.normal(mean_val * 1.2, std_val * 1.5, n_samples)
    
    data['id'] = range(1, n_samples + 1)
    
    # 90% Benigno, 10% Maligno
    n_benign = int(n_samples * 0.9)
    diagnosis = ['B'] * n_benign + ['M'] * (n_samples - n_benign)
    np.random.shuffle(diagnosis)
    data['diagnosis'] = diagnosis
    
    df = pd.DataFrame(data)
    mask_malignant = df['diagnosis'] == 'M'
    for col in df.columns:
        if col not in ['id', 'diagnosis']:
            df.loc[mask_malignant, col] = df.loc[mask_malignant, col] * 1.1
    
    return df


@pytest.fixture
def temp_dir():
    """
    Fixture que cria um diretório temporário e o remove após o teste.
    """
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image_array_rgb():
    """
    Fixture que retorna um array numpy de imagem RGB sintética.
    """
    return create_sample_image_array(shape=(224, 224, 3), grayscale=False, random_state=42)


@pytest.fixture
def sample_image_array_grayscale():
    """
    Fixture que retorna um array numpy de imagem em escala de cinza sintética.
    """
    return create_sample_image_array(shape=(256, 256, 1), grayscale=True, random_state=42)


@pytest.fixture
def sample_image_path(temp_dir):
    """
    Fixture que cria um arquivo de imagem temporário e retorna o caminho.
    """
    return create_temp_image_file(temp_dir, filename='test_image.jpg', shape=(224, 224, 3))


@pytest.fixture
def sample_image_path_grayscale(temp_dir):
    """
    Fixture que cria um arquivo de imagem em escala de cinza temporário.
    """
    return create_temp_image_file(temp_dir, filename='test_image_gray.jpg', shape=(256, 256, 1), grayscale=True)


@pytest.fixture
def sample_dataset_structure(temp_dir):
    """
    Fixture que cria uma estrutura de dataset temporária (train/test/val).
    """
    return create_temp_dataset_structure(temp_dir, n_images_per_class=5, structure='train_test_val')


@pytest.fixture
def sample_dataset_class_folders(temp_dir):
    """
    Fixture que cria uma estrutura de dataset com pastas de classes.
    """
    return create_temp_dataset_structure(temp_dir, n_images_per_class=5, structure='class_folders')


@pytest.fixture
def mock_kagglehub():
    """
    Fixture que mocka o kagglehub para evitar downloads reais.
    """
    with patch('kagglehub.dataset_download') as mock_download:
        # Simular caminho de retorno
        mock_path = tempfile.mkdtemp()
        mock_download.return_value = mock_path
        yield mock_download, mock_path
        shutil.rmtree(mock_path, ignore_errors=True)


@pytest.fixture
def trained_tabular_model(sample_tabular_data):
    """
    Fixture que cria um modelo tabular pequeno treinado para testes.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Preparar dados
    X = sample_tabular_data.drop(['diagnosis', 'id'], axis=1)
    y = sample_tabular_data['diagnosis']
    
    # Criar e treinar modelo pequeno
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42))
    ])
    
    model.fit(X, y)
    
    return model


@pytest.fixture
def trained_cnn_model():
    """
    Fixture que cria uma CNN pequena treinada para testes.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Criar modelo pequeno
    model = keras.Sequential([
        layers.Conv2D(4, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Treinar com dados sintéticos (1 época apenas)
    X_train = np.random.random((10, 32, 32, 3))
    y_train = np.random.randint(0, 2, 10)  # Array 1D primeiro
    y_train = tf.keras.utils.to_categorical(y_train, 2)  # Depois converter para one-hot
    
    model.fit(X_train, y_train, epochs=1, verbose=0)
    
    return model


@pytest.fixture
def sample_dataframe_with_images(sample_dataset_structure):
    """
    Fixture que retorna um DataFrame com informações de imagens.
    """
    data = []
    for item in sample_dataset_structure['image_paths']:
        data.append({
            'image_path': item['path'],
            'label': item['label'],
            'split': item['split']
        })
    
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def mock_matplotlib_show():
    """
    Fixture que mocka plt.show() para evitar que gráficos sejam exibidos durante testes.
    """
    with patch('matplotlib.pyplot.show'):
        yield


@pytest.fixture
def sample_breast_cancer_structure(temp_dir):
    """
    Fixture que cria estrutura simulando dataset CBIS-DDSM.
    """
    base_path = Path(temp_dir)
    
    # Criar estrutura CSV e JPEG
    csv_dir = base_path / "csv"
    jpeg_dir = base_path / "jpeg"
    csv_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar CSV de exemplo
    csv_data = {
        'pathology': ['BENIGN', 'MALIGNANT', 'BENIGN'],
        'image file path': [
            'Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/000000.dcm',
            'Mass-Training_P_00002_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/000001.dcm',
            'Mass-Training_P_00003_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/000002.dcm'
        ]
    }
    
    df_csv = pd.DataFrame(csv_data)
    csv_path = csv_dir / "mass_case_description_train_set.csv"
    df_csv.to_csv(csv_path, index=False)
    
    # Criar diretórios JPEG correspondentes
    for i, row in df_csv.iterrows():
        dicom_path = row['image file path']
        path_parts = dicom_path.split('/')
        if len(path_parts) >= 2:
            dicom_dir = path_parts[-2]
            jpeg_subdir = jpeg_dir / dicom_dir
            jpeg_subdir.mkdir(parents=True, exist_ok=True)
            
            # Criar imagem JPEG
            from fixtures.sample_data import create_temp_image_file
            create_temp_image_file(jpeg_subdir, filename='image.jpg', shape=(256, 256, 1), grayscale=True)
    
    return {
        'base_path': str(base_path),
        'csv_dir': str(csv_dir),
        'jpeg_dir': str(jpeg_dir)
    }
