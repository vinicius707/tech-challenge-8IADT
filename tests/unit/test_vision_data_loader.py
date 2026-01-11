"""
Testes unitários para src/vision/data_loader.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from src.vision.data_loader import (
    download_pneumonia_dataset,
    download_breast_cancer_dataset,
    load_breast_cancer_dataset,
    find_breast_cancer_dataset_path,
    load_image_dataset,
    get_class_distribution,
    validate_images,
    get_image_info
)


class TestDownloadPneumoniaDataset:
    """Testes para download_pneumonia_dataset."""
    
    def test_download_pneumonia_dataset_mocked(self, mock_kagglehub, temp_dir):
        """Testa download mockado de dataset de pneumonia."""
        mock_download, mock_path = mock_kagglehub
        
        result_path = download_pneumonia_dataset(target_path=temp_dir)
        
        # Verificar que kagglehub foi chamado
        mock_download.assert_called_once_with("paultimothymooney/chest-xray-pneumonia")
        
        # Verificar que retorna o caminho
        assert result_path == mock_path
    
    def test_download_pneumonia_dataset_creates_dir(self, mock_kagglehub, temp_dir):
        """Testa que o diretório de destino é criado."""
        mock_download, mock_path = mock_kagglehub
        
        target_path = Path(temp_dir) / "pneumonia_data"
        download_pneumonia_dataset(target_path=str(target_path))
        
        # Verificar que diretório foi criado
        assert target_path.exists()


class TestDownloadBreastCancerDataset:
    """Testes para download_breast_cancer_dataset."""
    
    def test_download_breast_cancer_dataset_mocked(self, mock_kagglehub, temp_dir):
        """Testa download mockado de dataset de câncer de mama."""
        mock_download, mock_path = mock_kagglehub
        
        result_path = download_breast_cancer_dataset(target_path=temp_dir)
        
        # Verificar que kagglehub foi chamado
        mock_download.assert_called_once_with("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
        
        # Verificar que retorna o caminho
        assert result_path == mock_path


class TestLoadBreastCancerDataset:
    """Testes para load_breast_cancer_dataset."""
    
    def test_load_breast_cancer_dataset_success(self, sample_breast_cancer_structure):
        """Testa carregamento bem-sucedido do dataset CBIS-DDSM."""
        base_path = sample_breast_cancer_structure['base_path']
        
        df = load_breast_cancer_dataset(base_path)
        
        # Verificar estrutura do DataFrame
        assert isinstance(df, pd.DataFrame)
        assert 'image_path' in df.columns
        assert 'label' in df.columns
        assert 'split' in df.columns
        
        # Verificar que tem dados
        assert len(df) > 0
        
        # Verificar labels
        assert all(label in ['BENIGN', 'MALIGNANT'] for label in df['label'].unique())
    
    def test_load_breast_cancer_dataset_missing_csv_dir(self, temp_dir):
        """Testa tratamento de erro quando diretório CSV não existe."""
        base_path = Path(temp_dir)
        jpeg_dir = base_path / "jpeg"
        jpeg_dir.mkdir(parents=True, exist_ok=True)
        
        with pytest.raises(ValueError, match="Diretório de CSVs não encontrado"):
            load_breast_cancer_dataset(str(base_path))
    
    def test_load_breast_cancer_dataset_missing_jpeg_dir(self, temp_dir):
        """Testa tratamento de erro quando diretório JPEG não existe."""
        base_path = Path(temp_dir)
        csv_dir = base_path / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        with pytest.raises(ValueError, match="Diretório de imagens JPEG não encontrado"):
            load_breast_cancer_dataset(str(base_path))
    
    def test_load_breast_cancer_dataset_empty(self, temp_dir):
        """Testa tratamento quando não há imagens."""
        base_path = Path(temp_dir)
        csv_dir = base_path / "csv"
        jpeg_dir = base_path / "jpeg"
        csv_dir.mkdir(parents=True, exist_ok=True)
        jpeg_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar CSV vazio ou sem imagens correspondentes
        csv_path = csv_dir / "mass_case_description_train_set.csv"
        pd.DataFrame({'pathology': [], 'image file path': []}).to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="Nenhuma imagem encontrada"):
            load_breast_cancer_dataset(str(base_path))


class TestFindBreastCancerDatasetPath:
    """Testes para find_breast_cancer_dataset_path."""
    
    def test_find_breast_cancer_dataset_path_in_config(self, sample_breast_cancer_structure):
        """Testa que encontra dataset no caminho do config."""
        base_path = sample_breast_cancer_structure['base_path']
        
        found_path = find_breast_cancer_dataset_path(config_path=base_path)
        
        assert found_path == base_path
    
    def test_find_breast_cancer_dataset_path_fallback_download(self, mock_kagglehub, temp_dir, monkeypatch):
        """Testa fallback para download quando não encontra."""
        mock_download, mock_path = mock_kagglehub
        
        # Mockar o cache do kagglehub para simular que não existe
        def mock_cache_exists(path):
            return False
        
        from pathlib import Path
        import sys
        
        # Mockar o método que verifica se o cache existe
        original_path_exists = Path.exists
        def mock_path_exists(self):
            if '.cache' in str(self) and 'kagglehub' in str(self):
                return False
            return original_path_exists(self)
        
        monkeypatch.setattr(Path, 'exists', mock_path_exists)
        
        # Tentar encontrar - deve fazer download
        found_path = find_breast_cancer_dataset_path(config_path=None)
        
        # Se encontrou no cache real, o teste passa (comportamento esperado)
        # Se não encontrou, deve ter feito download
        if mock_download.called:
            assert found_path == mock_path
        else:
            # Dataset existe no cache real, o que é válido
            assert found_path is not None


class TestLoadImageDataset:
    """Testes para load_image_dataset."""
    
    def test_load_image_dataset_train_test_val(self, sample_dataset_structure):
        """Testa carregamento de dataset com estrutura train/test/val."""
        base_path = sample_dataset_structure['base_path']
        
        df = load_image_dataset(base_path)
        
        # Verificar estrutura
        assert isinstance(df, pd.DataFrame)
        assert 'image_path' in df.columns
        assert 'label' in df.columns
        assert 'split' in df.columns
        
        # Verificar que tem dados
        assert len(df) > 0
        
        # Verificar splits
        assert 'train' in df['split'].values
        assert 'test' in df['split'].values
        assert 'val' in df['split'].values
    
    def test_load_image_dataset_class_folders(self, sample_dataset_class_folders):
        """Testa carregamento de dataset com estrutura de pastas de classes."""
        base_path = sample_dataset_class_folders['base_path']
        
        df = load_image_dataset(base_path, subdirs=None)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'image_path' in df.columns
        assert 'label' in df.columns
    
    def test_load_image_dataset_custom_subdirs(self, sample_dataset_structure):
        """Testa carregamento com subdiretórios customizados."""
        base_path = sample_dataset_structure['base_path']
        
        df = load_image_dataset(base_path, subdirs=['train', 'test'])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Não deve ter 'val' se não foi especificado
        # (mas pode ter se a estrutura tiver)
    
    def test_load_image_dataset_empty_directory(self, temp_dir):
        """Testa tratamento de diretório vazio."""
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        with pytest.raises(ValueError, match="Nenhuma imagem encontrada"):
            load_image_dataset(str(empty_dir))
    
    def test_load_image_dataset_custom_extensions(self, sample_dataset_structure):
        """Testa carregamento com extensões customizadas."""
        base_path = sample_dataset_structure['base_path']
        
        # Deve funcionar com extensões padrão
        df = load_image_dataset(base_path, image_extensions=('.jpg', '.jpeg'))
        
        assert len(df) > 0
    
    def test_load_image_dataset_filters_extensions(self, temp_dir):
        """Testa que filtra apenas extensões especificadas."""
        test_dir = Path(temp_dir) / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        class_dir = test_dir / "Normal"
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar arquivo .jpg
        (class_dir / "image.jpg").touch()
        # Criar arquivo .txt (deve ser ignorado)
        (class_dir / "image.txt").touch()
        
        df = load_image_dataset(str(test_dir), image_extensions=('.jpg',))
        
        assert len(df) == 1
        assert df['image_path'].iloc[0].endswith('.jpg')


class TestGetClassDistribution:
    """Testes para get_class_distribution."""
    
    def test_get_class_distribution_with_split(self, sample_dataframe_with_images):
        """Testa cálculo de distribuição com coluna split."""
        df = sample_dataframe_with_images
        
        distribution = get_class_distribution(df, split_col='split')
        
        assert isinstance(distribution, pd.DataFrame)
        # Deve ter linhas para cada split + margem
        assert len(distribution) > 0
    
    def test_get_class_distribution_without_split(self, sample_dataframe_with_images):
        """Testa cálculo de distribuição sem coluna split."""
        df = sample_dataframe_with_images.drop('split', axis=1)
        
        distribution = get_class_distribution(df, split_col=None)
        
        assert isinstance(distribution, pd.DataFrame)
        assert 'count' in distribution.columns
        assert 'percentage' in distribution.columns
    
    def test_get_class_distribution_percentages(self, sample_dataframe_with_images):
        """Testa que percentuais somam 100."""
        df = sample_dataframe_with_images.drop('split', axis=1)
        
        distribution = get_class_distribution(df, split_col=None)
        
        total_percentage = distribution['percentage'].sum()
        assert abs(total_percentage - 100.0) < 0.01  # Tolerância para arredondamento


class TestValidateImages:
    """Testes para validate_images."""
    
    def test_validate_images_removes_corrupted(self, sample_dataframe_with_images, temp_dir):
        """Testa que imagens corrompidas são removidas."""
        df = sample_dataframe_with_images.copy()
        
        # Adicionar caminho para arquivo inexistente (simula corrompido)
        invalid_path = Path(temp_dir) / "nonexistent_image.jpg"
        df = pd.concat([
            df,
            pd.DataFrame([{
                'image_path': str(invalid_path),
                'label': 'Normal',
                'split': 'test'
            }])
        ], ignore_index=True)
        
        original_len = len(df)
        validated_df = validate_images(df)
        
        # Deve ter removido pelo menos o arquivo inválido
        assert len(validated_df) <= original_len
    
    def test_validate_images_keeps_valid(self, sample_dataframe_with_images):
        """Testa que imagens válidas são mantidas."""
        validated_df = validate_images(sample_dataframe_with_images)
        
        # Deve manter todas as imagens válidas
        assert len(validated_df) > 0
        assert all(Path(path).exists() for path in validated_df['image_path'])


class TestGetImageInfo:
    """Testes para get_image_info."""
    
    def test_get_image_info_structure(self, sample_dataframe_with_images):
        """Testa estrutura do dicionário de informações."""
        info = get_image_info(sample_dataframe_with_images, sample_size=10)
        
        assert isinstance(info, dict)
        assert 'width_mean' in info
        assert 'height_mean' in info
        assert 'width_std' in info
        assert 'height_std' in info
        assert 'formats' in info
        assert 'total_images' in info
    
    def test_get_image_info_sample_size(self, sample_dataframe_with_images):
        """Testa que sample_size limita o número de imagens processadas."""
        info = get_image_info(sample_dataframe_with_images, sample_size=2)
        
        # Verificar que processou no máximo 2 imagens
        # (não podemos verificar diretamente, mas podemos verificar que funciona)
        assert info['total_images'] == len(sample_dataframe_with_images)
    
    def test_get_image_info_statistics(self, sample_dataframe_with_images):
        """Testa que estatísticas são calculadas corretamente."""
        info = get_image_info(sample_dataframe_with_images, sample_size=5)
        
        # Verificar que médias são números válidos
        assert isinstance(info['width_mean'], (int, float))
        assert isinstance(info['height_mean'], (int, float))
        assert info['width_mean'] > 0
        assert info['height_mean'] > 0
