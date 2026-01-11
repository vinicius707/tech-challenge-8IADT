"""
Módulo para carregamento e preparação de datasets de imagens médicas.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import kagglehub


def download_pneumonia_dataset(target_path: str = "data/images/pneumonia") -> str:
    """
    Baixa o dataset de pneumonia em raio-X do Kaggle.
    
    Parameters:
    -----------
    target_path : str
        Caminho onde o dataset será salvo.
    
    Returns:
    --------
    str
        Caminho para o dataset baixado.
    """
    print("Baixando dataset de pneumonia em raio-X...")
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Dataset baixado em: {path}")
    
    # Criar diretório de destino se não existir
    os.makedirs(target_path, exist_ok=True)
    
    return path


def download_breast_cancer_dataset(target_path: str = "data/images/breast_cancer") -> str:
    """
    Baixa o dataset de câncer de mama (CBIS-DDSM) do Kaggle.
    
    Parameters:
    -----------
    target_path : str
        Caminho onde o dataset será salvo.
    
    Returns:
    --------
    str
        Caminho para o dataset baixado.
    """
    print("Baixando dataset de câncer de mama (CBIS-DDSM)...")
    path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
    print(f"Dataset baixado em: {path}")
    
    # Criar diretório de destino se não existir
    os.makedirs(target_path, exist_ok=True)
    
    return path


def load_breast_cancer_dataset(base_path: str) -> pd.DataFrame:
    """
    Carrega o dataset CBIS-DDSM de câncer de mama.
    Este dataset tem estrutura diferente: imagens em jpeg/ e labels nos CSVs.
    
    Parameters:
    -----------
    base_path : str
        Caminho base do dataset (geralmente do kagglehub cache).
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com colunas 'image_path', 'label' e 'split'.
    """
    base_path = Path(base_path)
    data = []
    
    # Caminhos dos CSVs
    csv_dir = base_path / "csv"
    jpeg_dir = base_path / "jpeg"
    
    if not csv_dir.exists():
        raise ValueError(f"Diretório de CSVs não encontrado: {csv_dir}")
    if not jpeg_dir.exists():
        raise ValueError(f"Diretório de imagens JPEG não encontrado: {jpeg_dir}")
    
    # Lista de CSVs para processar
    csv_files = [
        ("mass_case_description_train_set.csv", "train"),
        ("mass_case_description_test_set.csv", "test"),
        ("calc_case_description_train_set.csv", "train"),
        ("calc_case_description_test_set.csv", "test")
    ]
    
    for csv_file, split in csv_files:
        csv_path = csv_dir / csv_file
        if not csv_path.exists():
            continue
        
        # Ler CSV
        df_csv = pd.read_csv(csv_path)
        
        # Verificar colunas necessárias
        if 'pathology' not in df_csv.columns or 'image file path' not in df_csv.columns:
            continue
        
        # Processar cada linha
        for _, row in df_csv.iterrows():
            # Obter label (BENIGN ou MALIGNANT)
            pathology = str(row['pathology']).strip().upper()
            if pathology not in ['BENIGN', 'MALIGNANT']:
                continue
            
            # Obter caminho da imagem DICOM do CSV
            dicom_path = str(row['image file path']).strip()
            if pd.isna(dicom_path) or not dicom_path:
                continue
            
            # Converter caminho DICOM para caminho JPEG
            # O caminho é algo como: Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/000000.dcm
            # O último diretório contém o ID DICOM que corresponde ao diretório em jpeg/
            path_parts = dicom_path.split('/')
            if len(path_parts) < 2:
                continue
            
            # O último diretório contém o ID DICOM que corresponde ao diretório em jpeg/
            dicom_dir = path_parts[-2]
            
            # Construir caminho do diretório JPEG
            jpeg_subdir = jpeg_dir / dicom_dir
            
            # Procurar qualquer arquivo JPEG neste diretório
            # (o nome do arquivo JPEG pode ser diferente do DICOM)
            if jpeg_subdir.exists() and jpeg_subdir.is_dir():
                jpeg_files = list(jpeg_subdir.glob("*.jpg"))
                if jpeg_files:
                    # Usar o primeiro arquivo JPEG encontrado no diretório
                    # (geralmente há apenas um por diretório)
                    jpeg_path = jpeg_files[0]
                    data.append({
                        'image_path': str(jpeg_path),
                        'label': pathology,
                        'split': split
                    })
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        raise ValueError(
            f"Nenhuma imagem encontrada em {base_path}. "
            f"Verifique se o dataset foi baixado corretamente e se os CSVs estão no diretório csv/."
        )
    
    return df


def find_pneumonia_dataset_path(config_path: Optional[str] = None) -> str:
    """
    Tenta encontrar o caminho do dataset de pneumonia.
    Primeiro verifica o caminho do config, depois o cache do kagglehub.
    
    Parameters:
    -----------
    config_path : str, optional
        Caminho do config.yaml para verificar primeiro.
    
    Returns:
    --------
    str
        Caminho para o dataset encontrado.
    """
    def _has_images(path: Path) -> bool:
        """Verifica se um caminho contém imagens do dataset de pneumonia."""
        # Verificar se tem estrutura train/test/val com classes
        train_dir = path / "train"
        test_dir = path / "test"
        val_dir = path / "val"
        
        # Verificar se pelo menos um dos subdiretórios existe e tem imagens
        for subdir in [train_dir, test_dir, val_dir]:
            if subdir.exists():
                # Verificar se tem subdiretórios de classes
                class_dirs = [d for d in subdir.iterdir() if d.is_dir()]
                if class_dirs:
                    # Verificar se pelo menos uma classe tem imagens
                    for class_dir in class_dirs:
                        if len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))) > 0:
                            return True
        
        # Verificar também em subdiretórios comuns (chest_xray, etc)
        common_subdirs = ['chest_xray', 'chest-xray']
        for common_subdir in common_subdirs:
            common_path = path / common_subdir
            if common_path.exists():
                for subdir in [common_path / "train", common_path / "test", common_path / "val"]:
                    if subdir.exists():
                        class_dirs = [d for d in subdir.iterdir() if d.is_dir()]
                        if class_dirs:
                            for class_dir in class_dirs:
                                if len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))) > 0:
                                    return True
        
        return False
    
    # #region agent log
    import json
    log_path = Path("/Users/vinicius/Documents/pós/tech-challenge-8IADT/.cursor/debug.log")
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "C,E",
                "location": "data_loader.py:199",
                "message": "find_pneumonia_dataset_path entry",
                "data": {"config_path": str(config_path) if config_path else None},
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    
    # Se um caminho foi fornecido, verificar se existe e tem imagens
    if config_path:
        config_path_obj = Path(config_path)
        # #region agent log
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A,B",
                    "location": "data_loader.py:211",
                    "message": "checking config_path",
                    "data": {"config_path": str(config_path_obj), "exists": config_path_obj.exists(), "has_images": _has_images(config_path_obj) if config_path_obj.exists() else False},
                    "timestamp": int(__import__("time").time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        if config_path_obj.exists() and _has_images(config_path_obj):
            return str(config_path_obj)
    
    # Tentar encontrar no cache do kagglehub
    home = Path.home()
    kaggle_cache = home / ".cache" / "kagglehub" / "datasets" / "paultimothymooney" / "chest-xray-pneumonia"
    
    # #region agent log
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "C",
                "location": "data_loader.py:226",
                "message": "checking kagglehub cache",
                "data": {"kaggle_cache": str(kaggle_cache), "exists": kaggle_cache.exists()},
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    
    if kaggle_cache.exists():
        # Procurar por versões (mais recente primeiro)
        for version_dir in sorted(kaggle_cache.glob("versions/*"), reverse=True):
            if version_dir.is_dir() and _has_images(version_dir):
                # #region agent log
                try:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "C",
                            "location": "data_loader.py:235",
                            "message": "found dataset in kagglehub cache",
                            "data": {"version_dir": str(version_dir)},
                            "timestamp": int(__import__("time").time() * 1000)
                        }) + "\n")
                except: pass
                # #endregion
                print(f"Dataset encontrado no cache: {version_dir}")
                return str(version_dir)
    
    # Se não encontrou, fazer download
    print("Dataset não encontrado. Fazendo download...")
    # #region agent log
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "E",
                "location": "data_loader.py:248",
                "message": "triggering download",
                "data": {"target_path": str(config_path or "data/images/pneumonia")},
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    return download_pneumonia_dataset(config_path or "data/images/pneumonia")


def find_breast_cancer_dataset_path(config_path: Optional[str] = None) -> str:
    """
    Tenta encontrar o caminho do dataset de câncer de mama.
    Primeiro verifica o caminho do config, depois o cache do kagglehub.
    
    Parameters:
    -----------
    config_path : str, optional
        Caminho do config.yaml para verificar primeiro.
    
    Returns:
    --------
    str
        Caminho para o dataset encontrado.
    """
    def _has_images(path: Path) -> bool:
        """Verifica se um caminho contém imagens do dataset CBIS-DDSM."""
        csv_dir = path / "csv"
        jpeg_dir = path / "jpeg"
        return csv_dir.exists() and jpeg_dir.exists() and len(list(jpeg_dir.glob("**/*.jpg"))) > 0
    
    # Se um caminho foi fornecido, verificar se existe e tem imagens
    if config_path:
        config_path_obj = Path(config_path)
        if config_path_obj.exists() and _has_images(config_path_obj):
            return str(config_path_obj)
    
    # Tentar encontrar no cache do kagglehub
    home = Path.home()
    kaggle_cache = home / ".cache" / "kagglehub" / "datasets" / "awsaf49" / "cbis-ddsm-breast-cancer-image-dataset"
    
    if kaggle_cache.exists():
        # Procurar por versões (mais recente primeiro)
        for version_dir in sorted(kaggle_cache.glob("versions/*"), reverse=True):
            if version_dir.is_dir() and _has_images(version_dir):
                print(f"Dataset encontrado no cache: {version_dir}")
                return str(version_dir)
    
    # Se não encontrou, fazer download
    print("Dataset não encontrado. Fazendo download...")
    return download_breast_cancer_dataset(config_path or "data/images/breast_cancer")


def load_image_dataset(
    base_path: str,
    subdirs: Optional[List[str]] = None,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
) -> pd.DataFrame:
    """
    Carrega imagens de um diretório estruturado e cria um DataFrame com paths e labels.
    
    Parameters:
    -----------
    base_path : str
        Caminho base do dataset.
    subdirs : list, optional
        Lista de subdiretórios para processar (ex: ['train', 'test', 'val']).
        Se None, processa todos os subdiretórios encontrados.
    image_extensions : tuple
        Extensões de arquivo de imagem aceitas.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com colunas 'image_path' e 'label'.
    """
    # #region agent log
    import json
    log_path = Path("/Users/vinicius/Documents/pós/tech-challenge-8IADT/.cursor/debug.log")
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A,B,C",
                "location": "data_loader.py:224",
                "message": "load_image_dataset entry",
                "data": {"base_path": str(base_path), "base_path_exists": str(Path(base_path).exists())},
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    base_path = Path(base_path)
    
    # #region agent log
    try:
        with open(log_path, "a") as f:
            resolved_path = base_path.resolve()
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A,B",
                "location": "data_loader.py:245",
                "message": "base_path resolved",
                "data": {"resolved_path": str(resolved_path), "exists": resolved_path.exists(), "is_dir": resolved_path.is_dir() if resolved_path.exists() else None, "listing": [str(p.name) for p in resolved_path.iterdir()] if resolved_path.exists() and resolved_path.is_dir() else []},
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    
    data = []
    
    # Se subdirs não for especificado, procurar por subdiretórios comuns
    if subdirs is None:
        subdirs = ['train', 'test', 'val', 'validation']
    
    # #region agent log
    try:
        with open(log_path, "a") as f:
            found_subdirs = [sd for sd in subdirs if (base_path / sd).exists()]
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "D",
                "location": "data_loader.py:256",
                "message": "checking subdirs",
                "data": {"subdirs_checked": subdirs, "found_subdirs": found_subdirs},
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    
    # Processar cada subdiretório
    for subdir in subdirs:
        subdir_path = base_path / subdir
        if not subdir_path.exists():
            continue
        
        # Procurar por classes (subdiretórios dentro de train/test/val)
        for class_dir in subdir_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            label = class_dir.name
            # Processar imagens no diretório da classe
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    data.append({
                        'image_path': str(img_file),
                        'label': label,
                        'split': subdir
                    })
    
    # #region agent log
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "D",
                "location": "data_loader.py:275",
                "message": "after first search, checking common subdirs",
                "data": {"data_found": len(data), "base_path_contents": [str(p.name) for p in base_path.iterdir()] if base_path.exists() and base_path.is_dir() else []},
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    
    # Se não encontrou estrutura train/test/val, tentar procurar em subdiretórios comuns
    if len(data) == 0:
        # Lista de subdiretórios comuns onde datasets podem estar organizados
        common_subdirs = ['chest_xray', 'chest-xray', 'data', 'images']
        
        # #region agent log
        try:
            with open(log_path, "a") as f:
                found_common = [csd for csd in common_subdirs if (base_path / csd).exists()]
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "D",
                    "location": "data_loader.py:283",
                    "message": "checking common subdirs",
                    "data": {"common_subdirs": common_subdirs, "found_common": found_common},
                    "timestamp": int(__import__("time").time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        for common_subdir in common_subdirs:
            potential_path = base_path / common_subdir
            if potential_path.exists() and potential_path.is_dir():
                # Tentar novamente com o subdiretório comum
                for subdir in subdirs:
                    subdir_path = potential_path / subdir
                    if not subdir_path.exists():
                        continue
                    
                    # Procurar por classes (subdiretórios dentro de train/test/val)
                    for class_dir in subdir_path.iterdir():
                        if not class_dir.is_dir():
                            continue
                        
                        label = class_dir.name
                        # Processar imagens no diretório da classe
                        for img_file in class_dir.iterdir():
                            if img_file.suffix.lower() in image_extensions:
                                data.append({
                                    'image_path': str(img_file),
                                    'label': label,
                                    'split': subdir
                                })
                
                # Se encontrou dados, parar de procurar
                if len(data) > 0:
                    break
        
        # Se ainda não encontrou, procurar diretamente por classes no caminho base
        if len(data) == 0:
            for class_dir in base_path.iterdir():
                if not class_dir.is_dir():
                    continue
                
                label = class_dir.name
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        data.append({
                            'image_path': str(img_file),
                            'label': label,
                            'split': 'unknown'
                        })
    
    df = pd.DataFrame(data)
    
    # #region agent log
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A,B,C,D,E",
                "location": "data_loader.py:324",
                "message": "load_image_dataset exit",
                "data": {"data_count": len(data), "df_length": len(df), "base_path": str(base_path)},
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    
    if len(df) == 0:
        raise ValueError(
            f"Nenhuma imagem encontrada em {base_path}. "
            f"Verifique se o caminho está correto e se o dataset foi baixado corretamente."
        )
    
    return df


def create_dataframe(
    base_path: str,
    structure_type: str = "class_folders"
) -> pd.DataFrame:
    """
    Cria um DataFrame com paths e labels das imagens.
    
    Parameters:
    -----------
    base_path : str
        Caminho base do dataset.
    structure_type : str
        Tipo de estrutura: 'class_folders' ou 'train_test_split'.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com informações das imagens.
    """
    return load_image_dataset(base_path)


def get_class_distribution(df: pd.DataFrame, split_col: Optional[str] = 'split') -> pd.DataFrame:
    """
    Analisa a distribuição de classes no dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com colunas 'label' e opcionalmente 'split'.
    split_col : str, optional
        Nome da coluna que indica o split (train/test/val).
    
    Returns:
    --------
    pd.DataFrame
        Tabela com distribuição de classes.
    """
    if split_col and split_col in df.columns:
        distribution = pd.crosstab(df[split_col], df['label'], margins=True)
    else:
        distribution = df['label'].value_counts().to_frame('count')
        distribution['percentage'] = (distribution['count'] / len(df) * 100).round(2)
    
    return distribution


def validate_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida imagens e remove arquivos corrompidos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com coluna 'image_path'.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com apenas imagens válidas.
    """
    from PIL import Image
    
    valid_paths = []
    
    for idx, row in df.iterrows():
        try:
            img = Image.open(row['image_path'])
            img.verify()  # Verifica se a imagem não está corrompida
            valid_paths.append(idx)
        except Exception as e:
            print(f"Imagem inválida removida: {row['image_path']} - {str(e)}")
    
    return df.loc[valid_paths].reset_index(drop=True)


def get_image_info(df: pd.DataFrame, sample_size: int = 100) -> dict:
    """
    Obtém informações sobre as imagens do dataset (dimensões, formatos, etc.).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com coluna 'image_path'.
    sample_size : int
        Número de imagens para amostrar (para análise rápida).
    
    Returns:
    --------
    dict
        Dicionário com informações sobre as imagens.
    """
    from PIL import Image
    
    sample_df = df.sample(min(sample_size, len(df)))
    
    widths = []
    heights = []
    formats = []
    
    for img_path in sample_df['image_path']:
        try:
            img = Image.open(img_path)
            widths.append(img.width)
            heights.append(img.height)
            formats.append(img.format)
        except Exception as e:
            print(f"Erro ao processar {img_path}: {str(e)}")
    
    info = {
        'width_mean': np.mean(widths) if widths else 0,
        'width_std': np.std(widths) if widths else 0,
        'height_mean': np.mean(heights) if heights else 0,
        'height_std': np.std(heights) if heights else 0,
        'formats': pd.Series(formats).value_counts().to_dict() if formats else {},
        'total_images': len(df)
    }
    
    return info


