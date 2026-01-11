"""
Utilitários para criar imagens sintéticas para testes.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import tempfile


def create_synthetic_image(width=224, height=224, channels=3, mode='RGB'):
    """
    Cria uma imagem sintética usando PIL.
    
    Parameters:
    -----------
    width : int
        Largura da imagem.
    height : int
        Altura da imagem.
    channels : int
        Número de canais (3 para RGB, 1 para grayscale).
    mode : str
        Modo PIL ('RGB', 'L', etc.).
    
    Returns:
    --------
    Image
        Objeto PIL Image.
    """
    if mode == 'L' or channels == 1:
        array = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        return Image.fromarray(array, mode='L')
    else:
        array = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        return Image.fromarray(array, mode='RGB')


def save_temp_image(image, temp_dir, filename='test.jpg'):
    """
    Salva uma imagem em um diretório temporário.
    
    Parameters:
    -----------
    image : PIL.Image
        Imagem a ser salva.
    temp_dir : str ou Path
        Diretório temporário.
    filename : str
        Nome do arquivo.
    
    Returns:
    --------
    str
        Caminho do arquivo salvo.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    filepath = temp_dir / filename
    image.save(filepath)
    return str(filepath)
