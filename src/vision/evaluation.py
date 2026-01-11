"""
Módulo para avaliação e interpretabilidade de modelos de classificação de imagens.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from tensorflow import keras
from typing import Tuple, Optional, List
import tensorflow as tf


def plot_training_history(history: keras.callbacks.History, figsize: Tuple[int, int] = (12, 4)):
    """
    Visualiza o histórico de treinamento (loss e accuracy).
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Histórico retornado pelo método fit().
    figsize : tuple
        Tamanho da figura.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss do Modelo')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Accuracy do Modelo')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = False
):
    """
    Plota a matriz de confusão.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Labels verdadeiros.
    y_pred : np.ndarray
        Labels preditos.
    class_names : list
        Nomes das classes.
    figsize : tuple
        Tamanho da figura.
    normalize : bool
        Se True, normaliza a matriz (percentual).
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Matriz de Confusão Normalizada'
    else:
        fmt = 'd'
        title = 'Matriz de Confusão'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plota a curva ROC para cada classe.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Labels verdadeiros (one-hot encoded ou índices).
    y_pred_proba : np.ndarray
        Probabilidades preditas.
    class_names : list
        Nomes das classes.
    figsize : tuple
        Tamanho da figura.
    """
    plt.figure(figsize=figsize)
    
    # Converter y_true para one-hot se necessário
    if len(y_true.shape) == 1:
        from tensorflow.keras.utils import to_categorical
        num_classes = len(class_names)
        y_true = to_categorical(y_true, num_classes)
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_predictions(
    model: keras.Model,
    generator: tf.keras.utils.Sequence,
    class_names: List[str],
    num_samples: int = 16,
    figsize: Tuple[int, int] = (16, 16)
):
    """
    Visualiza predições do modelo em amostras do conjunto de teste.
    
    Parameters:
    -----------
    model : keras.Model
        Modelo treinado.
    generator : tf.keras.utils.Sequence
        Data generator.
    class_names : list
        Nomes das classes.
    num_samples : int
        Número de amostras para visualizar.
    figsize : tuple
        Tamanho da figura.
    """
    # Obter batch do generator
    x_batch, y_batch = generator[0]
    
    # Fazer predições
    y_pred_proba = model.predict(x_batch[:num_samples], verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_batch[:num_samples], axis=1)
    
    # Calcular grid
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(x_batch[i])
        ax.axis('off')
        
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        confidence = y_pred_proba[i][y_pred[i]] * 100
        
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        ax.set_title(
            f'Verdadeiro: {true_label}\nPredito: {pred_label} ({confidence:.1f}%)',
            color=color,
            fontsize=10
        )
    
    # Remover eixos extras
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def grad_cam_visualization(
    model: keras.Model,
    img_array: np.ndarray,
    layer_name: Optional[str] = None,
    pred_index: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementa Grad-CAM para visualizar regiões importantes na imagem.
    
    Parameters:
    -----------
    model : keras.Model
        Modelo treinado.
    img_array : np.ndarray
        Imagem como array numpy (já pré-processada).
    layer_name : str, optional
        Nome da camada convolucional a ser usada. Se None, usa a última camada convolucional.
    pred_index : int, optional
        Índice da classe para calcular o Grad-CAM. Se None, usa a classe predita.
    
    Returns:
    --------
    tuple
        (heatmap, superimposed_img)
    """
    # Expandir dimensões se necessário
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    
    # Garantir que o modelo está construído
    if not model.built:
        # Fazer uma predição dummy para construir o modelo
        _ = model.predict(img_array, verbose=0)
    
    # Encontrar a última camada convolucional
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    if layer_name is None:
        raise ValueError("Nenhuma camada convolucional encontrada no modelo")
    
    # Criar modelo para Grad-CAM
    # Para modelos Sequential, sempre construir funcionalmente para evitar problemas com model.inputs
    # Detectar se é Sequential ou tentar usar model.inputs
    use_functional = isinstance(model, keras.Sequential)
    
    if not use_functional:
        # Tentar usar model.inputs diretamente (funciona para modelos funcionais)
        try:
            model_inputs = model.inputs
            conv_layer = model.get_layer(layer_name)
            conv_output = conv_layer.output
            model_output = model.output
            
            grad_model = keras.Model(
                inputs=model_inputs,
                outputs=[conv_output, model_output]
            )
        except (AttributeError, ValueError, TypeError):
            # Se falhar, usar construção funcional
            use_functional = True
    
    if use_functional:
        # Construir funcionalmente para modelos Sequential
        # Obter o shape de entrada
        if hasattr(model, 'input_shape') and model.input_shape:
            input_shape = model.input_shape[1:]  # Remover dimensão do batch
        else:
            input_shape = img_array.shape[1:]  # Usar shape da imagem
        
        # Criar input funcional
        model_input = keras.Input(shape=input_shape)
        x = model_input
        conv_output = None
        
        # Iterar pelas camadas para construir o modelo funcionalmente
        for layer in model.layers:
            x = layer(x)
            if layer.name == layer_name:
                conv_output = x
        
        model_output = x
        
        if conv_output is None:
            raise ValueError(f"Camada {layer_name} não encontrada no modelo")
        
        # Criar modelo funcional para Grad-CAM
        grad_model = keras.Model(inputs=model_input, outputs=[conv_output, model_output])
    
    # Calcular gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Calcular gradientes da classe predita em relação à saída da camada convolucional
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Pooling dos gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiplicar cada mapa de características pelo peso correspondente
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalizar heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Redimensionar heatmap para o tamanho da imagem original
    heatmap = np.uint8(255 * heatmap)
    
    # Criar imagem sobreposta
    import cv2
    img = img_array[0]
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Redimensionar heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
    
    # Superpor heatmap na imagem
    superimposed_img = heatmap_colored * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 1)
    
    return heatmap, superimposed_img


def plot_grad_cam(
    model: keras.Model,
    img_array: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Plota a imagem original e o Grad-CAM.
    
    Parameters:
    -----------
    model : keras.Model
        Modelo treinado.
    img_array : np.ndarray
        Imagem como array numpy.
    class_names : list
        Nomes das classes.
    figsize : tuple
        Tamanho da figura.
    """
    # Fazer predição
    if len(img_array.shape) == 3:
        pred_input = np.expand_dims(img_array, axis=0)
    else:
        pred_input = img_array
    
    predictions = model.predict(pred_input, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class] * 100
    
    # Calcular Grad-CAM
    heatmap, superimposed = grad_cam_visualization(model, img_array)
    
    # Plotar
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Imagem original
    axes[0].imshow(img_array)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Superposição
    axes[2].imshow(superimposed)
    axes[2].set_title(
        f'Predição: {class_names[pred_class]}\nConfiança: {confidence:.1f}%'
    )
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def evaluate_model(
    model: keras.Model,
    generator: tf.keras.utils.Sequence,
    class_names: List[str],
    verbose: bool = True
) -> dict:
    """
    Avalia o modelo completamente e retorna métricas.
    
    Parameters:
    -----------
    model : keras.Model
        Modelo treinado.
    generator : tf.keras.utils.Sequence
        Data generator.
    class_names : list
        Nomes das classes.
    verbose : bool
        Se True, imprime as métricas.
    
    Returns:
    --------
    dict
        Dicionário com métricas.
    """
    # Avaliar modelo
    results = model.evaluate(generator, verbose=0)
    
    # Obter predições
    y_pred_proba = model.predict(generator, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Obter labels verdadeiros
    y_true = []
    for i in range(len(generator)):
        _, y_batch = generator[i]
        y_true.extend(np.argmax(y_batch, axis=1))
    y_true = np.array(y_true)
    
    # Métricas
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Calcular AUC
    from tensorflow.keras.utils import to_categorical
    y_true_onehot = to_categorical(y_true, len(class_names))
    roc_auc = {}
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc[class_name] = auc(fpr, tpr)
    
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'classification_report': report,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if verbose:
        print("=" * 60)
        print("Métricas de Avaliação")
        print("=" * 60)
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        print("\nROC AUC por classe:")
        for class_name, auc_score in roc_auc.items():
            print(f"  {class_name}: {auc_score:.4f}")
        print("=" * 60)
    
    return metrics


