import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plota matriz de confusão baseado em evaluate.py
def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
    
def plot_training_curves(acc_history, loss_history, save_path=None):
    acc_df = pd.DataFrame.from_dict(acc_history, orient='index')
    loss_df = pd.DataFrame.from_dict(loss_history, orient='index')
    
    epochs = acc_df.columns
    
    acc_mean = acc_df.mean(axis=0)
    acc_std = acc_df.std(axis=0)    
    
    loss_mean = loss_df.mean(axis=0)
    loss_std = loss_df.std(axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, loss_mean, color='red', linewidth=2, label='Loss Média')
    axes[0].fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, color='red', alpha=0.2)
    axes[0].set_title('Convergência do Erro (Loss de Treino)')
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('CrossEntropy Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].plot(epochs, acc_mean, color='blue', linewidth=2, label='Acurácia Média')
    axes[1].fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, color='blue', alpha=0.2)
    axes[1].set_title('Evolução da Acurácia de Treino')
    axes[1].set_xlabel('Épocas')
    axes[1].set_ylabel('Acurácia (%)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.close()