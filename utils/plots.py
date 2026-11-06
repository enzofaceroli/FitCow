import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.15
)

COLORS = {
    "train": "#264653",
    "eval": "#C44E52",
    "train_fill": "#264653",
    "eval_fill": "#C44E52"
}

def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 16, "weight": "bold"},
        ax=ax
    )

    ax.set_xlabel("Predicted Class", fontsize=16, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=16, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)

    ax.tick_params(axis="both", labelsize=14)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

def _get_stats(history):
    df = pd.DataFrame.from_dict(history, orient="index")

    mean = df.mean(axis=0)
    std = df.std(axis=0)

    epochs = np.arange(1, len(mean) + 1)

    return epochs, mean, std

def _plot_curve(ax, train_history, eval_history, title, ylabel):
    epochs, train_mean, train_std = _get_stats(train_history)
    _, eval_mean, eval_std = _get_stats(eval_history)

    ax.plot(
        epochs,
        train_mean,
        color=COLORS["train"],
        linewidth=2.5,
        label="Training"
    )

    ax.fill_between(
        epochs,
        train_mean - train_std,
        train_mean + train_std,
        color=COLORS["train_fill"],
        alpha=0.15
    )

    ax.plot(
        epochs,
        eval_mean,
        color=COLORS["eval"],
        linewidth=2.5,
        label="Validation"
    )

    ax.fill_between(
        epochs,
        eval_mean - eval_std,
        eval_mean + eval_std,
        color=COLORS["eval_fill"],
        alpha=0.15
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")

    ax.tick_params(axis="both", labelsize=14)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.legend(
        frameon=True,
        fancybox=True,
        shadow=False,
        loc="best",
        fontsize=14,
        prop={"weight": "bold", "size": 14}
    )

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(1, len(epochs))

def _save_curve(train_history, eval_history, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    _plot_curve(
        ax,
        train_history,
        eval_history,
        title=title,
        ylabel=ylabel
    )

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_training_curves(
    acc_history,
    mae_history,
    loss_history,
    eval_acc_history,
    eval_mae_history,
    eval_loss_history,
    save_path=None
):
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    paths = {
        "loss": os.path.join(save_path, "plot_loss.pdf") if save_path else None,
        "accuracy": os.path.join(save_path, "plot_accuracy.pdf") if save_path else None,
        "mae": os.path.join(save_path, "plot_mae.pdf") if save_path else None
    }

    _save_curve(
        loss_history,
        eval_loss_history,
        title="Training and Validation Loss",
        ylabel="Loss",
        save_path=paths["loss"]
    ) if save_path else _save_curve(
        loss_history, eval_loss_history,
        "Training and Validation Loss", "Loss", None
    )

    _save_curve(
        acc_history,
        eval_acc_history,
        title="Training and Validation Accuracy",
        ylabel="Accuracy (%)",
        save_path=paths["accuracy"]
    ) if save_path else _save_curve(
        acc_history, eval_acc_history,
        "Training and Validation Accuracy", "Accuracy (%)", None
    )

    _save_curve(
        mae_history,
        eval_mae_history,
        title="Training and Validation MAE",
        ylabel="MAE (BCS points)",
        save_path=paths["mae"]
    ) if save_path else _save_curve(
        mae_history, eval_mae_history,
        "Training and Validation MAE", "MAE (BCS points)", None
    )