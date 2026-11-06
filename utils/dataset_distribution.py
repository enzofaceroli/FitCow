import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("assets/fitcow_label.csv")

# Extrai o identificador do animal
df["animal"] = df["file"].str.split("_").str[0]

# Conta animais únicos por classe
animal_counts = (
    df.groupby("class")["animal"]
    .nunique()
    .sort_index()
    .reset_index(name="count")
)

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))

barras = sns.barplot(
    data=animal_counts,
    x="class",
    y="count",
    ax=ax,
    color="navy"
)

# Valores no topo das barras
ax.bar_label(
    barras.containers[0],
    fmt="%d",
    fontsize=16,
    fontweight="bold",
    # padding=3,
    label_type="center",
    color="white"
)

ax.set_xlabel(
    "BCS Classes",
    fontsize=16,
    fontweight="bold"
)

ax.set_ylabel(
    "Number of Animals",
    fontsize=16,
    fontweight="bold"
)

ax.set_title(
    "Distribution of Animals Across BCS Classes",
    fontsize=16,
    fontweight="bold"
)

ax.tick_params(
    axis="both",
    labelsize=14
)

plt.tight_layout()

plt.savefig(
    "dataset_plots/sb_animals_per_class.pdf",
    format="pdf",
    dpi=300
)

plt.close()