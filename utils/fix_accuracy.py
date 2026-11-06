import pandas as pd
import glob
import os

results_3cls_df = pd.read_csv(
    'combined_results3cls.csv',
    sep=';',
    decimal=','
)

results_5cls_df = pd.read_csv(
    'combined_results5_cls.csv',
    sep=';',
    decimal=','
)

dataset_path = 'assets/cropped_dataset'

fold_count = {}

for i in range(1, 6):
    fold_path = os.path.join(dataset_path, f'fold_{i}')
    fold_count[str(i)] = len(glob.glob(f'{fold_path}/*.jpg'))

weights = [fold_count[str(i)] for i in range(1, 6)]
total = sum(weights)


def compute_weighted_accuracy(df):
    acc_cols = [f'fold_{i}_acc' for i in range(1, 6)]

    output = df[['model', 'batch_size', 'learning_rate']].copy()

    weighted_sum = sum(
        df[col] * weights[i]
        for i, col in enumerate(acc_cols)
    )

    output['accuracy'] = weighted_sum / total

    # formatar para 4 casas decimais
    output['learning_rate'] = output['learning_rate'].map(
        lambda x: f"{x:.5f}".replace('.', ',')
    )

    output['accuracy'] = output['accuracy'].map(
        lambda x: f"{x:.4f}".replace('.', ',')
    )

    return output


results_3cls_weighted = compute_weighted_accuracy(results_3cls_df)
results_5cls_weighted = compute_weighted_accuracy(results_5cls_df)

results_3cls_weighted.to_csv(
    'combined_results3cls_weighted.csv',
    sep=';',
    index=False
)

results_5cls_weighted.to_csv(
    'combined_results5cls_weighted.csv',
    sep=';',
    index=False
)