import pandas as pd
import glob

three_class_source = glob.glob('final_3cls_cropped/csv/*.csv')
five_class_source = glob.glob('final_5cls_cropped/csv/*.csv')

def main ():
    df_list = [pd.read_csv(file) for file in five_class_source]
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    combined_df = combined_df.sort_values(by="experiment_name", ascending=True)
    
    combined_df = combined_df.reset_index(drop=True)

    combined_df.to_csv('combined_results5_cls.csv', index=False, decimal=",", float_format="%.4f", sep=";")

    print(combined_df[["experiment_name", "mean_accuracy"]].sort_values(by="mean_accuracy", ascending=False))       

if __name__ == '__main__':
    main()