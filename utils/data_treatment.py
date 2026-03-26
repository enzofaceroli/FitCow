import pandas as pd

def mode_or_median(values):
    mode = values.mode()
    
    if len(mode) == 1:
        return mode.iloc[0]
    else:
        return values.median()

def main():
    df = pd.read_csv("assets/fitcow_base.csv")
    
    g1_cols = ["av1_g1", "av2_g1", "av3_g1"]
    g2_cols = ["av1_g2", "av2_g2", "av3_g2"]
    
    # removing NaN cols (temporary)
    clean_df = df.dropna(subset= g1_cols + g2_cols)
    
    clean_df = clean_df.reset_index(drop=True)
    
    clean_df["mode_g1"] = clean_df[g1_cols].apply(mode_or_median, axis=1)    
    clean_df["mode_g2"] = clean_df[g2_cols].apply(mode_or_median, axis=1) 
    
    clean_df["ID"] = clean_df["Rebanho"].astype(str) + clean_df["RGD"].astype(str)
    
    clean_df.to_csv("assets/test_df.csv", sep=";", decimal=",",  index=False)

if __name__ == '__main__':
    
    
    main()