import pandas as pd

def main():
    class_map = {
        2.5: 'magro',
        3.0: 'magro',
        3.5: 'ideal',
        4.0: 'gordo',
        4.5: 'gordo',
    }
    
    df = pd.read_csv('assets/fitcow_label.csv')

    df['class_2'] = df['class'].map(class_map)
    
    df.to_csv('assets/fitcow_label_reduced.csv', index=False)
    
if __name__ == '__main__': 
    main()
