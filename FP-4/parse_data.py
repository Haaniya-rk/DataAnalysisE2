import pandas as pd

def parsedata():
    df0 = pd.read_csv('data.csv')

    df0.columns = df0.columns.str.strip()
    
    df0.describe(include='all')
    
    df0 = df0[['sex', 'age', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'ca', 'thal']]
    
    df0.describe(include='all')
    
    df0['ca'] = df0['ca'].replace('?', pd.NA)
    df0['thal'] = df0['thal'].replace('?', pd.NA)
    
    df0['ca'] = pd.to_numeric(df0['ca'], errors='coerce')
    df0['thal'] = pd.to_numeric(df0['thal'], errors='coerce')
    
    df0 = df0.dropna(subset=['ca', 'thal']).copy()
    
    cols_to_keep = ['sex', 'age', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'ca', 'thal']
    df_filtered = df0[cols_to_keep].copy()
    
    print(df_filtered.shape)
    print(df_filtered[['ca', 'thal']].head())
    
    print("Datset after deleting rows where ca (number of major blood vessels coloured by fluorscopy) and type of 'thal' (thalessaemia) is unknown", df_filtered)
    
    df_filtered = df_filtered.rename( columns={'trestbps':'blood pressure', 'chol':'cholesterol', 'fbs': 'Fasting Blood Sugar', 'thalac':'maximum heart rate', 'exang':'exercise induced angina', 'ca':'severity of artery blockage', 'thal':'blood defect type'} )
    
    df_filtered.describe()
    
    bins = [0, 30, 40, 50, 60, 70, 120]
    labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
    
    df_filtered['age group'] = pd.cut(df0['age'], bins=bins, labels=labels, right=True)
    
    print(df_filtered)
    
    df_filtered['severity of artery blockage'] = -1 * df_filtered['severity of artery blockage'] 
    
    print(df_filtered[['severity of artery blockage']].head())
    
    df = df_filtered 
    df.describe()

    return df
