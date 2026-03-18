import pandas as pd
df = pd.read_csv(r'.\Data\prep_outputs\train_smote.csv', nrows=2)
print(df.columns.tolist())