import os
import pandas as pd


path = r'../dataset/abnormal'
files = [os.path.join(path, file) for file in os.listdir(path)]
results = []

for file in files:
    df = pd.read_csv(file)

    file_name = os.path.basename(file)
    dimensions = df.shape[1] - 1
    data_count = df.shape[0]
    buggy_count = (df.iloc[:, -1] == 1).sum()
    if data_count == 0:
        defects_rate = 'N/A'
    else:
        defects_rate = round(buggy_count / data_count, 3) * 100

    results.append({
        'Datasets': file_name,
        'Metrics': dimensions,
        'Modulus': data_count,
        'Defects': buggy_count,
        'Defects(%)': defects_rate
    })

results = pd.DataFrame(results)
results.to_excel('Dataset Details.xlsx', index=False)
