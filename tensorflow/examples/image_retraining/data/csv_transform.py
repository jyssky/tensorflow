import sys
import os.path

import pandas as pd


csv = pd.read_csv("/home/jys/Documents/kaggle/invasive_species_monitoring/result.csv",
                  names = ['name', 'invasive'], skiprows=1, header=None)

new_csv = pd.DataFrame()

print(csv)

new_csv['name'] = csv['name'].apply(lambda x: int(x))
new_csv['invasive'] = csv['invasive'].apply(lambda x: '%.9f' % float(x))

new_csv.to_csv("/home/jys/Documents/kaggle/invasive_species_monitoring/new_result.csv", index=False)
