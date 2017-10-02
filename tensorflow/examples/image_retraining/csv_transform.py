import sys
import os.path

import pandas as pd


# csv0 = pd.read_csv("/home/jys/Documents/kaggle/invasive_species_monitoring/"
#                   "train_labels.csv",
#                    names = ['name', 'invasive'], header=0)
# print(csv0)
# ls = [x + 1 for x in csv0[ abs(float(csv0['invasive']) - 0.5) < 0.2]].index.tolist()
#
# print(len(ls))

csv1 = pd.read_csv("/home/jys/Documents/kaggle/invasive_species_monitoring/"
                  "results/new_result.csv",
                   names = ['name', 'invasive'], skiprows=1, header=None)


csv2 = pd.read_csv("/home/jys/Documents/kaggle/invasive_species_monitoring/"
                  "results/output_graph_sigmod_weight_1_dropout_06_adam.csv",
                   names = ['name', 'invasive'], skiprows=1, header=None)


# csv1['invasive'] = csv1['invasive'].apply(lambda x: round(x))
# csv2['invasive'] = csv2['invasive'].apply(lambda x: round(x))


# csv['invasive'][csv['invasive'] < 0.4] = 0

# csv['invasive'] = csv['invasive'].apply(lambda x:   0.0 if x < 0.3 else x)

# csv_transform = pd.DataFrame()

# csv_transform['name'] = csv['name']
# csv_transform['invasive'] =

# csv.to_csv("/home/jys/Documents/kaggle/invasive_species_monitoring/result_xiaoyu_1st.csv", index=False)

# print([x + 1 for x in csv[ abs(csv['invasive'] - 0.5) < 0.2].index.tolist()])

# print(csv.loc[abs(csv['invasive'] - 0.5) < 0.2])

csv_comparison = pd.DataFrame()
csv_comparison['name'] = csv1['name']
csv_comparison['invasive_1'] = csv1['invasive']
csv_comparison['invasive_2'] = csv2['invasive']
csv_comparison['invasive_1_round'] = csv1['invasive'].apply(lambda x: round(x))
csv_comparison['invasive_2_round'] = csv2['invasive'].apply(lambda x: round(x))


# csv_comparison['invasive_03'] = csv_comparison.loc[abs(csv['invasive'] - 0.5) < 0.2]
print(csv_comparison.loc[csv_comparison['invasive_1_round'] != csv_comparison['invasive_2_round']])


#
# new_csv = pd.DataFrame()
#
# print(csv)
#
# new_csv['name'] = csv['name'].apply(lambda x: int(x))
# new_csv['invasive'] = csv['invasive'].apply(lambda x: '%.9f' % float(x))
#
# new_csv.to_csv("/home/jys/Documents/kaggle/invasive_species_monitoring/new_result.csv", index=False)
