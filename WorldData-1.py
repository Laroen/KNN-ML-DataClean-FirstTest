import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def DataChange(world_data):
    for i in world_data:
        if world_data[i].dtypes == 'float64':
            continue
        else:
            if '%' in world_data[i].values[0]:
                world_data[i] = world_data[i].replace('%', '', regex=True)
                world_data[i] = world_data[i].astype('float64')
            elif ',' in world_data[i].values[0] and '$' in world_data[i].values[0]:
                world_data[i] = world_data[i].str.replace(',', '')
                world_data[i] = world_data[i].str.replace('$', '').astype(np.float64)
            elif ',' in world_data[i].values[0]:
                world_data[i] = world_data[i].replace(',', '', regex=True)
                world_data[i] = world_data[i].astype(np.float64)
            elif '$' in world_data[i].values[0]:
                world_data[i] = world_data[i].str.replace('$', '').astype(np.float64)
    return world_data


all_files = [file for file in os.listdir('./')] # check all files in the base library and save names in a list
#print(all_files)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


world_data = pd.read_csv('world-data-2023.csv', sep=',', engine='pyarrow')

world_data = world_data.replace('', np.nan, regex=True)
world_data.fillna(0,inplace=True)
'''
column_names = world_data.columns
column_names = [x for x in column_names if '%' in x]
'''
#print(world_data.head())

world_data = DataChange(world_data)

world_data.fillna(0,inplace=True)


#print(world_data.dtypes)
#print(world_data.isna().any())  # checked that still have any NaN


world_data['Country_new'] = world_data['Country'].astype('string') + ' ' + '('+ world_data['Abbreviation'].astype('string') + ')'
print(world_data.head())
test_1 = pd.DataFrame(world_data.set_index('Country_new'))
test_1.drop(['Country','Abbreviation'],axis='columns', inplace=True)  # delete the useless columns

#print(test_1.head())


#language with population,


reg_data = world_data[['Land Area(Km2)','Armed Forces size', 'GDP', 'Population']]
'''
sns.pairplot(reg_data, hue='Armed Forces size')
plt.show()
'''
reg_data_feature = reg_data.drop('Armed Forces size', axis=1)
reg_data_target = reg_data['Armed Forces size']

reg_data_feature_train, reg_data_feature_test, reg_data_target_train, reg_data_target_test = train_test_split(reg_data_feature, reg_data_target, test_size=1/3, random_state=12345)



knn = KNeighborsClassifier()
knn.fit(reg_data_feature_train, reg_data_target_train)
print(reg_data.head(20))
print()
print(f"Armed Forces size with the followed datas ( Land Area: 90000, GDP: 9.463542e+10, Population: 10000000): "
      f"{knn.predict(pd.DataFrame([[90000, 9.463542e+10, 10000000]], columns=['Land Area(Km2)', 'GDP', 'Population']))[0]}")

#print(language.groupby('Official language').mean())