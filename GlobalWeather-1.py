import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# set display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# read the csv file
weather_data = pd.read_csv('GlobalWeatherRepository.csv', sep=',', engine='pyarrow')

#Check the DF and the information about the Columns
print(weather_data.head())
#print(weather_data.info())

# count the number of data in different countries
country_count = weather_data.pivot_table(index = ['country'], aggfunc = 'size')
#print(country_count)


air_qualities_us_epa = pd.DataFrame(weather_data.loc[0:,'air_quality_Carbon_Monoxide' : 'air_quality_us-epa-index'])
air_qualities_gb_defra = pd.DataFrame(weather_data.loc[0:,'air_quality_Carbon_Monoxide' : 'air_quality_gb-defra-index'])
air_qualities_gb_defra.drop(['air_quality_us-epa-index'], axis='columns', inplace=True)
print(air_qualities_us_epa.head())
print(air_qualities_gb_defra.head())

#sns.pairplot(air_qualities_us_epa, hue='air_quality_us-epa-index')
#sns.pairplot(air_qualities_gb_defra, hue='air_quality_gb-defra-index')
#plt.show()

air_qualities_gb_defra_feature = air_qualities_gb_defra.drop('air_quality_gb-defra-index', axis=1)
air_qualities_gb_defra_target = air_qualities_gb_defra['air_quality_gb-defra-index']

air_qualities_gb_defra_feature_train, air_qualities_gb_defra_feature_test, air_qualities_gb_defra_target_train,\
    air_qualities_gb_defra_target_test = train_test_split(air_qualities_gb_defra_feature,
                                                        air_qualities_gb_defra_target, test_size=1/3, random_state=12345)



knn = KNeighborsClassifier()
knn.fit(air_qualities_gb_defra_feature_train, air_qualities_gb_defra_target_train)
print(air_qualities_gb_defra.head(10))
print()
print(f"The air_quality_gb_defra: "
      f"{knn.predict(pd.DataFrame([[236.2, 147.3,  52.8 ,  26.9, 139.6, 203.3]], columns=['air_quality_Carbon_Monoxide','air_quality_Ozone', 'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide','air_quality_PM2.5', 'air_quality_PM10']))[0]}")
