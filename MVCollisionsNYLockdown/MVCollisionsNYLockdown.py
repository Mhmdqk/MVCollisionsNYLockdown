#  Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#  Load the dataset
df = pd.read_csv(r'C:\Users\muhammad.khan\Downloads\Motor_Vehicle_Collisions_-_Crashes.csv')

#  Explore the data
print(df.head())
print(df.info())
print(df.describe())

#  Clean the data 
print(df.isnull().sum())
df = df.drop_duplicates()
# Handle missing values 
# Example: df = df.dropna()

#  Identify Patterns and Trends

## Plotting Time Series Data
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
daily_collisions = df.groupby('CRASH DATE').size()
plt.figure(figsize=(10, 6))
plt.plot(daily_collisions.index, daily_collisions.values)
plt.title('Number of Motor Vehicle Collisions Over Time')
plt.xlabel('Date')
plt.ylabel('Collision Count')

## Highlight the decrease in collisions during March 2020
plt.axvspan('2020-03-01', '2020-04-01', color='red', alpha=0.3, label='Lockdown Period')

## Highlight the increase in collisions from April 1st, 2020, to August 14th, 2020
plt.axvspan('2020-04-01', '2020-08-14', color='green', alpha=0.3, label='Post-Lockdown Increase')

## Add legend
plt.legend()

plt.show()

## Analyzing Seasonality
decomposition = seasonal_decompose(daily_collisions, model='additive', period=365)
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(decomposition.trend)
plt.title('Trend')
plt.subplot(412)
plt.plot(decomposition.seasonal)
plt.title('Seasonality')
plt.subplot(413)
plt.plot(decomposition.resid)
plt.title('Residuals')
plt.tight_layout()
plt.show()

#  Explore the Data by Segmenting or Grouping

## Grouping by Borough
collisions_by_borough = df.groupby('BOROUGH').size()
plt.figure(figsize=(10, 6))
collisions_by_borough.plot(kind='bar')
plt.title('Motor Vehicle Collisions by Borough')
plt.xlabel('Borough')
plt.ylabel('Collision Count')
plt.xticks(rotation=45)
plt.show()

## Segmenting by Time of Day
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'], format='%H:%M').dt.time
df['CRASH HOUR'] = df['CRASH TIME'].apply(lambda x: x.hour)
collisions_by_hour = df.groupby('CRASH HOUR').size()
plt.figure(figsize=(10, 6))
collisions_by_hour.plot(kind='line')
plt.title('Motor Vehicle Collisions by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Collision Count')
plt.xticks(range(0, 24))
plt.grid(True)
plt.show()


