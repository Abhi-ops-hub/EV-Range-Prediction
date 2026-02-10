import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"EV_Range_Prediction_Dataset_2000.csv")

# data cleaning..deleting the missing data

df=df.dropna(subset=['battery_soc','remaining_range'])

# plotting battery SoC and Remaining range

plt.figure(figsize=(10,8))

plt.scatter(df['battery_soc'],df['remaining_range'])
plt.title('EV-remaining range(KM) VS battery SoC')
plt.xlabel('battery_soc')
plt.ylabel('remainig_range')
plt.legend()
plt.show()