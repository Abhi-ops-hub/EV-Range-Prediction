import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"EV_Range_Prediction_Dataset_2000.csv")

# data cleaning..deleting the missing data

df=df.dropna(subset=['battery_soc','power_consumption','speed','remaining_range'])

# plotting battery SoC and Remaining range

plt.figure(figsize=(10,8))

plt.scatter(df['battery_soc'],df['remaining_range'])
plt.title('EV-remaining range(KM) VS battery SoC')
plt.xlabel('battery_soc')
plt.ylabel('remainig_range')
plt.savefig("soc vs remaining range.png",dpi=300)
plt.show()

# now for power consumption

plt.figure(figsize=(8,6))
top_power=df.sort_values('power_consumption',ascending=False).head(20)
plt.scatter(top_power['speed'],top_power['power_consumption'],color="k",marker="o")
plt.xlabel("speed")
plt.ylabel("Power consumption")
plt.title("Top 20 Power Consumption at speed")

plt.xticks()

plt.grid()
plt.show()