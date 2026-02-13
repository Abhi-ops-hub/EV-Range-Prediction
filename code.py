import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"EV_Range_Prediction_Dataset_2000.csv")

# data cleaning..deleting the missing data

df=df.dropna(subset=['battery_soc','power_consumption','speed','remaining_range','regen_braking'])

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

plt.savefig("Top power consumption at speed.png",dpi=300)

plt.grid()
plt.show()
# For Power consumption vs Road Gradient
plt.figure(figsize=(8,6))
plt.scatter(df['road_gradient'], df['power_consumption'])
plt.xlabel("Road Gradient (%)")
plt.ylabel("Power Consumption")
plt.title("Effect of Road Gradient on Power Consumption")
plt.savefig("Power consumption at road gradient.png",dpi=300)
plt.show()

# top power consumption by driving style
print(df['driving_style'])

plt.figure(figsize=(8,6))
df.groupby('driving_style')['power_consumption'].max().plot(kind='bar',color=["r","g","k"])
plt.xlabel("Driving Style")
plt.ylabel("Max Power Consumption")
plt.title("Maximum Power Consumption by Driving Style")
plt.savefig("power consumption by driving style.png",dpi=300)
plt.show()

# traffic density impact on Range
print(df['traffic_density'])
plt.figure(figsize=(8,6))
df.groupby('traffic_density')['remaining_range'].mean().plot(kind='bar')
plt.xlabel("Traffic Density")
plt.ylabel("Average Remaining Range")
plt.title("Traffic Density vs Remaining Range")
plt.savefig("Range impact by traffic density.png",dpi=300)
plt.show()

# Regerative braking vs range
plt.figure(figsize=(8,6))
plt.scatter(df['regen_braking'],df['remaining_range'])
plt.xlabel("Regenrative_braking")
plt.ylabel("Remaining_range")
plt.title("Remianing_range vs Regenrative braking")
plt.savefig("regenerative braking vs remaining_range")
plt.show()

# time series range over time
df['timestamp'] = pd.to_datetime(df['timestamp'])

plt.figure(figsize=(8,6))
plt.plot(df['timestamp'], df['remaining_range'])
plt.xlabel("Time")
plt.ylabel("Remaining Range (km)")
plt.title("Remaining Range Over Time")
plt.savefig("time series over time")
plt.show()

# temperature vs remaining range
plt.figure(figsize=(8,6))
plt.scatter(df['temperature'], df['remaining_range'])
plt.xlabel("Temperature (°C)")
plt.ylabel("Remaining Range (km)")
plt.title("Temperature vs Remaining Range")
plt.savefig("Temperature vs remaining range")
plt.show()

# traffic density impact on range
plt.figure(figsize=(8,6))
df.groupby('traffic_density')['remaining_range'].mean().plot(kind='bar')
plt.xlabel("Traffic Density")
plt.ylabel("Average Remaining Range")
plt.title("Traffic Density vs Remaining Range")
plt.savefig("Traffic density impact")
plt.show()


