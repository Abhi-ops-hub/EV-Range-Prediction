# to check everything is working
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"EV_Range_Prediction_Dataset_2000.csv")

print(df.head(10))

f=df['timestamp'].head(5)
print(f)