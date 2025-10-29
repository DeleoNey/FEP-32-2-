import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad, simpson, trapezoid

df = pd.read_csv("../index_1.csv")
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])
df['money'] = df['money'].astype(float)

#групуємо по хвилинах
data = df.groupby(df['datetime']).agg(
    S=('coffee_name', 'count'),
    Revenue=('money', 'sum')
)
data['price'] = data['Revenue'] / data['S']