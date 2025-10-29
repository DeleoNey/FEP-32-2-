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

#3 - Формування часової сітки - Юра Марчак

data = data.sort_index()
t = (data.index - data.index[0]).total_seconds() / 3600  # години з початку
data['t'] = t.values

S = data['S'].values
p = data['price'].values
R_inst = p * S  # миттєвий виторг

t = data['t'].values
t0, t1 = t.min(), t.max()

# Інтерполяція функцій S(t) та p(t) - Шаповалов Олександр

S_interp = interp1d(t, S, kind='cubic')
p_interp = interp1d(t, p, kind='cubic')
R_interp = lambda x: p_interp(x) * S_interp(x)

# Обчислення інтегралів методами прямої формули - Припотнюк Влад

Q_rect = np.sum(S[:-1] * np.diff(t))
R_rect = np.sum(R_inst[:-1] * np.diff(t))


