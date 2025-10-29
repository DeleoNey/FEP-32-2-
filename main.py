import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad, simpson, trapezoid
#2 Новодворський Роман
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

#  Метод трапецій і Сімпсона
Q_trap = trapezoid(S, t)
R_trap = trapezoid(R_inst, t)

Q_simp = simpson(S, t)
R_simp = simpson(R_inst, t)
# Точніше значення через quad для порівняння - Олег Стецик
Q_quad, _ = quad(lambda x: S_interp(x), t0, t1)
R_quad, _ = quad(lambda x: R_interp(x), t0, t1)

# оцінка похибок - Катело Настя
def err(true, approx):
    return abs(true - approx), abs(true - approx) / true * 100

errors = {
    "Q_rect": err(Q_quad, Q_rect),
    "Q_trap": err(Q_quad, Q_trap),
    "Q_simp": err(Q_quad, Q_simp),
    "R_rect": err(R_quad, R_rect),
    "R_trap": err(R_quad, R_trap),
    "R_simp": err(R_quad, R_simp)
}

print("\nАбсолютні та відносні похибки:")
for k, v in errors.items():
    print(f"{k}: abs={v[0]:.4f}, rel={v[1]:.4f}%")

print("\nQ (quad) =", Q_quad)
print("R (quad) =", R_quad)

#Побудова графіків S(t), p(t), R_inst(t) - Войченко Ігор
plt.plot(t, S, label="S(t) - обсяг продажів")
plt.legend()
plt.grid()
plt.savefig("S(t).png")

plt.figure()
plt.plot(t, p, label="p(t) - ціна товару")
plt.legend()
plt.grid()
plt.savefig("p(t).png")

plt.figure()
plt.plot(t, R_inst, label="R_inst(t) - виторг")
plt.legend()
plt.grid()
plt.savefig("R_inst(t).png")

#Виділення піків продажів + висновок - Коваль Станіслав

peak_threshold = np.percentile(S, 85)
peak_times = data[data['S'] >= peak_threshold]

print("\nІнтервали пікових продажів:")
print(peak_times.index)
print("\n✅ Роботу завершено!")
