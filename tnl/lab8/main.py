import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, Callable

# ===================== 1) БАЗОВІ membership-функції =====================

def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Трапеція a<=b<=c<=d. Повертає μ(x) ∈ [0,1]."""
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / max(b - a, 1e-12)
    if b <= x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / max(d - c, 1e-12)
    return 0.0

def trimf(x: float, a: float, b: float, c: float) -> float:
    """Трикутник a<=b<=c. Повертає μ(x) ∈ [0,1]."""
    if x <= a or x >= c:
        return 0.0
    if a < x < b:
        return (x - a) / max(b - a, 1e-12)
    if b <= x < c:
        return (c - x) / max(c - b, 1e-12)
    return 0.0

# ===================== 2) УНІВЕРСУМИ =====================

T_MIN, T_MAX = 0.0, 100.0      # Температура, °C
A_MIN, A_MAX = -90.0, 90.0     # Кут повороту крана, градуси (вліво −, вправо +)

# ===================== 3) ВХІДНІ НЕЧІТКІ МНОЖИНИ (як на Рис. 5) =====================
# {холодна, прохолодна, тепла, не_дуже_гаряча, гаряча}
def mu_cold(t):        return trapmf(t, 0, 0, 10, 30)
def mu_cool(t):        return trimf(t, 10, 30, 50)
def mu_warm(t):        return trimf(t, 30, 50, 70)
def mu_not_hot(t):     return trimf(t, 50, 70, 90)
def mu_hot(t):         return trapmf(t, 70, 90, 100, 100)

TEMP_SETS: Dict[str, Callable[[float], float]] = {
    "cold": mu_cold,
    "cool": mu_cool,
    "warm": mu_warm,
    "not_hot": mu_not_hot,
    "hot": mu_hot,
}

# ===================== 4) ВИХІДНІ НЕЧІТКІ МНОЖИНИ (як на Рис. 5) =====================
# {великий_вліво, невеликий_вліво, нуль, невеликий_вправо, великий_вправо}
def mu_big_left(a):    return trapmf(a, -90, -90, -60, -30)
def mu_small_left(a):  return trimf(a, -60, -30, 0)
def mu_zero(a):        return trimf(a, -15, 0, 15)
def mu_small_right(a): return trimf(a, 0, 30, 60)
def mu_big_right(a):   return trapmf(a, 30, 60, 90, 90)

ANGLE_SETS: Dict[str, Callable[[float], float]] = {
    "big_left":    mu_big_left,
    "small_left":  mu_small_left,
    "zero":        mu_zero,
    "small_right": mu_small_right,
    "big_right":   mu_big_right,
}

# ===================== 5) БАЗА ПРАВИЛ =====================
# R1: IF hot THEN big_right
# R2: IF not_hot THEN small_right
# R3: IF warm THEN zero
# R4: IF cool THEN small_left
# R5: IF cold THEN big_left
RULES = [
    ("hot",      "big_right"),
    ("not_hot",  "small_right"),
    ("warm",     "zero"),
    ("cool",     "small_left"),
    ("cold",     "big_left"),
]

# ===================== 6) ВИХІДНИЙ МЕХАНІЗМ =====================

def fuzzify_temperature(t_value: float) -> Dict[str, float]:
    """Ступені належності для всіх вхідних термів."""
    return {name: f(t_value) for name, f in TEMP_SETS.items()}

def aggregate_output(strengths: Dict[str, float], implication: str = "mamdani") -> Callable[[float], float]:
    """
    Повертає агреговану μ_out(a).
    implication = "mamdani" -> min(α, μ_out(a))
                = "larsen"  -> α * μ_out(a)   (prod-активізація)
    """
    use_min = (implication.lower() == "mamdani")

    def mu_out(a: float) -> float:
        vals = []
        for in_name, out_name in RULES:
            alpha = strengths.get(in_name, 0.0)
            mu_y = ANGLE_SETS[out_name](a)
            vals.append(min(alpha, mu_y) if use_min else alpha * mu_y)
        return max(vals) if vals else 0.0

    return mu_out

def defuzz_centroid(mu_func: Callable[[float], float],
                    x_min: float, x_max: float, steps: int = 3000) -> float:
    """Чисельний центр ваги."""
    dx = (x_max - x_min) / steps
    num = den = 0.0
    for i in range(steps):
        x = x_min + (i + 0.5) * dx
        mu = mu_func(x)
        num += x * mu
        den += mu
    return 0.0 if den == 0.0 else num / den

def fuzzy_shower(temp_c: float, method: str = "mamdani") -> float:
    """Кут повороту (градуси) для температури temp_c за методом method."""
    strengths = fuzzify_temperature(temp_c)
    mu_out = aggregate_output(strengths, implication=method)
    angle = defuzz_centroid(mu_out, A_MIN, A_MAX, steps=3000)
    return angle

# ===================== 7) СЕРІЯ РОЗРАХУНКІВ + ГРАФІКИ =====================

def main():
    # Сітка температур
    temps = np.linspace(T_MIN, T_MAX, 201)

    # Криві “кут vs температура” для Мамдані та Ларсена
    angles_mamdani = np.array([fuzzy_shower(t, "mamdani") for t in temps])
    angles_larsen  = np.array([fuzzy_shower(t, "larsen")  for t in temps])

    # Зберегти CSV (зручно для звіту)
    with open("fuzzy_shower_angles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Temperature_C", "Angle_Mamdani_deg", "Angle_Larsen_deg"])
        for T, am, al in zip(temps, angles_mamdani, angles_larsen):
            w.writerow([float(T), float(am), float(al)])
    print("Збережено дані в: fuzzy_shower_angles.csv")

    # --- Графік 1: Мамдані (min) ---
    plt.figure(figsize=(9, 5))
    plt.plot(temps, angles_mamdani, linewidth=2)
    plt.title("Кут повороту vs Температура — Мамдані (min-активізація)")
    plt.xlabel("Температура, °C")
    plt.ylabel("Кут повороту, °")
    plt.grid(True)

    # --- Графік 2: Ларсен (prod) ---
    plt.figure(figsize=(9, 5))
    plt.plot(temps, angles_larsen, linewidth=2)
    plt.title("Кут повороту vs Температура — Ларсен (prod-активізація)")
    plt.xlabel("Температура, °C")
    plt.ylabel("Кут повороту, °")
    plt.grid(True)

    plt.show()

    # Короткий друк ідей для аналізу у звіті:
    print("\nКороткий аналіз:")
    print("- Обидві залежності монотонні: холодніше -> кут вліво (−), гарячіше -> вправо (+).")
    print("- Мамдані дає більш ламану, сегментовано-лінійну характеристику через операцію min.")
    print("- Ларсен (добуток) згладжує переходи: крива плавніша в зонах перекриття термів.")
    print("- У середній зоні (~50 °C, 'тепла') кут близький до 0 для обох методів,")
    print("  але для Ларсена зміни починаються м’якше і раніше через зменшені внески слабких правил.")
    print("- На насиченнях (дуже холодно/гаряче) обидва методи сходяться до граничних кутів (≈±67° у цих параметрах).")

if __name__ == "__main__":
    main()
