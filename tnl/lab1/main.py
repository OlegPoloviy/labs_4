import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Весь попередній код залишається без змін ---

# Матриця та відповідні значення зросту для множини "Висока людина"
high_person_heights = np.array([170, 175, 180, 185, 190, 195, 200])
high_person_matrix = np.array([
    [1, 1/3, 1/5, 1/7, 1/9, 1/9, 1/9],
    [3,   1, 1/3, 1/5, 1/7, 1/9, 1/9],
    [5,   3,   1, 1/3, 1/5, 1/7, 1/9],
    [7,   5,   3,   1, 1/3, 1/5, 1/7],
    [9,   7,   5,   3,   1, 1/3, 1/5],
    [9,   9,   7,   5,   3,   1, 1/3],
    [9,   9,   9,   7,   5,   3,   1]
])

# Матриця та відповідні значення зросту для множини "Низька людина"
low_person_heights = np.array([150, 155, 160, 165, 170, 175, 180])
low_person_matrix = np.array([
    [1,   3,   5,   7,   9,   9,   9],
    [1/3, 1,   3,   5,   7,   9,   9],
    [1/5, 1/3, 1,   3,   5,   7,   9],
    [1/7, 1/5, 1/3, 1,   3,   5,   7],
    [1/9, 1/7, 1/5, 1/3, 1,   3,   5],
    [1/9, 1/9, 1/7, 1/5, 1/3, 1,   3],
    [1/9, 1/9, 1/9, 1/7, 1/5, 1/3, 1]
])


def calculate_membership_vector(matrix):
    row_products = np.prod(matrix, axis=1)
    n = matrix.shape[0]
    geometric_means = np.power(row_products, 1/n)
    priority_vector = geometric_means / np.sum(geometric_means)
    return priority_vector

def s_shaped_func(x, a, b):
    # Додамо захист від занадто великих значень, щоб уникнути помилок
    # Для великих від'ємних значень exp повертає 0, що є коректним
    arg = -a * (x - b)
    return 1 / (1 + np.exp(np.clip(arg, -500, 500)))


def z_shaped_func(x, a, b):
    return 1 - s_shaped_func(x, a, b)


def plot_fuzzy_set(heights, membership_values, title, approx_func):
    """
    Будує графік функції приналежності та її апроксимацію.
    """
    normalized_membership = membership_values / np.max(membership_values)

    # !!! ОНОВЛЕННЯ: Додаємо початкові припущення для параметрів !!!
    initial_guess = [0.1, np.mean(heights)] # a=0.1, b=середнє значення зросту

    # Підбір параметрів з урахуванням початкового припущення
    try:
        params, _ = curve_fit(approx_func, heights, normalized_membership, p0=initial_guess)
        a, b = params
        print(f"Знайдені параметри для '{title}': a={a:.3f}, b={b:.3f}")
    except RuntimeError:
        print(f"Не вдалося знайти параметри для апроксимації для '{title}'. Спробуйте змінити initial_guess.")
        # Якщо підбір не вдався, виходимо, щоб не будувати некоректний графік
        return normalized_membership


    smooth_heights = np.linspace(heights.min(), heights.max(), 300)
    smooth_curve = approx_func(smooth_heights, a, b)

    plt.figure(figsize=(10, 6))
    plt.plot(heights, normalized_membership, 'o', label='Розраховані значення', markersize=8, zorder=10)
    plt.plot(smooth_heights, smooth_curve, '-', label=f'Апроксимація', linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Зріст (см)", fontsize=12)
    plt.ylabel("Ступінь приналежності μ(x)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.show()
    
    return normalized_membership


# --- Виконання розрахунків (залишається без змін) ---

# Обробка множини "Висока людина"
print("--- Нечітка множина 'Висока людина' ---")
mu_high_raw = calculate_membership_vector(high_person_matrix)
mu_high_normalized = plot_fuzzy_set(high_person_heights, mu_high_raw, 
                                    "Функція приналежності для множини 'Висока людина'", 
                                    s_shaped_func)

# Обробка множини "Низька людина"
print("\n--- Нечітка множина 'Низька людина' ---")
mu_low_raw = calculate_membership_vector(low_person_matrix)
mu_low_normalized = plot_fuzzy_set(low_person_heights, mu_low_raw, 
                                   "Функція приналежності для множини 'Низька людина'", 
                                   z_shaped_func)