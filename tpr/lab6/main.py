import numpy as np
from scipy.stats import norm

def calculate_thurstone_scale(alternatives, comparisons, num_experts):
    """
    Розраховує шкальні оцінки за методом Терстоуна (Випадок V).

    Args:
        alternatives (list): Список назв альтернатив.
        comparisons (dict): Словник з результатами порівнянь.
                            Ключ - кортеж (альтернатива_i, альтернатива_j).
                            Значення - кількість експертів, що обрали i кращою за j.
        num_experts (int): Загальна кількість експертів.

    Returns:
        dict: Словник зі шкальними оцінками (математичним сподіванням) для кожної альтернативи.
    """
    num_alternatives = len(alternatives)
    alt_to_idx = {name: i for i, name in enumerate(alternatives)}

    # 1. Створення матриці частот (F)
    freq_matrix = np.zeros((num_alternatives, num_alternatives))
    for (alt_i, alt_j), count in comparisons.items():
        i, j = alt_to_idx[alt_i], alt_to_idx[alt_j]
        freq_matrix[i, j] = count
        # Симетрична частина: скільки разів j виграла у i
        freq_matrix[j, i] = num_experts - count

    print("--- 1. Матриця частот (F) ---")
    print("Скільки разів альтернатива в рядку була кращою за альтернативу в стовпці:")
    print(freq_matrix)
    print("-" * 35)

    # 2. Створення матриці пропорцій (P)
    # Щоб уникнути пропорцій 0 або 1, які дають -inf/+inf Z-оцінку,
    # використовується поправка. Якщо це станеться, можна додати 0.5 до всіх частот.
    # В нашому прикладі таких значень немає, тому поправка не потрібна.
    prop_matrix = freq_matrix / num_experts

    print("--- 2. Матриця пропорцій (P) ---")
    print("Частка експертів, які вважають альтернативу в рядку кращою:")
    print(np.round(prop_matrix, 3))
    print("-" * 35)

    # 3. Створення матриці Z-оцінок (Z)
    # norm.ppf - це обернена функція кумулятивного розподілу (пробіт-функція)
    # Ігноруємо попередження для діагональних елементів, де P=0.5, Z=0.
    with np.errstate(divide='ignore'):
        z_matrix = norm.ppf(prop_matrix)
    
    # Замінюємо нескінченні значення, якщо вони виникли
    z_matrix[z_matrix == np.inf] = 2 # Апроксимація для P=1
    z_matrix[z_matrix == -np.inf] = -2 # Апроксимація для P=0
    np.fill_diagonal(z_matrix, 0) # Діагональ завжди 0

    print("--- 3. Матриця Z-оцінок (Z) ---")
    print("На скільки стандартних відхилень цінність альтернативи в рядку більша:")
    print(np.round(z_matrix, 3))
    print("-" * 35)

    # 4. Розрахунок шкальних оцінок (математичне сподівання)
    # Усереднюємо значення по стовпцях
    scale_values = np.mean(z_matrix, axis=0)

    # 5. Нормалізація шкали (зсув до 0)
    normalized_scale_values = scale_values - np.min(scale_values)
    
    # Формування результату
    result = {name: value for name, value in zip(alternatives, normalized_scale_values)}
    
    return result


# --- Приклад використання для нашого 4-го набору ---

# Наші альтернативи
alternatives = ["D1", "D2", "D3", "D4", "D5"]

# Загальна кількість експертів
num_experts = 3

# Вводимо дані попарних порівнянь на основі аналізу з попереднього завдання.
# Наприклад, для пари (D1, D2): 2 експерти обрали D1, 1 експерт обрав D2.
# Отже, 'D1' виграла у 'D2' 2 рази.
comparisons_data = {
    ('D1', 'D2'): 2,
    ('D1', 'D3'): 3,
    ('D1', 'D4'): 3,
    ('D1', 'D5'): 2,
    ('D2', 'D3'): 2,
    ('D2', 'D4'): 3,
    ('D2', 'D5'): 2,
    ('D3', 'D4'): 3,
    ('D3', 'D5'): 2,
    ('D5', 'D4'): 3,
}

# Виклик функції та отримання результатів
final_scores = calculate_thurstone_scale(alternatives, comparisons_data, num_experts)

# Виведення фінального результату
print("--- 4. Фінальні шкальні оцінки (математичне сподівання) ---")
print("Оцінки нормалізовані, де найнижчий пріоритет має значення 0.")

# Сортуємо результат для наочності
sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

for name, score in sorted_scores:
    print(f"{name}: {score:.4f}")