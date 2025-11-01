import numpy as np
import matplotlib.pyplot as plt

# --- Початкові дані з попереднього завдання ---

# 1. Визначення універсуму
universe = np.arange(150, 201, 1)

# 2. Визначення функцій належності
def low_person_membership(x):
    if x <= 150: return 1
    elif 150 < x < 175: return (175 - x) / 25
    else: return 0

def high_person_membership(x):
    if x <= 175: return 0
    elif 175 < x < 200: return (x - 175) / 25
    else: return 1

# Розрахунок значень функцій належності для всього універсуму
membership_A = np.array([low_person_membership(x) for x in universe])
membership_B = np.array([high_person_membership(x) for x in universe])

def medium_person_membership(x):
    if 165 <= x <= 175:
        return (x - 165) / 10
    elif 175 < x <= 185:
        return (185 - x) / 10
    else:
        return 0

# Створення масивів значень для кожної множини
membership_A = np.array([low_person_membership(x) for x in universe])
membership_B = np.array([high_person_membership(x) for x in universe])
membership_C = np.array([medium_person_membership(x) for x in universe])

def medium_person_membership(x):
    if 165 <= x <= 175:
        return (x - 165) / 10
    elif 175 < x <= 185:
        return (185 - x) / 10
    else:
        return 0

# Створення масивів значень для кожної множини
membership_A = np.array([low_person_membership(x) for x in universe])
membership_B = np.array([high_person_membership(x) for x in universe])
membership_C = np.array([medium_person_membership(x) for x in universe])

# --- Реалізація операцій над нечіткими множинами ---

# Доповнення (NOT A)
complement_A = 1 - membership_A

# Перетин (A AND B)
intersection_AB = np.minimum(membership_A, membership_B)

# Об'єднання (A OR B)
union_AB = np.maximum(membership_A, membership_B)

# Різниця (A \ B)
difference_A_B = np.minimum(membership_A, 1 - membership_B)

# Симетрична різниця (A Δ B)
sym_difference_AB = np.maximum(difference_A_B, np.minimum(membership_B, 1 - membership_A))

# Концентрування (для множини A)
concentration_A = membership_A**2

# Розтягування (для множини A)
dilation_A = membership_A**0.5

# --- Візуалізація результатів ---

# Створюємо фігуру з кількома графіками
fig, axs = plt.subplots(4, 2, figsize=(15, 20))
fig.suptitle('Операції над нечіткими множинами A={Низька людина} та B={Висока людина}', fontsize=16)

# Функція для налаштування графіків
def plot_fuzzy_set(ax, x, y, title, labels):
    if not isinstance(y, list):
        y = [y]
    if not isinstance(labels, list):
        labels = [labels]
    
    for i in range(len(y)):
        ax.plot(x, y[i], label=labels[i])
    
    ax.set_title(title)
    ax.set_xlabel('Зріст (см)')
    ax.set_ylabel('Ступінь належності')
    ax.grid(True)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

# 1. Початкові множини
plot_fuzzy_set(axs[0, 0], universe, [membership_A, membership_B], 'Початкові множини', ['A: Низька людина', 'B: Висока людина'])

# 2. Доповнення
plot_fuzzy_set(axs[0, 1], universe, [membership_A, complement_A], 'Доповнення (NOT A)', ['A', 'NOT A'])

# 3. Перетин
plot_fuzzy_set(axs[1, 0], universe, [membership_A, membership_B, intersection_AB], 'Перетин (A ∩ B)', ['A', 'B', 'A ∩ B'])

# 4. Об'єднання
plot_fuzzy_set(axs[1, 1], universe, [membership_A, membership_B, union_AB], 'Об\'єднання (A ∪ B)', ['A', 'B', 'A ∪ B'])

# 5. Різниця
plot_fuzzy_set(axs[2, 0], universe, [membership_A, membership_B, difference_A_B], 'Різниця (A \\ B)', ['A', 'B', 'A \\ B'])

# 6. Симетрична різниця
plot_fuzzy_set(axs[2, 1], universe, [membership_A, membership_B, sym_difference_AB], 'Симетрична різниця (A Δ B)', ['A', 'B', 'A Δ B'])

# 7. Концентрування
plot_fuzzy_set(axs[3, 0], universe, [membership_A, concentration_A], 'Концентрування CON(A)', ['A', 'CON(A) = A^2'])

# 8. Розтягування
plot_fuzzy_set(axs[3, 1], universe, [membership_A, dilation_A], 'Розтягування DIL(A)', ['A', 'DIL(A) = A^0.5'])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

membership_A = np.array([low_person_membership(x) for x in universe]) # "Низька людина"
membership_B = np.array([high_person_membership(x) for x in universe]) # "Висока людина"

# --- Розрахунок функцій належності для висловлювань ---

# 1. “Низька і висока людина” (Перетин)
low_and_high = np.minimum(membership_A, membership_B)

# 2. “Низька або висока людина” (Об'єднання)
low_or_high = np.maximum(membership_A, membership_B)

# 3. “Не висока людина” (Доповнення)
not_high = 1 - membership_B

# 4. “Злегка низька людина” (Розтягування)
slightly_low = membership_A**0.5

# 5. “Дуже висока людина” (Концентрування)
very_high = membership_B**2

# --- Візуалізація результатів ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(3, 2, figsize=(14, 18))
fig.suptitle('Графіки функцій належності для лінгвістичних висловлювань', fontsize=18)

# Графік 1: Низька і висока людина
axs[0, 0].plot(universe, membership_A, 'b--', alpha=0.5, label='A: Низька')
axs[0, 0].plot(universe, membership_B, 'g--', alpha=0.5, label='B: Висока')
axs[0, 0].plot(universe, low_and_high, 'r-', linewidth=2, label='Результат: "Низька і висока"')
axs[0, 0].set_title('"Низька і висока людина" (A ∩ B)')
axs[0, 0].legend()
axs[0, 0].set_ylim(-0.05, 1.05)

# Графік 2: Низька або висока людина
axs[0, 1].plot(universe, membership_A, 'b--', alpha=0.5, label='A: Низька')
axs[0, 1].plot(universe, membership_B, 'g--', alpha=0.5, label='B: Висока')
axs[0, 1].plot(universe, low_or_high, 'r-', linewidth=2, label='Результат: "Низька або висока"')
axs[0, 1].set_title('"Низька або висока людина" (A ∪ B)')
axs[0, 1].legend()
axs[0, 1].set_ylim(-0.05, 1.05)

# Графік 3: Не висока людина
axs[1, 0].plot(universe, membership_B, 'g--', alpha=0.5, label='B: Висока')
axs[1, 0].plot(universe, not_high, 'r-', linewidth=2, label='Результат: "Не висока"')
axs[1, 0].set_title('"Не висока людина" (NOT B)')
axs[1, 0].legend()
axs[1, 0].set_ylim(-0.05, 1.05)

# Графік 4: Злегка низька людина
axs[1, 1].plot(universe, membership_A, 'b--', alpha=0.5, label='A: Низька')
axs[1, 1].plot(universe, slightly_low, 'r-', linewidth=2, label='Результат: "Злегка низька"')
axs[1, 1].set_title('"Злегка низька людина" (DIL(A))')
axs[1, 1].legend()
axs[1, 1].set_ylim(-0.05, 1.05)

# Графік 5: Дуже висока людина
axs[2, 0].plot(universe, membership_B, 'g--', alpha=0.5, label='B: Висока')
axs[2, 0].plot(universe, very_high, 'r-', linewidth=2, label='Результат: "Дуже висока"')
axs[2, 0].set_title('"Дуже висока людина" (CON(B))')
axs[2, 0].legend()
axs[2, 0].set_ylim(-0.05, 1.05)

# Ховаємо порожній графік
fig.delaxes(axs[2,1])

# Додаємо загальні підписи для осей
for ax in axs.flat:
    ax.set(xlabel='Зріст (см)', ylabel='Ступінь належності')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Перевірка властивостей ---

print("Перевірка основних властивостей операцій над нечіткими множинами:\n")

# 1. Комутативність: A ∪ B = B ∪ A та A ∩ B = B ∩ A
# Об'єднання
lhs_union = np.maximum(membership_A, membership_B)
rhs_union = np.maximum(membership_B, membership_A)
print(f"Комутативність (об'єднання): A ∪ B == B ∪ A. Результат: {np.allclose(lhs_union, rhs_union)}")
# Перетин
lhs_intersect = np.minimum(membership_A, membership_B)
rhs_intersect = np.minimum(membership_B, membership_A)
print(f"Комутативність (перетин):   A ∩ B == B ∩ A. Результат: {np.allclose(lhs_intersect, rhs_intersect)}\n")


# 2. Асоціативність: (A ∪ B) ∪ C = A ∪ (B ∪ C) та (A ∩ B) ∩ C = A ∩ (B ∩ C)
# Об'єднання
lhs_union = np.maximum(np.maximum(membership_A, membership_B), membership_C)
rhs_union = np.maximum(membership_A, np.maximum(membership_B, membership_C))
print(f"Асоціативність (об'єднання): (A ∪ B) ∪ C == A ∪ (B ∪ C). Результат: {np.allclose(lhs_union, rhs_union)}")
# Перетин
lhs_intersect = np.minimum(np.minimum(membership_A, membership_B), membership_C)
rhs_intersect = np.minimum(membership_A, np.minimum(membership_B, membership_C))
print(f"Асоціативність (перетин):   (A ∩ B) ∩ C == A ∩ (B ∩ C). Результат: {np.allclose(lhs_intersect, rhs_intersect)}\n")


# 3. Дистрибутивність: A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)
lhs = np.minimum(membership_A, np.maximum(membership_B, membership_C))
rhs = np.maximum(np.minimum(membership_A, membership_B), np.minimum(membership_A, membership_C))
print(f"Дистрибутивність: A ∩ (B ∪ C) == (A ∩ B) ∪ (A ∩ C). Результат: {np.allclose(lhs, rhs)}\n")


# 4. Інволюція (закон подвійного заперечення): NOT(NOT A) = A
lhs = 1 - (1 - membership_A)
rhs = membership_A
print(f"Інволюція: NOT(NOT A) == A. Результат: {np.allclose(lhs, rhs)}\n")


# 5. Закони де Моргана
# Закон 1: NOT(A ∪ B) = (NOT A) ∩ (NOT B)
lhs = 1 - np.maximum(membership_A, membership_B)
rhs = np.minimum(1 - membership_A, 1 - membership_B)
print(f"Закон де Моргана 1: NOT(A ∪ B) == (NOT A) ∩ (NOT B). Результат: {np.allclose(lhs, rhs)}")

# Закон 2: NOT(A ∩ B) = (NOT A) ∪ (NOT B)
lhs = 1 - np.minimum(membership_A, membership_B)
rhs = np.maximum(1 - membership_A, 1 - membership_B)
print(f"Закон де Моргана 2: NOT(A ∩ B) == (NOT A) ∪ (NOT B). Результат: {np.allclose(lhs, rhs)}\n")