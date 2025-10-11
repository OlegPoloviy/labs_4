import matplotlib.pyplot as plt
import math
import numpy as np


def factorial(num):
    """Обчислює факторіал числа."""
    if num < 0:
        raise ValueError("Факторіал не визначений для від'ємних чисел")
    if num == 0:
        return 1
    result = 1
    for i in range(1, num + 1):
        result *= i
    return result

def combinations(n, k):
    """Обчислює кількість комбінацій C(n, k)."""
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))

def binomial_pmf(k, n, p):
    """Обчислює ймовірність для біномінального розподілу (PMF)."""
    q = 1 - p
    comb = combinations(n, k)
    probability = comb * (p**k) * (q**(n-k))
    return probability


def poisson_pmf(k, lam):
    """
    Обчислює ймовірність для розподілу Пуассона (Probability Mass Function).
    k: кількість подій
    lam (λ): середнє число подій
    """
    # Реалізація формули: (λ^k * e^(-λ)) / k!
    
    # Використовуємо math.exp для e^(-λ) для кращої точності
    prob = (math.pow(lam, k) * math.exp(-lam)) / factorial(k)
    return prob


def normal_pdf(x, mean, std_dev):
    """
    Обчислює значення функції щільності ймовірності (PDF) для нормального розподілу.
    x: точка, для якої робиться розрахунок
    mean: середнє значення (a або μ)
    std_dev: стандартне відхилення (σ)
    """
    # Коефіцієнт перед експонентою: 1 / (σ * sqrt(2π))
    coefficient = 1.0 / (std_dev * math.sqrt(2 * math.pi))
    
    # Степенева частина експоненти: -(x-μ)² / (2σ²)
    exponent = -((x - mean)**2) / (2 * std_dev**2)
    
    # Повертаємо результат
    return coefficient * math.exp(exponent)

def pareto_pdf(x, x0, alpha):
    """
    Обчислює значення функції щільності ймовірності (PDF) для розподілу Парето.
    x: точка, для якої робиться розрахунок
    x0: параметр масштабу (мінімальне значення)
    alpha: параметр форми
    """
    # Розподіл визначений тільки для x >= x0
    if x < x0:
        return 0
    else:
        # Реалізація формули: (α * x₀^α) / x^(α+1)
        return (alpha * (x0**alpha)) / (x**(alpha + 1))
    

def student_t_pdf(x, alpha):
    """
    Обчислює значення функції щільності ймовірності (PDF) для t-розподілу Стьюдента.
    x: точка, для якої робиться розрахунок
    alpha: кількість ступенів свободи
    """
    # Гамма-функція від ((α+1)/2)
    gamma_num = math.gamma((alpha + 1) / 2)
    
    # Гамма-функція від (α/2)
    gamma_den = math.gamma(alpha / 2)
    
    # Коефіцієнт перед степенем
    coefficient = gamma_num / (gamma_den * math.sqrt(alpha * math.pi))
    
    # Степенева частина
    power_term = (1 + (x**2) / alpha)**(- (alpha + 1) / 2)
    
    return coefficient * power_term


# Вхідні дані
N = 9

print("--- РОЗПОДІЛ БЕРНУЛЛІ ---")

# Розрахунок параметрів
p_bern = 1 / (N + 1)
q_bern = 1 - p_bern

# Виведення параметрів
print(f"Номер варіанта N: {N}")
print(f"Параметр p: {p_bern:.2f}")
print(f"Математичне сподівання E(X): {p_bern:.2f}")
print(f"Дисперсія D(X): {p_bern * q_bern:.2f}")
print("-" * 30)

# Дані для графіка
x_bern_values = [0, 1]
y_bern_probs = [q_bern, p_bern]

# Створення першого вікна для графіка
plt.figure(1, figsize=(8, 6))

# Побудова графіка
bars = plt.bar(x_bern_values, y_bern_probs, color=['#1f77b4', '#ff7f0e'], width=0.4, edgecolor='black')

# Оформлення графіка Бернуллі
plt.title(f'Графік розподілу Бернуллі\nдля N={N} (p={p_bern:.2f})', fontsize=14)
plt.xlabel('Значення випадкової величини (x)', fontsize=12)
plt.ylabel('Ймовірність P(X=x)', fontsize=12)
plt.xticks([0, 1], labels=['0 (Невдача)', '1 (Успіх)'])
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Додавання підписів над стовпцями
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# --- БЛОК 3: БІНОМІНАЛЬНИЙ РОЗПОДІЛ ---

print("\n--- БІНОМІНАЛЬНИЙ РОЗПОДІЛ (самописний) ---")

# Розрахунок параметрів
p_binom = 1 / (N + 1)
n_binom = N + 2

# Виведення параметрів
print(f"Параметр n (кількість випробувань): {n_binom}")
print(f"Параметр p (ймовірність успіху): {p_binom:.2f}")
print(f"Математичне сподівання E(X): {n_binom * p_binom:.2f}")
print(f"Дисперсія D(X): {n_binom * p_binom * (1-p_binom):.2f}")
print("-" * 30)

# Розрахунок ймовірностей для графіка
x_binom_values = list(range(n_binom + 1))
y_binom_probs = []
for k in x_binom_values:
    prob = binomial_pmf(k, n_binom, p_binom)
    y_binom_probs.append(prob)

# Створення другого вікна для графіка
plt.figure(2, figsize=(12, 7))

# Побудова графіка
plt.bar(x_binom_values, y_binom_probs, color='mediumseagreen', edgecolor='black', label='P(X=x)')

# Оформлення графіка біномінального розподілу
plt.title(f'Графік біномінального розподілу\nдля n={n_binom}, p={p_binom:.2f}', fontsize=16)
plt.xlabel('Кількість успіхів (x)', fontsize=12)
plt.ylabel('Ймовірність', fontsize=12)
plt.xticks(x_binom_values)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Лінія математичного сподівання
plt.axvline(x=n_binom * p_binom, color='darkred', linestyle='--', label=f'Мат. сподівання E(X) = {n_binom * p_binom:.2f}')
plt.legend()


# --- БЛОК 4: Відображення всіх графіків ---
plt.show()

# Розрахунок параметрів розподілу Пуассона
lam = N + 10
k_max_from_task = N + 5 # k_max згідно з завданням

# Ви можете змінити k_max_plot на k_max_from_task, щоб побачити результат завдання.
k_max_plot = 35 # Рекомендоване значення для повноцінного графіка

# Виведення інформації
print("--- РОЗПОДІЛ ПУАССОНА (самописний) ---")
print(f"Номер варіанта N: {N}")
print(f"Параметр λ (лямбда): {lam}")
print(f"Математичне сподівання E(X) = λ = {lam}")
print(f"Дисперсія D(X) = λ = {lam}")
print(f"Максимальне k для графіка (за завданням): {k_max_from_task}")
print(f"Максимальне k для графіка (використано в коді): {k_max_plot}")
print("-" * 30)

# --- 3. Розрахунок ймовірностей для графіка ---
# Створюємо діапазон значень k (кількість подій від 0 до k_max_plot)
k_values = list(range(k_max_plot + 1))
y_probabilities = []

# Для кожного можливого значення k обчислюємо ймовірність
for k in k_values:
    prob = poisson_pmf(k, lam)
    y_probabilities.append(prob)

# --- 4. Побудова графіка ---
plt.figure(figsize=(14, 8))

# Стовпчикова діаграма
plt.bar(k_values, y_probabilities, color='coral', edgecolor='black', label='P(X=k)')

# Оформлення графіка
plt.title(f'Графік розподілу Пуассона\nдля λ={lam}', fontsize=16)
plt.xlabel('Кількість подій (k)', fontsize=12)
plt.ylabel('Ймовірність', fontsize=12)
plt.xticks(k_values, rotation=45) # Повернемо підписи, якщо їх багато
if len(k_values) > 30: # Показувати кожен другий підпис, якщо графік широкий
    plt.gca().set_xticks(k_values[::2])

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Лінія математичного сподівання
plt.axvline(x=lam, color='blue', linestyle='--', label=f'Мат. сподівання E(X) = {lam}')
plt.legend()

# Показати графік
plt.show()

a = -N
b = N

# Кількість можливих цілочисельних значень
num_values = b - a + 1

# Ймовірність для кожного значення
probability = 1 / num_values

# Виведення інформації
print("--- РІВНОМІРНИЙ ДИСКРЕТНИЙ РОЗПОДІЛ ---")
print(f"Номер варіанта N: {N}")
print(f"Нижня межа a: {a}")
print(f"Верхня межа b: {b}")
print(f"Кількість можливих значень: {num_values}")
print(f"Ймовірність кожного значення P(X=k): 1/{num_values} ≈ {probability:.4f}")
print("-" * 30)
# Розрахунок числових характеристик
mean = (a + b) / 2
variance = (num_values**2 - 1) / 12
print(f"Математичне сподівання E(X): {mean:.2f}")
print(f"Дисперсія D(X): {variance:.2f}")
print("-" * 30)

# --- 2. Розрахунок даних для графіка ---
# Список усіх можливих значень від a до b включно
x_values = list(range(a, b + 1))

# Список ймовірностей (однакове значення для всіх x)
y_probabilities = [probability] * num_values

# --- 3. Побудова графіка ---
plt.figure(figsize=(12, 7))

# Використовуємо стовпчикову діаграму
plt.bar(x_values, y_probabilities, color='purple', alpha=0.7, edgecolor='black', label=f'P(X=k) ≈ {probability:.3f}')

# --- 4. Оформлення графіка ---
plt.title(f'Графік рівномірного розподілу\nдля a={a}, b={b}', fontsize=16)
plt.xlabel('Значення випадкової величини (k)', fontsize=12)
plt.ylabel('Ймовірність', fontsize=12)
plt.xticks(x_values)
# Повернемо підписи, якщо їх багато
if len(x_values) > 20:
    plt.xticks(rotation=45)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Лінія математичного сподівання
plt.axvline(x=mean, color='green', linestyle='--', label=f'Мат. сподівання E(X) = {mean:.2f}')
plt.legend()

# Показати графік
plt.show()

mean = float(N)
variance = float(N / 2)
std_dev = math.sqrt(variance)

# Виведення інформації
print("--- НОРМАЛЬНИЙ РОЗПОДІЛ (самописний) ---")
print(f"Номер варіанта N: {N}")
print(f"Середнє значення a (μ): {mean}")
print(f"Дисперсія σ^2: {variance}")
print(f"Стандартне відхилення σ: {std_dev:.3f}")
print("-" * 30)

# --- 3. Генерація даних для графіка ---
# Створимо гладкий діапазон значень x за допомогою numpy.linspace
# Це потрібно для побудови плавної кривої
x_values = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 500)

# Розрахуємо значення PDF для кожного x, викликаючи нашу самописну функцію
y_pdf_values = [normal_pdf(x, mean, std_dev) for x in x_values]

# --- 4. Побудова графіка ---
plt.figure(figsize=(12, 7))

# Малюємо криву
plt.plot(x_values, y_pdf_values, color='green', lw=2, label='Функція щільності (PDF)')

# Оформлення
plt.title(f'Графік нормального розподілу \nдля a={mean}, σ²={variance}', fontsize=16)
plt.xlabel('Значення випадкової величини (x)', fontsize=12)
plt.ylabel('Щільність ймовірності', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Вертикальна лінія для середнього
plt.axvline(mean, color='red', linestyle='--', label=f'Середнє a = {mean}')

# Зафарбовуємо область ±1σ
x_fill = np.linspace(mean - std_dev, mean + std_dev, 100)
y_fill = [normal_pdf(x, mean, std_dev) for x in x_fill]
plt.fill_between(x_fill, y_fill, color='green', alpha=0.2, label='Діапазон ±1σ (~68%)')

plt.legend()
plt.show()


alpha = float(N)
x0 = float(N)

# Виведення інформації
print("--- РОЗПОДІЛ ПАРЕТО (самописний) ---")
print(f"Номер варіанта N: {N}")
print(f"Параметр форми α: {alpha}")
print(f"Параметр масштабу x₀: {x0}")
print("-" * 30)

# Розрахунок числових характеристик
mean = None
variance = None
if alpha > 1:
    mean = (alpha * x0) / (alpha - 1)
    print(f"Математичне сподівання E(X): {mean:.3f}")
if alpha > 2:
    variance = (alpha * x0**2) / ((alpha - 1)**2 * (alpha - 2))
    print(f"Дисперсія D(X): {variance:.3f}")
print("-" * 30)

# --- 3. Генерація даних для графіка ---
# Створимо діапазон значень x. Починаємо з x0 і йдемо вправо.
# Візьмемо діапазон до 40, щоб побачити, як спадає "хвіст".
x_values = np.linspace(x0, 40, 500)

# Розрахуємо значення PDF для кожного x за допомогою нашої функції
y_pdf_values = [pareto_pdf(x, x0, alpha) for x in x_values]

# --- 4. Побудова графіка ---
plt.figure(figsize=(12, 7))

# Малюємо криву
plt.plot(x_values, y_pdf_values, color='darkorange', lw=2, label='Функція щільності (PDF)')

# Оформлення
plt.title(f'Графік розподілу Парето\nдля α={alpha}, x₀={x0}', fontsize=16)
plt.xlabel('Значення випадкової величини (x)', fontsize=12)
plt.ylabel('Щільність ймовірності', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Вертикальна лінія для середнього значення (якщо воно існує)
if mean:
    plt.axvline(mean, color='blue', linestyle='--', label=f'Мат. сподівання E(X) = {mean:.2f}')

plt.legend()
plt.show()

alpha = float(N)

# Виведення інформації
print("--- РОЗПОДІЛ СТЬЮДЕНТА (самописний) ---")
print(f"Номер варіанта N: {N}")
print(f"Кількість ступенів свободи α: {alpha}")
print("-" * 30)

# Розрахунок числових характеристик
mean = 0
variance = None
if alpha > 2:
    variance = alpha / (alpha - 2)
    print(f"Математичне сподівання E(X): {mean}")
    print(f"Дисперсія D(X): {variance:.3f}")
print("-" * 30)

# --- 3. Генерація даних для графіка ---
# Створимо діапазон значень x. Для t-розподілу варто взяти трохи ширший діапазон,
# ніж для нормального, щоб побачити "важчі хвости".
x_values = np.linspace(-6, 6, 500)

# Розрахуємо значення PDF для кожного x
y_t_pdf_values = [student_t_pdf(x, alpha) for x in x_values]

# Розрахуємо PDF для стандартного нормального розподілу (μ=0, σ=1) для порівняння
y_norm_pdf_values = [normal_pdf(x, 0, 1) for x in x_values]

# --- 4. Побудова графіка ---
plt.figure(figsize=(12, 7))

# Малюємо криву t-розподілу
plt.plot(x_values, y_t_pdf_values, color='firebrick', lw=2, label=f'Розподіл Стьюдента (α={int(alpha)})')

# Малюємо криву нормального розподілу
plt.plot(x_values, y_norm_pdf_values, color='black', linestyle='--', lw=2, label='Стандартний нормальний розподіл')

# Оформлення
plt.title(f'Графік розподілу Стьюдента\nдля α={int(alpha)}', fontsize=16)
plt.xlabel('Значення випадкової величини (x)', fontsize=12)
plt.ylabel('Щільність ймовірності', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

alpha = float(N)

# Виведення інформації
print("--- РОЗПОДІЛ СТЬЮДЕНТА (самописний) ---")
print(f"Номер варіанта N: {N}")
print(f"Кількість ступенів свободи α: {alpha}")
print("-" * 30)

# Розрахунок числових характеристик
mean = 0
variance = None
if alpha > 2:
    variance = alpha / (alpha - 2)
    print(f"Математичне сподівання E(X): {mean}")
    print(f"Дисперсія D(X): {variance:.3f}")
print("-" * 30)

# --- 3. Генерація даних для графіка ---
# Створимо діапазон значень x. Для t-розподілу варто взяти трохи ширший діапазон,
# ніж для нормального, щоб побачити "важчі хвости".
x_values = np.linspace(-6, 6, 500)

# Розрахуємо значення PDF для кожного x
y_t_pdf_values = [student_t_pdf(x, alpha) for x in x_values]

# Розрахуємо PDF для стандартного нормального розподілу (μ=0, σ=1) для порівняння
y_norm_pdf_values = [normal_pdf(x, 0, 1) for x in x_values]

# --- 4. Побудова графіка ---
plt.figure(figsize=(12, 7))

# Малюємо криву t-розподілу
plt.plot(x_values, y_t_pdf_values, color='firebrick', lw=2, label=f'Розподіл Стьюдента (α={int(alpha)})')

# Малюємо криву нормального розподілу
plt.plot(x_values, y_norm_pdf_values, color='black', linestyle='--', lw=2, label='Стандартний нормальний розподіл')

# Оформлення
plt.title(f'Графік розподілу Стьюдента\nдля α={int(alpha)}', fontsize=16)
plt.xlabel('Значення випадкової величини (x)', fontsize=12)
plt.ylabel('Щільність ймовірності', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()