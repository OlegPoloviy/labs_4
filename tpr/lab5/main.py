import random
import statistics
import math
import numpy as np # Використовуємо numpy для зручного обчислення середнього геометричного
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. Вхідні дані ---
N = 9
data_size = 3000
min_value = 1
max_value = 5 * N

# --- 2. Генерація множини даних ---
try:
    # Важливо: дані мають бути додатними для середнього геометричного та гармонічного
    data = [random.randint(min_value, max_value) for _ in range(data_size)]
    print(f"Успішно згенеровано {data_size} випадкових чисел в діапазоні [{min_value}, {max_value}].")
except Exception as e:
    print(f"Виникла помилка при генерації даних: {e}")
    data = []

if data:
    # --- 3. Ранжування вибірки ---
    ranked_data = sorted(data)
    print("\n--- Проранжована вибірка (перші 100 значень) ---")
    print(ranked_data[:100])

    # --- 4. Обчислення середніх величин ---
    print("\n--- Розрахунок середніх величин ---")
    
    # 4.1. Середнє арифметичне
    try:
        mean_value = statistics.mean(data)
        print(f"Середнє арифметичне: {mean_value:.4f}")
    except Exception as e:
        print(f"Помилка при розрахунку середнього арифметичного: {e}")

    # 4.2. Медіана
    try:
        median_value = statistics.median(data)
        print(f"Медіана: {median_value}")
    except Exception as e:
        print(f"Помилка при розрахунку медіани: {e}")

    # 4.3. Мода
    try:
        mode_value = statistics.mode(data)
        print(f"Мода: {mode_value}")
    except statistics.StatisticsError:
        print("Мода: Неможливо визначити єдину моду (кілька значень зустрічаються однакову кількість разів).")
    except Exception as e:
        print(f"Помилка при розрахунку моди: {e}")
        
    # 4.4. Середнє квадратичне
    # Формула: корінь квадратний із суми квадратів елементів, поділеної на їх кількість.
    try:
        rms_value = math.sqrt(sum(x**2 for x in data) / len(data))
        print(f"Середнє квадратичне: {rms_value:.4f}")
    except Exception as e:
        print(f"Помилка при розрахунку середнього квадратичного: {e}")
        
    # 4.5. Середнє геометричне
    # Формула: корінь n-го степеня з добутку всіх елементів.
    # Використовуємо numpy для уникнення переповнення при великому добутку.
    try:
        # Альтернативний розрахунок через логарифми, щоб уникнути величезних чисел
        log_sum = sum(math.log(x) for x in data)
        geometric_mean_value = math.exp(log_sum / len(data))
        print(f"Середнє геометричне: {geometric_mean_value:.4f}")
    except Exception as e:
        print(f"Помилка при розрахунку середнього геометричного: {e}")
        
    # 4.6. Середнє гармонічне
    # Формула: кількість елементів, поділена на суму обернених значень (1/x).
    try:
        harmonic_mean_value = statistics.harmonic_mean(data)
        print(f"Середнє гармонічне: {harmonic_mean_value:.4f}")
    except Exception as e:
        print(f"Помилка при розрахунку середнього гармонічного: {e}")


    # --- 5. Графічне відображення ---
    try:
        frequency = Counter(ranked_data)
        numbers = list(frequency.keys())
        counts = list(frequency.values())

        plt.figure(figsize=(14, 7))
        plt.bar(numbers, counts, color='teal', edgecolor='black')
        
        plt.xlabel("Значення числа")
        plt.ylabel("Частота появи")
        plt.title("Гістограма частот згенерованих випадкових чисел")
        plt.xticks(range(min_value, max_value + 1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        print("\nГрафік частот буде відображено в окремому вікні.")
        plt.show()
    except Exception as e:
        print(f"Не вдалося побудувати графік: {e}")