# import numpy as np

# def calculate_transitive_closure(matrix):
#     """
#     Обчислює транзитивне замикання для нечіткого відношення
#     за допомогою max-min композиції (аналог алгоритму Флойда-Воршелла).
#     """
#     # Отримуємо розмірність матриці (кількість агентів)
#     n = matrix.shape[0]
    
#     # Створюємо копію початкової матриці, яку будемо оновлювати
#     T = matrix.copy()
    
#     # Головний цикл алгоритму
#     # k - індекс проміжного агента (посередника)
#     for k in range(n):
#         # i - індекс агента-відправника
#         for i in range(n):
#             # j - індекс агента-одержувача
#             for j in range(n):
#                 # Знаходимо надійність шляху через посередника k
#                 indirect_path_strength = np.minimum(T[i, k], T[k, j])
                
#                 # Обираємо сильніший шлях: поточний або новий опосередкований
#                 T[i, j] = np.maximum(T[i, j], indirect_path_strength)
                
#     return T

# def print_matrix(title, matrix, headers):
#     """Допоміжна функція для красивого виводу матриці."""
#     print(f"--- {title} ---")
    
#     # Вивід заголовків стовпців
#     header_str = "{: >12}".format("") + " ".join(["{: >12}".format(h) for h in headers])
#     print(header_str)
#     print("-" * len(header_str))

#     # Вивід рядків матриці
#     for i, row in enumerate(matrix):
#         row_str = "{: >12}".format(headers[i])
#         for val in row:
#             row_str += " {: >12.2f}".format(val)
#         print(row_str)
#     print("\n")


# # --- Вхідні дані з нашого сценарію ---

# # 1. Список агентів (універсум X)
# agents = ["Спектр", "Фантом", "Кассандра", "Голіаф", "Тінь"]

# # 2. Початкова матриця відношення Q (прямі зв'язки)
# Q = np.array([
#     [1.0, 0.6, 0.2, 0.1, 0.9],  # Спектр
#     [0.6, 1.0, 0.3, 0.5, 0.4],  # Фантом
#     [0.2, 0.3, 1.0, 0.8, 0.2],  # Кассандра
#     [0.1, 0.5, 0.8, 1.0, 0.1],  # Голіаф
#     [0.9, 0.4, 0.2, 0.1, 1.0]   # Тінь
# ])


# # --- Виконання та вивід результатів ---

# # Виводимо початкову матрицю
# print_matrix("Початкова матриця Q (Прямі зв'язки)", Q, agents)

# # Обчислюємо транзитивне замикання
# T_Q = calculate_transitive_closure(Q)

# # Виводимо результуючу матрицю
# print_matrix("Транзитивне замикання T(Q) (Ефективність комунікації)", T_Q, agents)


import numpy as np

def calculate_transitive_closure(matrix):
    """
    Обчислює транзитивне замикання для нечіткого відношення
    за допомогою max-min композиції (аналог алгоритму Флойда-Воршелла).
    """
    n = matrix.shape[0]
    T = matrix.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                indirect_path_strength = np.minimum(T[i, k], T[k, j])
                T[i, j] = np.maximum(T[i, j], indirect_path_strength)
    return T

def print_matrix(title, matrix, headers):
    """Допоміжна функція для красивого виводу матриці."""
    print(f"--- {title} ---")
    header_str = "{: >16}".format("") + " ".join(["{: >16}".format(h) for h in headers])
    print(header_str)
    print("-" * len(header_str))
    for i, row in enumerate(matrix):
        row_str = "{: >16}".format(headers[i])
        for val in row:
            row_str += " {: >16.2f}".format(val)
        print(row_str)
    print("\n")

# --- Вхідні дані для Сценарію 2: IT-Команда ---

# 1. Список членів команди
team_members = ["Анна (Лідер)", "Богдан (Backend)", "Вікторія (Front)", "Григорій (QA)", "Дмитро (DevOps)"]

# 2. Початкова матриця відношення Q₂
Q2 = np.array([
    [1.0, 0.9, 0.8, 0.7, 0.8],  # Анна
    [0.9, 1.0, 0.8, 0.6, 0.5],  # Богдан
    [0.8, 0.8, 1.0, 0.7, 0.2],  # Вікторія
    [0.7, 0.6, 0.7, 1.0, 0.1],  # Григорій
    [0.8, 0.5, 0.2, 0.1, 1.0]   # Дмитро
])

# --- Виконання та вивід результатів ---

print_matrix("Початкова матриця Q₂ (Прямі зв'язки в IT-команді)", Q2, team_members)

T_Q2 = calculate_transitive_closure(Q2)

print_matrix("Транзитивне замикання T(Q₂) (Ефективність комунікації в IT-команді)", T_Q2, team_members)