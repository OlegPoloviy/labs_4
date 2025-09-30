import numpy as np

# Множина
M = ["З", "Л", "В", "О"]

# Відношення P ("не холодніше")
P = np.array([
    [1, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1]
])

# Відношення Q ("так само")
Q = np.eye(4, dtype=int)  # одинична матриця

def print_matrix(name, mat):
    print(f"{name}:")
    for row in mat:
        print(" ".join(str(x) for x in row))
    print()

# 1. Перетин
intersect = np.logical_and(P, Q).astype(int)
print_matrix("P ∩ Q", intersect)

# 2. Об'єднання
union = np.logical_or(P, Q).astype(int)
print_matrix("P ∪ Q", union)

# 3. Різниця P\Q
diff = np.logical_and(P, np.logical_not(Q)).astype(int)
print_matrix("P \\ Q", diff)

# 4. Симетрична різниця
sym_diff = np.logical_xor(P, Q).astype(int)
print_matrix("P Δ Q", sym_diff)

# 5. Доповнення (відносно MxM)
comp = np.logical_not(P).astype(int)
print_matrix("¬P", comp)

# 6. Композиція (P∘Q = P, Q∘P = P)
comp_pq = (Q @ P > 0).astype(int)
print_matrix("P ∘ Q", comp_pq)

# 7. Обернене відношення
P_inv = P.T
print_matrix("P^(-1)", P_inv)
