import numpy as np

def print_matrix(title, matrix):
    print(f"--- {title} ---")
    print(matrix.astype(int))
    print("\n")

def calculate_transitive_closure(p_matrix):
    """
    Обчислює транзитивне замикання P̂ за формулою:
    P̂ = P ∪ P² ∪ ... ∪ Pⁿ
    """
    n = p_matrix.shape[0]
    p_hat = p_matrix.copy()
    p_power = p_matrix.copy()

    # Ми послідовно обчислюємо P², P³, P⁴... і додаємо їх до результату
    for _ in range(n - 1):
        p_power = (p_power @ p_matrix).astype(bool)
        
        # p_hat | p_power -> це логічне "АБО" (об'єднання ∪)
        p_hat = p_hat | p_power
        
    return p_hat

if __name__ == "__main__":
    # a. Вхідні дані: початкова матриця відношення P

    P = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=bool)

    print_matrix("Початкова матриця відношення P", P)

    # --- Пункт 4.а: Знайти транзитивне замикання P̂ ---
    P_hat = calculate_transitive_closure(P)
    print_matrix("a. Транзитивне замикання P̂ = P ∪ P² ∪ P³ ∪ P⁴", P_hat)

    # --- Пункт 4.b: Знайти відношення досяжності P̃ ---
    # E - одинична матриця (identity matrix)
    E = np.identity(P.shape[0], dtype=bool)
    # P̃ = E ∪ P̂  (об'єднання з одиничною матрицею)
    P_tilde = E | P_hat
    print_matrix("b. Відношення досяжності P̃ = E ∪ P̂", P_tilde)

    # --- Пункт 4.c: Знайти відношення взаємної досяжності P̿ ---
    # P̃⁻¹ - це транспонована матриця P̃
    P_tilde_inverse = P_tilde.T
    # P̿ = P̃ ∩ P̃⁻¹ (логічне "І")
    P_bar = P_tilde & P_tilde_inverse
    print_matrix("c. Відношення взаємної досяжності P̿ = P̃ ∩ P̃⁻¹", P_bar)