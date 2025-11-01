# Creating example fuzzy relations S (X x Y) and Q (Y x Z) and computing their max-min composition R = S ∘ Q.
import pandas as pd
import numpy as np

# Sets
X = ["Менеджер", "Програміст", "Водій", "Секретар"]
Y = ["гнучкість мислення", "уміння швидко приймати рішення", "концентрація уваги",
     "зорова пам'ять", "витривалість", "швидкість реакції рухів", "відповідальність"]
Z = ["Андрієнко", "Василенко", "Іваненко", "Дмитренко", "Петренко", "Романенко"]

# Fuzzy relation S: X x Y (membership degrees 0..1)
S = np.array([
    [0.8, 0.9, 0.6, 0.5, 0.7, 0.5, 0.9],  # Менеджер
    [0.7, 0.6, 0.9, 0.8, 0.6, 0.4, 0.8],  # Програміст
    [0.4, 0.7, 0.7, 0.9, 0.8, 0.9, 0.8],  # Водій
    [0.5, 0.6, 0.7, 0.8, 0.6, 0.6, 0.7],  # Секретар
])

# Fuzzy relation Q: Y x Z (how well each candidate has each characteristic)
Q = np.array([
    # A   V   I   D   P   R  (columns correspond to Z)
    [0.6,0.7,0.5,0.4,0.6,0.5],  # гнучкість мислення
    [0.7,0.6,0.8,0.5,0.4,0.6],  # уміння швидко приймати рішення
    [0.8,0.7,0.6,0.6,0.9,0.5],  # концентрація уваги
    [0.6,0.6,0.7,0.5,0.6,0.5],  # зорова пам'ять
    [0.7,0.5,0.6,0.6,0.7,0.6],  # витривалість
    [0.5,0.6,0.7,0.8,0.6,0.7],  # швидкість реакції рухів
    [0.9,0.8,0.7,0.6,0.8,0.7],  # відповідальність
])

# Compute max-min composition R (X x Z)
R = np.zeros((S.shape[0], Q.shape[1]))
for i in range(S.shape[0]):  # for each profession x
    for k in range(Q.shape[1]):  # for each candidate z
        mins = np.minimum(S[i, :], Q[:, k])  # min over y for given x and z
        R[i, k] = np.max(mins)  # max over y

# Create DataFrames for nicer display
df_S = pd.DataFrame(S, index=X, columns=Y)
df_Q = pd.DataFrame(Q, index=Y, columns=Z)
df_R = pd.DataFrame(R, index=X, columns=Z)

print("Нечітке відношення S (спеціальності → характеристики):")
print(df_S, "\n")

print("Нечітке відношення Q (характеристики → претенденти):")
print(df_Q, "\n")

print("Композиція R = S ∘ Q (спеціальності → претенденти):")
print(df_R.round(3), "\n")


# Also produce ranking: for each candidate, professions sorted by degree
rankings = {}
for j, candidate in enumerate(Z):
    col = df_R[candidate]
    sorted_prof = col.sort_values(ascending=False)
    rankings[candidate] = sorted_prof

# Display textual ranking in a small DataFrame: best profession and degree, then second best
summary = []
for candidate in Z:
    sorted_prof = rankings[candidate]
    best = sorted_prof.index[0], sorted_prof.iloc[0]
    second = sorted_prof.index[1], sorted_prof.iloc[1]
    summary.append({
        "Претендент": candidate,
        "1-й вибір (професія)": best[0],
        "Ступінь належності (μ)": round(float(best[1]),3),
        "2-й вибір (професія)": second[0],
        "Ступінь належності (μ)": round(float(second[1]),3)
    })

df_summary = pd.DataFrame(summary).set_index("Претендент")
cjt.display_dataframe_to_user("Резюме рекомендацій (топ-2 для кожного претендента)", df_summary)

# Print R to output as well
df_R.round(3)
