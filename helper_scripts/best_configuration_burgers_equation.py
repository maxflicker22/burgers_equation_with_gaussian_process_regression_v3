import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate  # Für schöne Tabellenformatierung


# Lade die CSV-Datei
file_path = "../Figures/Burgers_Experiment_Results.csv"  # Ersetze mit deinem Dateipfad
df = pd.read_csv(file_path)

# Entferne mögliche Leerzeichen in den Spaltennamen
df.columns = df.columns.str.strip()


# Sichere Umwandlung von `relativ_error` in Listen
def safe_eval(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except (SyntaxError, ValueError):
        return []  # Falls Umwandlung fehlschlägt, leere Liste zurückgeben


# Wandle `relativ_error` um
df['relativ_error'] = df['relativ_error'].apply(safe_eval)

# Finde die 3 einzigartigen nu-Werte
unique_nu_values = sorted(df['nu'].unique())  # Sortiere nach Größe
print(f"Gefundene nu-Werte: {unique_nu_values}")

# Initialisiere leere Liste für die Top 10 Ergebnisse pro nu
best_configs = []
top_5_configs = []  # Liste für die besten 5 Konfigurationen pro nu

# Durchlaufe alle `nu`-Werte und finde die 10 besten Lösungen (geringstes mse)
for nu_value in unique_nu_values:
    best_nu_df = df[df['nu'] == nu_value].nsmallest(10, 'mse')  # Wähle die 10 kleinsten mse
    best_configs.append(best_nu_df)

    # Speichere die 5 besten nur mit relevanten Spalten
    best_5 = best_nu_df[['jitter', 'dt', 'n_train', 'opt_method', 'mse']].head(10)
    best_5.insert(0, "nu", nu_value)  # Füge nu-Wert hinzu
    top_5_configs.append(best_5)

# Kombiniere alle Ergebnisse
final_df = pd.concat(best_configs)

# Zeige die besten Konfigurationen in der Konsole an
print("\n✅ Die besten 30 Konfigurationen nach mse (10 pro nu):")
print(final_df.head(10))  # Zeigt die ersten 10 Zeilen in der Konsole

# Speichere die Ergebnisse in eine neue CSV
final_df.to_csv("beste_konfigurationen.csv", index=False)
print("\n✅ Die besten 30 Konfigurationen wurden gespeichert in 'beste_konfigurationen.csv'\n")

# === 🖥️ SCHÖNE PRINT-AUSGABE DER TOP 5 PRO NU ===

# Erstelle eine formatierte Tabelle
table_data = pd.concat(top_5_configs)
print("📊 **Top 5 Konfigurationen pro nu-Wert (geringstes mse zuerst)**")
print(tabulate(table_data, headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt=".6f"))

# === 🔥 PLOTTING DER RELATIVEN FEHLER ===

plt.figure(figsize=(12, 6))

# Farben für unterschiedliche nu-Werte
colors = sns.color_palette("husl", len(unique_nu_values))

# Durchlaufe die Top 30 und plotte relativ_error
for i, (index, row) in enumerate(final_df.iterrows()):
    if len(row['relativ_error']) > 0:
        time_steps = np.arange(len(row['relativ_error'])) * row['dt']  # Zeitpunkte berechnen
        nu_value = row['nu']
        color = colors[list(unique_nu_values).index(nu_value)]  # Gleiche Farbe für gleichen nu-Wert

        plt.plot(time_steps, row['relativ_error'], label=f"nu={nu_value}, n_train={row['n_train']}, method={row['opt_method']}", alpha=0.7,
                 color=color)

plt.xlabel("Time [s]")
plt.ylabel("Relative Error")
plt.title("Comparison of the Best 30 Solutions by Relative Error")
plt.legend(loc='upper right', fontsize=8, ncol=2)
plt.grid(True)
plt.show()
