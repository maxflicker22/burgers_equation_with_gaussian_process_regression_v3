import pandas as pd
import ast


# Lade die CSV-Datei
file_path = "./beste_konfigurationen.csv"  # Ersetze mit deinem Dateipfad
df = pd.read_csv(file_path)

# Entferne mögliche Leerzeichen in den Spaltennamen
df.columns = df.columns.str.strip()

# Identifiziere Spalten, die Listen enthalten (z. B. `relativ_error`, `analytic_solution`, `predicted_solution`)
list_columns = []
for col in df.columns:
    sample_value = df[col].dropna().iloc[0]  # Beispielwert aus der Spalte
    if isinstance(sample_value, str):
        try:
            eval_value = ast.literal_eval(sample_value)
            if isinstance(eval_value, (list, dict)):
                list_columns.append(col)  # Speichere Spalten, die Listen enthalten
        except (ValueError, SyntaxError):
            continue

# Erstelle eine neue Tabelle ohne die Spalten mit Listen
df_filtered = df.drop(columns=list_columns)

# Sortiere nach mse aufsteigend (kleinster Wert zuerst)
df_sorted = df_filtered.sort_values(by="mse", ascending=True)

print(df_sorted.head(10))  # Zeigt die ersten 10 Zeilen in der Konsole

# Speichere die sortierte Tabelle als CSV
csv_output = "sortierte_konfigurationsdaten.csv"
df_sorted.to_csv(csv_output, index=False)
print(f"✅ Die sortierte Tabelle wurde gespeichert als '{csv_output}'")

# === 🔥 LaTeX Export ===

# Definiere das LaTeX-Format
latex_table = df_sorted.to_latex(index=False, float_format="%.5f", column_format="|l" * len(df_sorted.columns) + "|")

# Speichere die LaTeX-Tabelle in eine Datei
latex_output = "sortierte_konfigurationsdaten.tex"
with open(latex_output, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(f"✅ Die LaTeX-Tabelle wurde gespeichert als '{latex_output}'")
