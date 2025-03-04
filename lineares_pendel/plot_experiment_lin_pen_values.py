import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Lade die Daten aus den CSVs
# ==============================
df_results = pd.read_csv("experiment_results.csv")
df_hyperparams = pd.read_csv("hyperparameter_history.csv")

# Stelle sicher, dass `opt_method` eine Kategorie ist, um saubere Plots zu erstellen
df_results["opt_method"] = df_results["opt_method"].astype("category")
df_hyperparams["opt_method"] = df_hyperparams["opt_method"].astype("category")

# ==============================
# 2. Erstelle die erste Figure mit 3 Plots
# ==============================
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()
# Setze das Design für die Plots
sns.set_style("whitegrid")

# Plot 1: MSE gegen n_train
sns.lineplot(data=df_results, x="n_train", y="mse_pigp", hue="opt_method", marker="X", ax=axes[0], alpha=0.6, linestyle='--')
#axes[0].set_title("Mean Squared Error (MSE) vs. n_train")
axes[0].set_ylabel("Mean Squared Error", fontsize=14)
axes[0].set_xlabel("Training pints n", fontsize=14)
axes[0].legend(title="Optimization method", fontsize=14)
axes[0].grid(True)
axes[0].set_xlim(0,30)


# Plot 2: Standardabweichung vs. n_train
sns.lineplot(data=df_results, x="n_train", y="std_pigp", hue="opt_method", marker="X", ax=axes[1], alpha=0.8, linestyle='--')
#axes[1].set_title("Standardabweichung vs. n_train")
axes[1].set_ylabel("Standard deviation", fontsize=14)
axes[1].set_xlabel("Training points n", fontsize=14)
axes[1].legend(title="Optimization method", fontsize=14)
axes[1].grid(True)
axes[1].set_xlim(0,30)

#  Plot 3: Optimierungszeit vs. n_train
sns.lineplot(data=df_results, x="n_train", y="opt_time_sec", hue="opt_method", marker="X", ax=axes[2])
#axes[2].set_title("Optimierungszeit vs. n_train")
axes[2].set_ylabel("Optimization time [s]", fontsize=14)
axes[2].set_xlabel("Training points n", fontsize=14)
axes[2].legend(title="Optimization method", fontsize=14)
axes[2].grid(True)
axes[2].set_xlim(0,200)

# Styling & Anzeigen
#plt.tight_layout()
#plt.show()

# ==============================
# 3. Erstelle die zweite Figure für die Hyperparameter-Optimierungspfade
# ==============================
#fig, ax = plt.subplots(figsize=(12, 8))  # Explizite Achse `ax` definieren

# Wähle einen bestimmten `n_train`-Wert für die Visualisierung
selected_n_train = 20 # Kann nach Wunsch geändert werden

# Filtere die Hyperparameter-Daten für diesen `n_train`-Wert
df_hyperparams_subset = df_hyperparams[df_hyperparams["n_train"] == selected_n_train]

# Erstelle eine eigene Farbliste für jede Methode
color_palette = sns.color_palette("tab10", n_colors=len(df_hyperparams_subset["opt_method"].unique()))
method_colors = {method: color_palette[i] for i, method in enumerate(df_hyperparams_subset["opt_method"].unique())}

# Variable für die Farbskala (`dgl_parameter`)
all_dgl_params = []

# Erstelle einen Plot für jede Methode
for method in df_hyperparams_subset["opt_method"].unique():
    df_subset = df_hyperparams_subset[df_hyperparams_subset["opt_method"] == method]

    # Sortiere nach Reihenfolge der Iteration
    df_subset = df_subset.sort_index().reset_index(drop=True)

    # Speichere die `dgl_parameter`-Werte für die Farblegende
    all_dgl_params.extend(df_subset["dgl_parameter"].values)

    # Farbskala basierend auf `dgl_parameter`
    norm = plt.Normalize(df_subset["dgl_parameter"].min(), df_subset["dgl_parameter"].max())
    cmap = plt.cm.viridis

    # Wähle die Farbe für die Methode (Start- & Endpunkt)
    method_color = method_colors[method]

    # Zeichne die Linien mit `dgl_parameter`-abhängiger Farbe
    for i in range(len(df_subset) - 1):
        axes[3].plot(
            [df_subset["signal_noise"].iloc[i], df_subset["signal_noise"].iloc[i + 1]],
            [df_subset["length_scale"].iloc[i], df_subset["length_scale"].iloc[i + 1]],
            color=method_color, linewidth=1, alpha=0.6
            #color=cmap(norm(df_subset["dgl_parameter"].iloc[i])), linewidth=2, alpha=0.8
        )

    # Scatter-Punkte mit Farbverlauf (`dgl_parameter`)
    sc = axes[3].scatter(df_subset["signal_noise"], df_subset["length_scale"],
                    c=df_subset["dgl_parameter"], cmap="viridis", edgecolors="k", s=50)

    # Markiere den Startpunkt in der Farbe der Methode
    axes[3].scatter(df_subset["signal_noise"].iloc[0], df_subset["length_scale"].iloc[0],
               color=method_color, s=100, edgecolors="black", marker="o", label=f"Start {method}")

    # Markiere das gefundene Optimum in der Farbe der Methode
    axes[3].scatter(df_subset["signal_noise"].iloc[-1], df_subset["length_scale"].iloc[-1],
               color=cmap(norm(df_subset["dgl_parameter"].iloc[-1])), s=100, edgecolors="black", marker="X", label=f"Optimal {method}")

# Styling für den Optimierungspfad-Plot
axes[3].set_title(f"Hyperparameter-Opimization paths for n = {selected_n_train} Training points", fontsize=14)
axes[3].set_xlabel(r'Signal Noise $\sigma_s$', fontsize=14)
axes[3].set_ylabel(r'Length Scale $\ell$', fontsize=14)
axes[3].legend(title="Optimization method", fontsize=14)
axes[3].grid(True)

# Füge eine Farblegende für den `dgl_parameter` hinzu
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(all_dgl_params), max(all_dgl_params)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes[3])  # 🎯 Hier explizit `ax=ax` setzen!
cbar.set_label(r"ODE-Parameter $\phi$", fontsize=14)

#  Beschriftungen unter den Subplots mittig setzen
labels = ["(a)", "(b)", "(c)", "(d)"]
for i, ax in enumerate(axes.flatten()):
    ax.text(0.5, -0.2, labels[i], transform=ax.transAxes, fontsize=14, fontweight="bold", ha="center")

#  Anzeigen des Plots
plt.tight_layout()
plt.savefig("experiment_lineares_pendel.png")
plt.show()