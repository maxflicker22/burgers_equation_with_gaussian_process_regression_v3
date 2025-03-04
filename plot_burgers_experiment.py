import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.interpolate import griddata
import matplotlib as mpl
import string  # Für a), b), c), ...

# Sichere Umwandlung mit Fehlerbehandlung
def safe_eval(x):
    try:
        if isinstance(x, str):  # Nur Strings umwandeln
            return ast.literal_eval(x)
        return x  # Falls es schon eine Liste ist, unverändert lassen
    except (SyntaxError, ValueError):  # Fehler abfangen
        return np.nan  # Oder eine leere Liste []

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert string representations of lists into actual lists
    try:

        df['relativ_error'] = df['relativ_error'].apply(safe_eval)
        df['opt_hypers'] = df['opt_hypers'].apply(safe_eval)
    except Exception as e:
        print(f"Error appeared: {e}")

    return df


def plot_relative_error(df, nu, n_train, opt_method):
    """
    Plots the relative error over time for given parameters.

    Parameters:
        csv_file (str): Path to the CSV file.
        nu (float): The value of nu to filter the data.
        n_train (int): The number of training points.
        opt_method (str): The optimization method used.
    """
    # Load the data

    # Filter data based on the given parameters
    df_filtered = df[(df['nu'] == nu) & (df['n_train'] == n_train) & (df['opt_method'] == opt_method)]

    if df_filtered.empty:
        print("No data found for the specified parameters.")
        return

    # Extract relevant columns
    length_relative_error = len(df_filtered['relativ_error'])

    time_steps = np.arange(length_relative_error) * df_filtered['dt']  # x-axis
    n_train = np.full_like(time_steps, df_filtered['n_train'])  # y-axis
    relativ_error = np.array(df_filtered['relativ_error'])  # z-axis
    jitter = df_filtered['jitter'].iloc[0]  # Assuming constant jitter


    # Plotting the error over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, relativ_error, label='Relative Error (Normalized L2 Norm)', color='b', linestyle='-', marker='x')
    plt.xlabel('Time [s]', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.title(f'Error Between Predicted and Exact Solutions Over Time\n'
              f'$\nu={nu:.6f}$, $\Delta t={df_filtered["dt"]}$, $n={n_train}$, Jitter={jitter}, Method={opt_method}', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    save_path = (f'./Figures/Burgers_Relative_Error_Over_Time_nu_{nu:.6f}_dt_{df_filtered["dt"]}_'
                 f'n_{n_train}_jitter_{jitter}_method_{opt_method}.png')
    plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()


def plot_mse_over_n(df):
    """
    Plots the mean squared error (MSE) over n_train, considering only jitter = 1e-4.
    Ensures that only one value per (opt_method, nu, n_train) is present and sorts by n_train before plotting.

    Parameters:
        df (pd.DataFrame): Dataframe containing the data.
    """
    # Filter for jitter = 1e-4
    df_filtered = df[df['jitter'] == 1e-7]

    # Keep only one unique entry per (opt_method, nu, n_train)
    df_filtered = df_filtered.drop_duplicates(subset=['opt_method', 'nu', 'n_train'])

    # Sort by n_train
    df_filtered = df_filtered.sort_values(by=['n_train'])

    unique_nu = df_filtered['nu'].unique()
    for nu in unique_nu:
        subset = df_filtered[df_filtered['nu'] == nu]
        plt.figure()
        for method in subset['opt_method'].unique():
            method_subset = subset[subset['opt_method'] == method]
            plt.plot(method_subset['n_train'], method_subset['mse'], marker='x', linestyle='--', label=method, alpha = 1)
        plt.xlabel('Training points  n', fontsize=14)
        plt.ylabel('MSE', fontsize=14)
        plt.title(r'MSE over $n$ for $\nu=\frac{0.01}{\pi}$', fontsize=14)
        print(nu)
        #plt.ylim((0, 1))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'mse_over_n_nu_{nu}.png', dpi=300)
        plt.show()




def plot_relative_error_3D_scatter(df):
    unique_nu = df['nu'].unique()

    for nu in unique_nu:
        subset = df[df['nu'] == nu]
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Farbschema für die Methoden
        methods = subset['opt_method'].unique()
        colors = sns.color_palette("husl", len(methods))  # Erzeugt eine Farbpalette mit verschiedenen Farben

        # Durchlaufe alle Methoden und plotte Punkte
        for method, color in zip(methods, colors):
            method_subset = subset[subset['opt_method'] == method]

            for _, row in method_subset.iterrows():
                if isinstance(row['relativ_error'], list) and len(row['relativ_error']) > 0:
                    length_relative_error = len(row['relativ_error'])
                    time_steps = np.arange(length_relative_error) * row['dt']  # x-axis
                    n_train = np.full_like(time_steps, row['n_train'])  # y-axis
                    relativ_error = np.array(row['relativ_error'])  # z-axis

                    # Scatter-Plot für die Methode
                    ax.scatter(time_steps, n_train, relativ_error, color=color, label=method, alpha=0.7)

        ax.set_xlabel("Time")
        ax.set_ylabel("n_train")
        ax.set_zlabel("Relative Error")
        ax.set_title(f"Relative Error 3D-Scatter-Plot for nu={nu}")

        # Legende hinzufügen (zeigt nur eine Instanz pro Methode)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Entfernt doppelte Labels
        ax.legend(by_label.values(), by_label.keys(), title="Optimization Method")

        # Speichern und anzeigen
        plt.savefig(f'relative_error_3D_scatter_nu_{nu}.png')
        plt.show()

def plot_relative_error_3D_heatmap(df):
    unique_nu = df['nu'].unique()
    fig_size = (12, 9)  # Optimale Größe für LaTeX (A4)

    # Einheitliche Reihenfolge der Methoden
    global_method_order = ["CG", "L-BFGS-B", "Newton-CG", "Powell", "none"]

    for nu in unique_nu:
        subset = df[df['nu'] == nu]
        methods = [m for m in global_method_order if m in subset['opt_method'].unique()]

        # Z-Limit setzen (Quantil nur für nu = 1)
        if nu == 1:
            all_errors = np.concatenate([
                row['relativ_error'] for _, row in subset.iterrows()
                if isinstance(row['relativ_error'], list) and len(row['relativ_error']) > 0
            ])
            global_min_z = 0
            global_max_z = np.quantile(all_errors, 0.95) if len(all_errors) > 0 else 1  # 95%-Quantil als Obergrenze
            time_limit = 0.45  # Zeitbegrenzung für nu = 1
        else:
            global_min_z = 0
            global_max_z = 1  # Für nu = 0.1 und 0.01/π bleibt das Limit 1
            time_limit = None  # Keine Begrenzung für andere Fälle

        # Erstelle eine Figur mit 2x3 Subplots (6. Feld für Colorbar)
        fig, axes = plt.subplots(2, 3, figsize=fig_size, subplot_kw={'projection': '3d'})
        fig.suptitle(rf"Relative Error for $\nu$ = {nu if nu in [0.1, 1] else '0.01/π'}", fontsize=18)

        # Farbskala wird mit dem neuen Z-Limit synchronisiert!
        norm = mpl.colors.Normalize(vmin=global_min_z, vmax=global_max_z)
        cmap = 'viridis'

        subplot_labels = list(string.ascii_lowercase)  # ['a', 'b', 'c', 'd', 'e']

        for i, (method, ax) in enumerate(zip(methods, axes.flat[:-1])):  # Letzter Subplot für Colorbar bleibt leer
            method_subset = subset[subset['opt_method'] == method]

            all_time_steps, all_n_train, all_rel_errors = [], [], []

            for _, row in method_subset.iterrows():
                if isinstance(row['relativ_error'], list) and len(row['relativ_error']) > 0:
                    length_relative_error = len(row['relativ_error'])
                    time_steps = np.arange(length_relative_error) * row['dt']  # x-axis

                    # Falls `nu = 1`, nur Werte bis t = 0.45 behalten
                    if time_limit is not None:
                        valid_indices = time_steps <= time_limit
                        time_steps = time_steps[valid_indices]
                        relativ_error = np.array(row['relativ_error'])[valid_indices]
                    else:
                        relativ_error = np.array(row['relativ_error'])  # z-axis

                    n_train = np.full_like(time_steps, row['n_train'])  # y-axis

                    all_time_steps.extend(time_steps)
                    all_n_train.extend(n_train)
                    all_rel_errors.extend(relativ_error)

            # Umwandeln in NumPy-Arrays
            all_time_steps = np.array(all_time_steps)
            all_n_train = np.array(all_n_train)
            all_rel_errors = np.array(all_rel_errors)

            if len(all_time_steps) > 0:
                # Erstelle ein Gitter für die Heatmap
                grid_x, grid_y = np.meshgrid(
                    np.linspace(all_time_steps.min(), all_time_steps.max(), 50),
                    np.linspace(all_n_train.min(), all_n_train.max(), 50)
                )
                grid_z = griddata((all_time_steps, all_n_train), all_rel_errors, (grid_x, grid_y), method='cubic')

                # 3D-Oberflächenplot (Heatmap)
                surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cmap, norm=norm, edgecolor='none', alpha=0.9)

            # Achsentitel setzen
            ax.set_xlabel(r"Time $\Delta t$ [s]", fontsize=10)
            ax.set_ylabel("Training points n", fontsize=10)
            ax.set_zlabel("Relative Error", fontsize=10)
            ax.set_title(f"({subplot_labels[i]}) Method: {method}", fontsize=14, loc='left')  # a), b), c) ...

            # Setze die neue Z-Achse
            ax.set_zlim(global_min_z, global_max_z)

        # Entferne den letzten (6.) Plot und nutze ihn für die Colorbar
        fig.delaxes(axes[1, 2])  # Entferne das rechte untere Subplot-Feld
        cbar_ax = fig.add_axes([0.78, 0.2, 0.015, 0.15])  # Noch kleiner & weiter nach unten!
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, cax=cbar_ax, label="Relative Error")

        # Speichern als hochauflösende PNG für LaTeX
        plt.savefig(f"relative_error_3D_heatmap_nu_{nu}.png", dpi=300, bbox_inches='tight')
        plt.show()

def plot_opt_hypers_3D_heatmap(df):
    unique_nu = df['nu'].unique()
    fig_size = (15, 20)

    # Einheitliche Reihenfolge der Methoden über alle Plots hinweg
    global_method_order = sorted(df['opt_method'].unique())

    # Benennung der Hyperparameter
    hyperparameter_labels = [r"$\boldsymbol{\sigma_0^2}$",
                             r"$\boldsymbol{\sigma^2}$",
                             r"$\boldsymbol{\sigma_n^2}$"]
    for nu in unique_nu:
        subset = df[df['nu'] == nu]
        methods = [m for m in global_method_order if m in subset['opt_method'].unique()]

        # Bestimme die Anzahl der Hyperparameter
        example_row = subset.iloc[0]
        num_hypers = len(example_row['opt_hypers'][0]) if isinstance(example_row['opt_hypers'], list) else 3

        # Erstelle eine Figur mit len(methods) x num_hypers Subplots
        fig, axes = plt.subplots(len(methods), num_hypers, figsize=fig_size, subplot_kw={'projection': '3d'})
        fig.suptitle(rf"Optimized Hyperparameters for $\nu$ = {nu if nu in [0.1, 1] else '0.01/π'}", fontsize=22)
        fig.subplots_adjust(right=0.85)  # Mehr Platz am rechten Rand der Figure
        if len(methods) == 1:
            axes = np.expand_dims(axes, axis=0)

        for col in range(num_hypers):
            # Wähle Transformation basierend auf Hyperparameter-Index
            if col in [0, 1, 2]:  # Hyperparameter σ₀ & σ → exp()
                transform = lambda x: np.exp(x)
            else:  # Hyperparameter σₙ bleibt normal
                transform = lambda x: x

            all_hypers = np.concatenate([
                transform(np.array(row['opt_hypers'])[:, col])
                for _, row in subset.iterrows()
                if isinstance(row['opt_hypers'], list) and len(row['opt_hypers']) > 0
            ])

            # Quantile für die Z-Skalierung berechnen
            global_min_z = np.min(all_hypers) if len(all_hypers) > 0 else 0
            quantile_90 = np.quantile(all_hypers, 0.9) if len(all_hypers) > 0 else global_min_z

            # Z-Maximum für Hyperparameter σ₀ & σ auf 90%-Quantil setzen
            if col in [0, 1]:
                global_max_z = quantile_90
            else:
                global_max_z = np.max(all_hypers) if len(all_hypers) > 0 else 1

            norm = mpl.colors.Normalize(vmin=global_min_z, vmax=global_max_z)
            cmap = 'viridis'

            for row, method in enumerate(methods):
                ax = axes[row, col]
                method_subset = subset[subset['opt_method'] == method]

                all_time_steps, all_n_train, all_hyper_values = [], [], []

                for _, row_data in method_subset.iterrows():
                    if isinstance(row_data['opt_hypers'], list) and len(row_data['opt_hypers']) > 0:
                        length_hypers = len(row_data['opt_hypers'])
                        time_steps = np.arange(length_hypers) * row_data['dt']

                        if nu == 1:
                            valid_indices = time_steps <= 0.45
                            time_steps = time_steps[valid_indices]
                            hyper_values = transform(np.array(row_data['opt_hypers'])[valid_indices, col])
                        else:
                            hyper_values = transform(np.array(row_data['opt_hypers'])[:, col])

                        n_train = np.full_like(time_steps, row_data['n_train'])

                        all_time_steps.extend(time_steps)
                        all_n_train.extend(n_train)
                        all_hyper_values.extend(hyper_values)

                all_time_steps = np.array(all_time_steps)
                all_n_train = np.array(all_n_train)
                all_hyper_values = np.array(all_hyper_values)

                if len(all_time_steps) > 0:
                    grid_x, grid_y = np.meshgrid(
                        np.linspace(all_time_steps.min(), all_time_steps.max(), 50),
                        np.linspace(all_n_train.min(), all_n_train.max(), 50)
                    )
                    grid_z = griddata((all_time_steps, all_n_train), all_hyper_values, (grid_x, grid_y), method='cubic')

                    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cmap, norm=norm, edgecolor='none', alpha=0.9)

                    # 90%-Quantil als Referenzlinie für ALLE Hyperparameter
                    #ax.contour(grid_x, grid_y, grid_z, levels=[quantile_90], colors='red', linestyles='--')

                #if row == len(methods) - 1:
                ax.set_xlabel(r"Time $\Delta t$ [s]", fontsize=10)

                #if col == 0:
                ax.set_ylabel("Training points n", fontsize=10)

                ax.set_zlabel(hyperparameter_labels[col], fontsize=12, labelpad = 10)

                ax.set_zlim(global_min_z, global_max_z)

                if col == 0:
                    ax.text2D(-0.1, 0.5, method, transform=ax.transAxes, fontsize=14, rotation=90, va='center', ha='right')

            axes[0, col].set_title(hyperparameter_labels[col], fontsize=14)

            cbar_ax = fig.add_axes([0.1 + col * 0.3, 0.05, 0.2, 0.02])
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            fig.colorbar(sm, cax=cbar_ax, orientation='horizontal').set_label(hyperparameter_labels[col], fontsize=14)

        plt.savefig(f"opt_hypers_3D_heatmap_nu_{nu}.png", dpi=300, bbox_inches='tight')
        plt.show()


def plot_mse_vs_dt(df, selected_n, selected_opt_method, selected_nu):
    subset = df[(df['n_train'] == selected_n) & (df['opt_method'] == selected_opt_method) & (df['nu'] == selected_nu)]
    plt.figure()
    plt.plot(subset['dt'], subset['mse'], marker='o', linestyle='-',
             label=f'n_train={selected_n}, opt_method={selected_opt_method}, nu={selected_nu}')
    plt.xlabel('dt')
    plt.ylabel('MSE')
    plt.title(f'MSE vs dt for selected configuration')
    plt.legend()
    plt.grid()
    plt.savefig(f'mse_vs_dt_n_{selected_n}_opt_{selected_opt_method}_nu_{selected_nu}.png')
    plt.show()


if __name__ == "__main__":
    file_path = "./Figures/Burgers_Experiment_Results.csv"
    df = load_data(file_path)

    #plot_mse_over_n(df)
    #plot_relative_error_3D_scatter(df)
    #plot_relative_error_3D_heatmap(df)
    plot_opt_hypers_3D_heatmap(df)
    #plot_relative_error(df, 1, 57, "CG")

    # Example: User-defined configuration for MSE vs dt
    selected_n = 27  # Change as needed
    selected_opt_method = 'Powell'  # Change as needed
    selected_nu = 0.003183  # Change as needed
    #plot_mse_vs_dt(df, selected_n, selected_opt_method, selected_nu)
