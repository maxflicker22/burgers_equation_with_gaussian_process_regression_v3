import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from jax import grad

# Sicherstellen, dass JAX float64 verwendet
# ohne funktioniert scipy.optimize nicht mit jax
jax.config.update("jax_enable_x64", True)


# Hilfsfunktion
def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Störfunktion
def source_function(x):
    #return x**2
    return jnp.zeros_like(x)

# Evaluiere Mean Squared Error
def evaluate_mean_squared_error(analytic_solution, predicted_func):
    mse = jnp.mean((analytic_solution - predicted_func) ** 2)
    return mse


# Berechne Gesamt Standardabweichung aus Kovarianz Matrix
def calculate_total_standarddeviation(cov_matrix):
    return jnp.sqrt(np.trace(np.abs(cov_matrix)))


# Lösungsfunktion des linearen Pendels
def lineares_pendel_dgl_trial_function(x):
    phi_max = 1
    g = 1
    l = 1
    homogen = phi_max * jnp.sin(jnp.sqrt(g / l) * x)
    #inhomogen = homogen + 1 / (4 * np.pi ** 2 - (g / l)) * np.sin(2 * np.pi * x)
    return homogen


# RBF-Kernel Funktion
def rbf_kernel(x, x_prime, signal_noise, length_scale):
    """ Berechnet den RBF-Kernel zwischen zwei Vektoren x und x_prime"""
    return signal_noise ** 2 * jnp.exp(-jnp.abs(x - x_prime) ** 2 / (2 * length_scale ** 2))


# Kff Kernel Funktion → Returns Matrix
def kff_function(x, x_prime, signal_noise, length_scale):
    # Erzeuge ein Gitter aus x, x_prime
    x, x_prime = jnp.meshgrid(x, x_prime, indexing='ij')
    return rbf_kernel(x, x_prime, signal_noise, length_scale)



# Kgf Kernel Funktion → Returns Matrix
def kgf_function(x, x_prime, signal_noise, length_scale, dgl_parameter):
    # Erzeuge ein Gitter aus x, x_prime
    x, x_prime = jnp.meshgrid(x, x_prime, indexing='ij')

    # Kff Funktion berechnen
    kff = rbf_kernel(x, x_prime, signal_noise, length_scale)

    #Vorfaktoren der zweiten Ableitung
    kgf = (x ** 2 - 2 * x_prime * x + x_prime ** 2 - length_scale ** 2) / (length_scale ** 4)

    return (kgf + dgl_parameter) * kff


# Kfg Kernel Funktion → Returns Matrix
def kfg_function(x, x_prime, signal_noise, length_scale, dgl_parameter):
    return kgf_function(x_prime, x, signal_noise, length_scale, dgl_parameter)


# Kgg Kernel Funktion → Returns Matrix
def kgg_function(x, x_prime, signal_noise, length_scale, dgl_parameter):
    # verschiedene Kernel funktionen definieren
    kff = kff_function(x, x_prime, signal_noise, length_scale)
    kfg = kfg_function(x, x_prime, signal_noise, length_scale, dgl_parameter)
    kgf = kgf_function(x, x_prime, signal_noise, length_scale, dgl_parameter)

    # Erzeuge ein Gitter aus x, x_prime
    x, x_prime = jnp.meshgrid(x, x_prime, indexing='ij')

    #Vorfaktor der 2 zweifachen Ableitungen
    kgg = (
              (x_prime ** 4 - 4 * x * x_prime ** 3 + (6 * x ** 2 - 6 * length_scale ** 2) * x_prime ** 2
               + (12 * length_scale ** 2 * x - 4 * x ** 3) * x_prime
               + x ** 4 - 6 * length_scale ** 2 * x ** 2 + 3 * length_scale ** 4)
          ) / (length_scale ** 8)

    # Zusammenfügen der einzelnen Terme ---> dgl_parameter*kff muss abgezogen werden sonst 2 Mal vorhanden
    return kgg * kff + dgl_parameter * (kfg + kgf - dgl_parameter * kff)


# Funktion für Block Matrix von beobachteten Punkten → Ky
def covariance_matrix_training_points(xf, xg, signal_noise, length_scale, sigma_f, sigma_g, dgl_parameter):
    kff = kff_function(xf, xf, signal_noise, length_scale) + sigma_f ** 2 * jnp.identity(len(xf))
    kfg = kfg_function(xf, xg, signal_noise, length_scale, dgl_parameter)
    kgf = kfg.T
    kgg = kgg_function(xg, xg, signal_noise, length_scale, dgl_parameter) + sigma_g ** 2 * jnp.identity(len(xg))

    Ky = jnp.block([
        [kff, kfg],
        [kgf, kgg]
    ])


    #print("Ist die Matrix Ky positiv definit?", is_positive_definite(Ky))

    return Ky


# Funktion Berechne negative log marginal likelihood mittels cholesky Decomposition
def posterio_mean_variance_lml(y, covariance_matrix, k_star_star, k_star_):
    # Für Numerische Stabilität füge einen Jitter hinzug
    jitter = np.mean(np.diag(covariance_matrix)) * 10e-5
    covariance_matrix = covariance_matrix + np.eye(len(y)) * jitter


    # Cholskey Decomposition für pos. semidefinit Matrix K = L @ L.T
    L = np.zeros_like(covariance_matrix)
    try:
        L = np.linalg.cholesky(covariance_matrix)
    except np.linalg.LinAlgError:
        print("Error: matrix not pos semi definit")


    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    # Berechne posterior Mean
    post_mean = k_star_ @ alpha

    v = np.linalg.solve(L, k_star_.T)

    # Berechne posterior Varianz
    post_variance = k_star_star - v.T @ v

    # Berechne Log Marginal Likelihood
    log_marg_likelihood = -1/2 * y.T @ alpha - np.trace(L) - len(y)/2 * np.log(2*np.pi)

    return post_mean, post_variance, log_marg_likelihood


# Berechne Negative log marginal likelihood zum Optimieren der Hyperparameter
def nlml_for_optimization(hyp, args):

    # Unpacking hyperparameters
    signal_noise, length_scale, dgl_parameter = hyp  # Assuming hyp has exactly 3 elements

    # Unpacking arguments
    xf, xg, sigma_f, sigma_g, y = args  # Assuming args has exactly 4 elements

    covariance_matrix = covariance_matrix_training_points(xf, xg, signal_noise, length_scale, sigma_f, sigma_g, dgl_parameter)

    # Für Numerische Stabilität füge einen Jitter hinzug
    jitter = jnp.mean(jnp.diag(covariance_matrix)) * 10e-5

    covariance_matrix += jnp.eye(len(xf) + len(xg)) * jitter

    # Überprüfen ob matrix instabil ist, mittels kkonditions zahl
    #cond_number = jnp.linalg.cond(covariance_matrix)
    #print("Konditionszahl der Matrix:", cond_number) # cond Number größer 10^6 → sehr schlecht

    # Cholskey Decomposition für pos. semidefinit Matrix K = L @ L.T
    L = jnp.zeros_like(covariance_matrix)
    try:
        L = jnp.linalg.cholesky(covariance_matrix)
    except jnp.linalg.LinAlgError:
        print("Error: matrix not pos semi definit")

    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))

    # Berechne Negative Log Marginal Likelihood
    nlml = 1 / 2 * y.T @ alpha + jnp.trace(L) + len(y) / 2 * jnp.log(2 * np.pi)

    #print("NLML: ", nlml)
    return nlml


# JAX-Gradient für Optimierung
grad_nlml = jax.grad(nlml_for_optimization, argnums=0)


def run_experiment(n_trainingspoints, opt_method):
    # Hilfsfunktion
    def callback(xk):
        """Speichert die aktuellen Werte der Hyperparameter bei jeder Iteration."""
        hyperparam_history.append({"signal_noise": xk[0], "length_scale": xk[1], "dgl_parameter": xk[2]})

    # Kontinuum von X
    x = np.linspace(0,10,500)

    # Generiere zufällig n_trainingspoints Werte aus x
    xf = np.linspace(0,10, n_trainingspoints)
    xg = xf

    # Definiere Gausches Weißes Rauschen
    sigma_f = 0.1
    sigma_g = 0.1

    # Definiere Hyperparameter
    signal_noise = 0.5
    length_scale = 3
    # Definiere DGL-Parameter:
    dgl_parameter = 2

    hyperparameters = np.array([signal_noise, length_scale, dgl_parameter])



    # Erzeuge Zufallsrauschen (Mittelwert = 0, Standardabweichung = sigma)
    εf = np.random.normal(0, sigma_f**2, n_trainingspoints)
    εg = np.random.normal(0, sigma_g**2, n_trainingspoints)

    # Generiere Data fürs Training bzw. beobachtete Daten
    yf = lineares_pendel_dgl_trial_function(xf) + εf
    yg = source_function(xg) + εg

    # yf und yg zu y kombinieren
    y = np.concatenate((yf, yg))

    # Liste zur Speicherung der getesteten Hyperparameter
    hyperparam_history = []


    # Deklariere X_star (Werte die man predicted)
    x_star = x

    # Berechne Jacobian von Nlml → dnlml_dtheta
    grad_func_nlml = jax.grad(nlml_for_optimization, argnums=0)

    # -------------Optimierung der Hyperparameter------------------
    start_opt = time.time()
    result = minimize(nlml_for_optimization, hyperparameters, args=([xf, xg, sigma_f, sigma_g, y]), method=opt_method , jac=grad_func_nlml, callback=callback)
    opt_time = time.time() - start_opt
    hyperparameters = result.x

    if  result.success:
        print(f"⚠Optimierung success für {opt_method} mit {n_trainingspoints} Trainingspunkten. result {result}")
    if not result.success:
        print(f"⚠️ Optimierung fehlgeschlagen für {opt_method} mit {n_trainingspoints} Trainingspunkten. result {result}")


    # Berechne Covariance Matrix von Beobachteten bzw. Trainings Punkten
    Ky = covariance_matrix_training_points(xf, xg, hyperparameters[0], hyperparameters[1], sigma_f, sigma_g, hyperparameters[2])


    # Erstelle Matrizen für der prediction Punkte f_star
    k_ff_star = kff_function(xf, x_star, hyperparameters[0], hyperparameters[1])
    k_gf_star = kgf_function(xg, x_star, hyperparameters[0], hyperparameters[1], hyperparameters[2])
    k_f_star_f_star = kff_function(x_star, x_star, hyperparameters[0], hyperparameters[1])

    # Kombiniere Kovarianzen mit Prediction Points und Training Points
    k_star_ = np.concatenate((k_ff_star, k_gf_star)).T

    # Posterio Mean Trial Function
    post_mean_f_star, post_variance_f_star, log_marg_likelihood_f_star = posterio_mean_variance_lml(y, Ky, k_f_star_f_star, k_star_)

    # Posterio Vanilla Vergleich Mean Trial Function
    post_mean_f_vanilla, post_variance_f_vanilla, log_marg_likelihood_f_vanilla = posterio_mean_variance_lml(yf, kff_function(xf, xf, hyperparameters[0], hyperparameters[1]) + sigma_f ** 2 * np.identity(len(xf)), kff_function(x_star, x_star, hyperparameters[0], hyperparameters[1]), kff_function(x_star, xf, hyperparameters[0], hyperparameters[1]))


    #Berechne Mean Squareed Error:
    mse_pigp = evaluate_mean_squared_error(lineares_pendel_dgl_trial_function(x_star), post_mean_f_star)
    mse_vanilla_gp = evaluate_mean_squared_error(lineares_pendel_dgl_trial_function(x_star), post_mean_f_vanilla)

    std_pigp = calculate_total_standarddeviation(post_variance_f_star)
    std_vanilla = calculate_total_standarddeviation(post_variance_f_vanilla)
    print("Mean Squared Error Phisics Informed GP:", mse_pigp)
    print("Mean Squared Error Vanilla GP:", mse_vanilla_gp)
    print("Gesamt Standardabweichung PIGP", calculate_total_standarddeviation(post_variance_f_star))
    print("Gesamt Standardabweichung Vanilla-GP", calculate_total_standarddeviation(post_variance_f_vanilla))

    return {
        "n_train": n_trainingspoints,
        "opt_method": opt_method,
        "mse_pigp": mse_pigp,
        "std_pigp": std_pigp,
        "mse_vanilla": mse_vanilla_gp,
        "std_vanilla": std_vanilla,
        "opt_time_sec": opt_time,
        "optimized_signal_noise": hyperparameters[0],
        "optimized_length_scale": hyperparameters[1],
        "optimized_dgl_parameter": hyperparameters[2],
        "optimizer_success": result.success,
        "hyperparam_history": hyperparam_history
    }




# ==============================
#  Berechnung aller Experimente
# ==============================

if __name__ == '__main__':
    opt_methods = ["CG", "L-BFGS-B", "Newton-CG"]
    n_train_list = [5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 45, 60, 90, 120, 150, 200]

    print("🔄 Starte Berechnungen...")
    start_time = time.time()

    results = []
    for method in opt_methods:
        for n in n_train_list:
            print(f"🔄 Starte Experiment mit {method} und {n} Trainingspunkten...")
            result = run_experiment(n, method)
            results.append(result)  # Speichere das Ergebnis in der Liste

    # Speichern der Ergebnisse
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiment_results.csv", index=False)

    # Speichern aller Hyperparameter-Historien
    hyperparam_data = []
    for res in results:
        for hp in res["hyperparam_history"]:
            hyperparam_data.append({
                "n_train": res["n_train"],
                "opt_method": res["opt_method"],
                "signal_noise": hp["signal_noise"],
                "length_scale": hp["length_scale"],
                "dgl_parameter": hp["dgl_parameter"]
            })

    df_hyperparams = pd.DataFrame(hyperparam_data)
    df_hyperparams.to_csv("hyperparameter_history.csv", index=False)

    print(f"✅ Berechnungen abgeschlossen in {time.time() - start_time:.2f} Sekunden.")
    print("Ergebnisse gespeichert in `experiment_results.csv` und `hyperparameter_history.csv`")





