import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import seaborn as sns
import jax
import jax.numpy as jnp
from jax import grad, jacfwd

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





# Anzahl Trainings points
n_trainingspoints = 8

# Kontinuum von X
x = np.linspace(0,10,500)

# Generiere zufällig n_trainingspoints Werte aus x
#xf = np.random.choice(x[10:-10],n_trainingspoints, replace=False)
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


# Deklariere X_star (Werte die man predicted)
x_star = x

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

# Berechne Mean Squareed Error:
mse_pigp = evaluate_mean_squared_error(lineares_pendel_dgl_trial_function(x_star), post_mean_f_star)
mse_vanilla_gp = evaluate_mean_squared_error(lineares_pendel_dgl_trial_function(x_star), post_mean_f_vanilla)

print("Mean Squared Error Phisics Informed GP:", mse_pigp)
print("Mean Squared Error Vanilla GP:", mse_vanilla_gp)
print(f"Der MSE von PIGP ist {mse_vanilla_gp/mse_pigp} mal kleiner")
print("Gesamt Standardabweichung PIGP", calculate_total_standarddeviation(post_variance_f_star))
print("Gesamt Standardabweichung Vanilla-GP", calculate_total_standarddeviation(post_variance_f_vanilla))
print(f"Gesamt Standardabweichung von PIGP ist {calculate_total_standarddeviation(post_variance_f_vanilla)/calculate_total_standarddeviation(post_variance_f_star)} mal kleiner")

# Berechne Jacobian von Nlml → dnlml_dtheta
grad_func_nlml = jax.grad(nlml_for_optimization, argnums=0)

# -------------Optimierung der Hyperparameter------------------
result = minimize(nlml_for_optimization, hyperparameters, args=([xf, xg, sigma_f, sigma_g, y]), method="L-BFGS-B", jac=grad_func_nlml)

print("Optimierte Variable with jac:", result.x)
hyperparameters = result.x
if  result.success:
    print(f"⚠Optimierung success für CG mit {n_trainingspoints} Trainingspunkten. result {result}")
if not result.success:
    print(f"⚠️ Optimierung fehlgeschlagen für CG mit {n_trainingspoints} Trainingspunkten. result {result}")


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

#x_f_test, x_g_test = np.meshgrid(x_star, x_star, indexing="ij")
#contourf = plt.contourf(x_f_test, x_g_test, post_variance_f_star, cmap="viridis")
#plt.colorbar(contourf)
#plt.show()

#Berechne Mean Squareed Error:
mse_pigp = evaluate_mean_squared_error(lineares_pendel_dgl_trial_function(x_star), post_mean_f_star)
mse_vanilla_gp = evaluate_mean_squared_error(lineares_pendel_dgl_trial_function(x_star), post_mean_f_vanilla)

print("Mean Squared Error Phisics Informed GP:", mse_pigp)
print("Mean Squared Error Vanilla GP:", mse_vanilla_gp)
print(f"Der MSE von PIGP ist {mse_vanilla_gp/mse_pigp} mal kleiner")
print("Gesamt Standardabweichung PIGP", calculate_total_standarddeviation(post_variance_f_star))
print("Gesamt Standardabweichung Vanilla-GP", calculate_total_standarddeviation(post_variance_f_vanilla))
print(f"Gesamt Standardabweichung von PIGP ist {calculate_total_standarddeviation(post_variance_f_vanilla)/calculate_total_standarddeviation(post_variance_f_star)} mal kleiner")




"""
# Erstelle Matrizen für prediction Punkte g_star
k_fg_star = kfg_function(xf, x_star, hyperparameters[0], hyperparameters[1], dgl_parameter)
k_gg_star = kgg_function(xg, x_star, hyperparameters[0], hyperparameters[1], dgl_parameter)
k_g_star_g_star = kgg_function(x_star, x_star, hyperparameters[0], hyperparameters[1], dgl_parameter)

# Kombiniere Kovarianzen mit Prediction Points und Training Points
k_star_ = np.concatenate((k_fg_star, k_gg_star)).T

# Posterio Mean Trial Function
post_mean_g_star, post_variance_g_star, log_marg_likelihood_g_star = log_marginal_likelihood(y, Ky, k_g_star_g_star, k_star_)
"""

# Plot
# Subplots erstellen (1 Zeile, 2 Spalten)
fig, axes = plt.subplots(2,1, figsize=(12,8))

# Erster Plot: Lösungsfunktion
axes[0].plot(x, lineares_pendel_dgl_trial_function(x), label=r"Analytical solution $u$(x)", color="blue", linestyle='--', alpha=1, linewidth=2)
axes[0].plot(x, post_mean_f_star, label=r'Predicted Solution $u_*(x)$', color="red", linewidth=2, alpha=0.7)
axes[0].fill_between(
    x_star,
    post_mean_f_star - 2 * np.diag(post_variance_f_star),
    post_mean_f_star + 2 * np.diag(post_variance_f_star),
    alpha=0.2,
    color="red",
    label=r"2$\sigma$ Confidence Interval"
)
#axes.plot(x, post_mean_f_vanilla, label="post mean f_vanilla(x)", color="red")
#axes.fill_between(
#    x_star,
#    post_mean_f_vanilla - 2 * np.diag(post_variance_f_vanilla),
#    post_mean_f_vanilla + 2 * np.diag(post_variance_f_vanilla),
#    alpha=0.2,
#    color="red",
#    label="2-Sigma Confidence Interval"
#)
axes[0].scatter(xf,yf, color="black", marker="x", label="Training points", linewidth=3)
axes[0].axhline(0, color="grey", linestyle="--")  # Horizontale Linie bei 0
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_xlabel("x", fontsize=14)
axes[0].set_ylabel("u(x)", fontsize=14)
#axes.set_title("Lösungsfunktion des linearen Pendels")
axes[0].legend(fontsize=12)
axes[0].grid()


# Zweiter Plot: Source Function
axes[1].plot(x, source_function(x), label = "Source function f(x)", color = "red")
#axes[1].plot(x, post_mean_g_star, label="post mean f_star(x)", color="green")
axes[1].scatter(xg,yg, color="black", marker = "x", label="Training points")
axes[1].axhline(0, color="grey", linestyle="--")  # Horizontale Linie bei 0
axes[1].set_ylim(-1.5, 1.5)
axes[1].set_xlabel("x", fontsize=14)
axes[1].set_ylabel("f(x)", fontsize=14)
#axes[1].set_title("Source Function der DGL")
axes[1].legend(fontsize=12)
axes[1].grid()


plt.savefig("lineares_pendel.png")
plt.show()


# ---------------- Plotte nlml gegen hypers dgl_para = fixed
"""
args = (xf, xg, sigma_f, sigma_g, y)

# Wähle einen festen Wert für dgl_parameter
dgl_fixed = 1.0

# Wertebereiche für Signal-Noise und Length-Scale
signal_noise_values = jnp.linspace(1, 2.0, 100)  # 50 Werte zwischen 0.1 und 2.0
length_scale_values = jnp.linspace(1, 2.0, 100)

# Erstelle ein Gitter für die Werte
Z = jnp.zeros((len(signal_noise_values), len(length_scale_values)))

# Berechne NLML für jedes (signal_noise, length_scale)-Paar
for i, signal_noise in enumerate(signal_noise_values):
    for j, length_scale in enumerate(length_scale_values):
        hyp = (signal_noise, length_scale, dgl_fixed)
        Z = Z.at[i, j].set(nlml_for_optimization(hyp, args))

# Erstelle eine Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(Z.T, extent=[1, 2.0, 1, 2.0], origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(label="Negative Log Marginal Likelihood (NLML)")
plt.xlabel("Signal Noise")
plt.ylabel("Length Scale")
plt.title("NLML Heatmap für dgl_parameter = {:.2f}".format(dgl_fixed))
plt.show()

"""





