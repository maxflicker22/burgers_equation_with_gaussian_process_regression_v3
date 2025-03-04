import time
import numpy as np
import matplotlib.pyplot as plt
import itertools
import csv
import os
from scipy.optimize import minimize
from scipy.stats import qmc
from cleanNewProject.config import Modelvariables
from cleanNewProject.likelihood import likelihood
from cleanNewProject.predictor import predictor
from cleanNewProject.Exact.exact_solution import burgers_viscous_time_exact1


# Your main function encapsulating the entire code
def run_burgers_experiment(nu, dt, number_generated_points, jitter, noise_n, optimization_method, csv_writer):
    class StopOptimization(Exception):
        pass

    # Callback function to stop optimization
    def callback(hyp):
        value, gradient = likelihood(hyp)
        if value is None or gradient is None:
            raise StopOptimization("None value encountered, stopping optimization")

    def init_pde_solution_func(x):
        u = np.sin(-np.pi * x)
        return u

    dim = Modelvariables['D']
    lb = np.array([-1] * dim)  # Lower bounds
    ub = np.array([1] * dim)  # Upper bounds

    Modelvariables['noise_b'] = 0
    Modelvariables['noise_n'] = noise_n

    init_guess = np.log([5, 5, np.exp(-5)])
    Modelvariables['hyp'] = init_guess

    Modelvariables['nu'] = nu
    Modelvariables['dt'] = dt

    Modelvariables['Ngp'] = number_generated_points

    number_measured_points = number_generated_points
    Modelvariables['Nmp'] = number_measured_points

    Modelvariables['jitter'] = jitter

    Modelvariables['S0'] = np.zeros((number_measured_points, number_measured_points))

    T = 1
    nsteps = int(T / Modelvariables['dt'])

    num_plots = 6

    # number of points in x which are for plotting and for relative error evalution
    num_x_plots = 500
    x_plots = np.linspace(-1,1,num_x_plots)


    # Parameters _Exact Solution
    vxn = num_x_plots
    vx = np.linspace(-1.0, 1.0, vxn)

    vt = np.linspace(0.0, 1.0, nsteps)

    x_b = np.array([[-1], [1]])
    Modelvariables['x_b'] = x_b

    u_b = np.array([[0], [0]])
    u_b = u_b + Modelvariables['noise_b'] * np.random.randn(*u_b.shape)
    Modelvariables['u_b'] = u_b

    xstar = np.linspace(-0.9, 0.9, number_generated_points).reshape(-1, 1)

    #x_u = lb + lhs_samples * (ub - lb)
    x_u = xstar
    Modelvariables['x_u'] = x_u
    init_x_u = x_u

    u = init_pde_solution_func(x_u)
    u = u + np.random.normal(0, Modelvariables['noise_n'], len(x_u)).reshape(-1, 1)

    Modelvariables['u'] = u
    init_u = u

    # Compute the exact solution
    vu = burgers_viscous_time_exact1(nu, vxn, vx, nsteps, vt)

    # Create lists to track data
    errors, optimized_hyp, predicted_solutions, analytic_solutions = [], [], [], []

    # Plot the initial data
    fontsize = 18
    fig, axes = plt.subplots(num_plots // 2, 2, figsize=(15, 20))


    axes = axes.flatten()

    j = 0
    axes[j].plot(vx, vu[:, 0], label='Exact Solution')
    axes[j].set_xlabel('Space (x)', fontsize = fontsize)
    axes[j].set_ylabel('U(x,t)', fontsize = fontsize)
    axes[j].grid()
    axes[j].set_ylim([-1.5, 1.5])
    axes[j].plot(init_x_u, init_u, 'rx', linewidth=1, label="measured")
    axes[j].legend()
    axes[j].set_title(f"Time: {0}s", fontsize = fontsize)
    plt.savefig(f'./Figures/Burgers_Final_TrainingsPoints_{number_generated_points}_nu_{nu:.6f}_dt_{dt}_noise_{noise_n}_jitter_{jitter}_method_{optimization_method}.png', dpi=300)

    plots_equal_distance = np.linspace(0, nsteps - 1, num_plots, endpoint=True, dtype=int)

    # Initialize timing
    start_time = time.time()

    for step, delta_t in enumerate(vt):
        if optimization_method != "none":
            try:
                result = minimize(
                    lambda hyp: likelihood(hyp)[0],
                    Modelvariables['hyp'],
                    jac=lambda hyp: likelihood(hyp)[1],
                    method=optimization_method,
                    callback=lambda hyp: callback(hyp),
                    options={'maxiter': 5000}
                )

                Modelvariables['hyp'] = result.x
                print("Optimization result:", result)
                print("Optimized hyperparameters:", result.x)

            except StopOptimization as e:
                print("Optimizing failed")
                print(str(e))



        NLML, DNLML = likelihood(Modelvariables['hyp'])

        print(f"Optimal NLML Value: {NLML}")
        print(f"Optimal DNLML Value: {DNLML}")

        Kpred, Kvar = predictor(x_plots)

        # Fill List with Values
        errors.append(np.linalg.norm(Kpred[:, 0] - vu[:, step]) / np.linalg.norm(vu[:, step]))
        optimized_hyp.append(Modelvariables['hyp'].tolist())
        predicted_solutions.append(Kpred[:, 0].tolist())
        analytic_solutions.append(vu[:, step].tolist())

        x_u = xstar
        Modelvariables['u'], Modelvariables['S0'] = predictor(x_u)
        Modelvariables['x_u'] = x_u

        if step in plots_equal_distance:
            if step != 0:
                j += 1
                axes[j].set_title(f"Time: {delta_t:.2f}s", fontsize = fontsize)
                axes[j].plot(vx, vu[:, step], label=f'Exact Solution')
                axes[j].set_xlabel('Space (x)', fontsize = fontsize)
                axes[j].set_ylabel('U(x,t)', fontsize = fontsize)
                axes[j].grid()
                axes[j].set_ylim([-1.5, 1.5])

                std_dev = np.sqrt(np.diag(np.abs(Kvar)))
                axes[j].plot(x_plots, Kpred, 'r--', linewidth=1, label="predicted")

                axes[j].fill_between(x_plots.flatten(), (Kpred[:, 0] - 1.96 * std_dev), (Kpred[:, 0] + 1.96 * std_dev), color='red', alpha=0.2)
                axes[j].legend()
                plt.tight_layout()
                plt.savefig(f'./Figures/Burgers_Final_TrainingsPoints_{number_generated_points}_nu_{nu:.6f}_dt_{dt}_noise_{noise_n}_jitter_{jitter}_method_{optimization_method}.png', dpi=300)
                print("!!!!!!!!!!!PLOTTET!!!!!!!!!!")

        # Calculate the error (Relative L2 Norm)
        try:
            print(f'Step: {step}, Time = {delta_t:.6f}, NLML = {NLML.item():.6e}, error_u = {errors[step].item():.6e}')
        except Exception as e:
            print(f"Error while printing NLML and results ocurred: {e}")


        # Calculate and print progress and estimated time remaining
        elapsed_time = time.time() - start_time
        percent_complete = (step + 1) / nsteps
        estimated_time_per_percent_complete = elapsed_time / percent_complete
        percented_remaining = 1 - percent_complete
        estimated_time_remaining = estimated_time_per_percent_complete * percented_remaining
        print(f"Progress: {percent_complete:.2%} complete. Estimated time remaining: {estimated_time_remaining:.2f} seconds.")

    plt.tight_layout()
    plt.savefig(f'./Figures/Burgers_Final_TrainingsPoints_{number_generated_points}_nu_{nu:.6f}_dt_{dt}_noise_{noise_n}_jitter_{jitter}_method_{optimization_method}.png', dpi=300)
    plt.show()

    # Plotting the error over time
    plt.figure(figsize=(10, 6))
    plt.plot(vt, errors, label='Relative Error (Normalized L2 Norm)', color='b', linestyle='-', marker='o')
    plt.xlabel('Time [s]' , fontsize = fontsize)
    plt.ylabel('Relative Error', fontsize = fontsize)
    plt.title(f'Error Between Predicted and Exact Solutions Over Time\nnu={nu:.6f}, dt={dt}, Ngp={number_generated_points}, noise={noise_n}, jitter={jitter}, method={optimization_method}', fontsize = fontsize)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Figures/Burgers_Relative_Error_Over_Time_nu_{nu:.6f}_dt_{dt}_Ngp_{number_generated_points}_noise_{noise_n}_jitter_{jitter}_method_{optimization_method}.png', dpi=300)
    plt.show()



    # Save the final error values
    mse_over_time = np.mean(np.array(errors) ** 2)
    csv_writer.writerow(
        [mse_over_time, nu, dt, jitter, optimization_method, number_generated_points, optimized_hyp, errors,
         analytic_solutions, predicted_solutions])


# Define the parameter grid
nus = [0.01/np.pi, 1, 0.1]
dts = [1e-3]
number_generated_points_list = [57, 87, 107, 127]
jitters = [1e-7]
noise_ns = [0.03]
optimization_methods = ['Newton-CG', 'L-BFGS-B']

nus = [0.01/np.pi]
dts = [1e-3]
number_generated_points_list = [127]
jitters = [1e-7]
noise_ns = [0.03]
optimization_methods = ['L-BFGS-B']


csv_file = './Figures/Burgers_Experiment_Results.csv'
file_exists = os.path.isfile(csv_file)

with open(csv_file, 'a', newline='') as f:
    csv_writer = csv.writer(f)
    if not file_exists:
        csv_writer.writerow(
            ["mse", "nu", "dt", "jitter", "opt_method", "n_train", "opt_hypers", "relativ_error", "analytic_solution",
             "predicted_solution"])

    for params in itertools.product(nus, dts, number_generated_points_list, jitters, noise_ns, optimization_methods):
        run_burgers_experiment(*params, csv_writer)
