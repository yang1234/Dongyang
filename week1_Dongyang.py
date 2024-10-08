import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference
from scipy import stats

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference

    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    kp = 1000  # PD controller proportional gain
    kd = 100   # PD controller derivative gain

    # Initialize data storage
    tau_mes_all = []           # Measured torque storage
    regressor_all = []         # Dynamic regressor storage

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity

        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        # Compute regressor and store it
        if current_time > 1:
            cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
            regressor_all.append(cur_regressor)
        
            tau_mes_column = tau_mes.reshape(-1, 1)  # Convert 1D array (7,) to 2D column vector (7, 1)
            tau_mes_all.append(tau_mes_column)  # Append the converted column vector to the list

            print(f"Added tau_mes shape: {np.array(tau_mes).shape}, Total tau_mes_all: {len(tau_mes_all)}")
            print(f"Added cur_regressor shape: {np.array(cur_regressor).shape}, Total regressor_all: {len(regressor_all)}")

        current_time += time_step

    # After data collection, save the data as NumPy array files
    tau_mes_all = np.vstack(tau_mes_all)  # Shape: (7n, 1)
    regressor_all = np.vstack(regressor_all)  # Shape: (7n, 70)
    print(f"Shape of tau_mes_all: {tau_mes_all.shape}")
    print(f"Shape of regressor_all: {regressor_all.shape}")

    print(f"Current time in seconds: {current_time:.2f}")
    # After data collection, stack all the regressor and torque data and compute parameters 'a' using pseudoinverse
    a = np.linalg.pinv(regressor_all) @ tau_mes_all  # Compute dynamic parameters
    print(f"Estimated parameters: {a.flatten()}")  # Print estimated parameters
    # Compute the model output using estimated parameters 'a'
    predicted_tau = regressor_all @ a  # predicted output

    # Compare predicted torque with measured torque
    residuals = tau_mes_all - predicted_tau  # residual
    print(f"Shape of residuals: {residuals.shape}")

    # Compute the metrics for the linear model
    SS_res = np.sum(residuals ** 2)  # Residual sum of squares
    SS_tot = np.sum((tau_mes_all - np.mean(tau_mes_all)) ** 2)  # Total sum of squares
    R_squared = 1 - (SS_res / SS_tot)  # R-squared value

    # Adjusted R-squared:
    n = tau_mes_all.shape[0]  # Number of samples
    p = regressor_all.shape[1]  # Number of predictors
    adjusted_R_squared = 1 - (1 - R_squared) * (n - 1) / (n - p - 1)  
    F_statistic = (SS_tot - SS_res) / p / (SS_res / (n - p - 1))  # F-statistic

    # Compute standard errors of the parameters
    residual_variance = SS_res / (n - p - 1)  # Residual variance
    covariance_matrix = residual_variance * np.linalg.pinv(np.dot(regressor_all.T, regressor_all))  # Parameter covariance matrix
    standard_errors = np.sqrt(np.diag(covariance_matrix))  # Standard errors

    # Compute 95% confidence intervals
    confidence_level = 0.95
    t_value = stats.t.ppf((1 + confidence_level) / 2, n - p - 1)  # Critical value of the t-distribution
    confidence_intervals = np.column_stack((a - t_value * standard_errors, a + t_value * standard_errors))  # Confidence intervals

    # Print results
    print(f"Estimated parameters (a): {a}")
    print(f"R-squared: {R_squared}")
    print(f"Adjusted R-squared: {adjusted_R_squared}")
    print(f"F-statistic: {F_statistic}")
    print(f"95% Confidence intervals for parameters:\n{confidence_intervals}")

    # TODO plot the torque prediction error for each joint (optional)
    # Convert residual matrix to a format convenient for visualization
    # Because the shape of the residual matrix is (7n, 1), convert it back to (n, 7)
    num_time_steps = len(tau_mes_all) // 7  # Total number of time steps

    residuals_matrix = residuals.reshape(num_time_steps, 7)

    # Plot the torque error for each joint
    # Use integer indices instead of actual time
    time_vector = np.arange(num_time_steps)

    fig1, axs1 = plt.subplots(7, 1, figsize=(10, 20), sharex=True)
    for i in range(7):
        axs1[i].plot(time_vector, residuals_matrix[:, i], label=f'Joint {i+1}')
        axs1[i].set_ylabel('Torque Error')
        axs1[i].legend()

    axs1[-1].set_xlabel('Time Steps')
    plt.suptitle('Torque Prediction Error for Each Joint')
    plt.show()

    # Assuming measured_torque and predicted_torque are ready
    # This is typically after data processing and torque prediction calculation
    measured_torque = tau_mes_all.reshape(num_time_steps, 7)  # Ensure tau_mes_all is the correct shape

    # Convert predicted_tau to the correct shape to match time steps
    predicted_torque = predicted_tau.reshape(num_time_steps, 7)  # Ensure predicted_tau is the correct shape
    # Ensure the time vector and torque data lengths match
    time_vector = np.arange(num_time_steps)

    fig2, axs2 = plt.subplots(7, 1, figsize=(10, 20), sharex=True)
    for i in range(7):
        axs2[i].plot(time_vector, measured_torque[:, i], label='Measured Torque')
        axs2[i].plot(time_vector, predicted_torque[:, i], linestyle='--', label='Predicted Torque')
        axs2[i].set_title(f'Joint {i+1} Torque Comparison')
        axs2[i].legend()

    axs2[-1].set_xlabel('Time Steps')
    plt.suptitle('Original vs Predicted Torque for Each Joint')
    plt.show()

if __name__ == '__main__':
    main()
