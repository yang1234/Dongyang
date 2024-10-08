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
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []           #U
    regressor_all = []         #Y
    
    
    # Data collection loop
    while current_time  < max_time:
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
            for index in range(len(sim.bot)):  # Conditionally display the robot model
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
        #print(f"cur_regressor shape before append: {cur_regressor.shape}") 
        
        # TODO: Compute regressor and store it
            regressor_all.append(cur_regressor)  
        
            tau_mes_column = tau_mes.reshape(-1, 1)  # 将一维数组 (7,) 转换为二维列向量 (7, 1)
            tau_mes_all.append(tau_mes_column)  # 添加转换后的列向量到列表




            print(f"Added tau_mes shape: {np.array(tau_mes).shape}, Total tau_mes_all: {len(tau_mes_all)}")
            print(f"Added cur_regressor shape: {np.array(cur_regressor).shape}, Total regressor_all: {len(regressor_all)}")

            #  current time
        current_time += time_step
        # Convert lists to NumPy arrays ensuring proper shapes
    # 在循环结束后，将数据保存为NumPy数组文件
    tau_mes_all = np.vstack(tau_mes_all)  # 将形状  转为 (7n, 70)
    regressor_all = np.vstack(regressor_all)  # 转为 (7n, 1)
    print(f"Shape of tau_mes_all: {tau_mes_all.shape}")
    print(f"Shape of regressor_all: {regressor_all.shape}")

    print(f"Current time in seconds: {current_time:.2f}")  
    # After data collection, stack all the regressor and all the torque and compute the parameters 'a' using pseudoinverse
    a = np.linalg.pinv(regressor_all) @ tau_mes_all  # 计算动态参数
    print(f"Estimated parameters: {a.flatten()}")  # 打印估计的参数
     # 使用估计的参数 a 来计算模型输出
    predicted_tau = regressor_all @ a  # predicted output

    # 比较预测的扭矩与实际测量的扭矩
    residuals = tau_mes_all - predicted_tau  # residual
    print(f"Shape of residuals: {residuals.shape}")
    # TODO compute the metrics for the linear model
      # R_squared:
    SS_res = np.sum(residuals ** 2)  # 残差平方和
    SS_tot = np.sum((tau_mes_all - np.mean(tau_mes_all)) ** 2)  # 总平方和
    R_squared = 1 - (SS_res / SS_tot)  # R 平方值

     # adjusted R_squared:
    n = tau_mes_all.shape[0]  # 样本数量
    p = regressor_all.shape[1]  # 自变量数量
    adjusted_R_squared = 1 - (1 - R_squared) * (n - 1) / (n - p - 1)  
    F_statistic = (SS_tot - SS_res) / p / (SS_res / (n - p - 1))  # F 

    # 计算参数的标准误差
    residual_variance = SS_res / (n - p - 1)  # 残差方差
    covariance_matrix = residual_variance * np.linalg.pinv(np.dot(regressor_all.T, regressor_all))  # 参数协方差矩阵
    standard_errors = np.sqrt(np.diag(covariance_matrix))  # 标准误差

    # 计算 95% 置信区间
    confidence_level = 0.95
    t_value = stats.t.ppf((1 + confidence_level) / 2, n - p - 1)  # t 分布的临界值
    confidence_intervals = np.column_stack((a - t_value * standard_errors, a + t_value * standard_errors))  # 置信区间

    # 打印结果
    print(f"Estimated parameters (a): {a}")
    print(f"R-squared: {R_squared}")
    print(f"Adjusted R-squared: {adjusted_R_squared}")
    print(f"F-statistic: {F_statistic}")
    print(f"95% Confidence intervals for parameters:\n{confidence_intervals}")
   
    
    
    # TODO plot the  torque prediction error for each joint (optional)
    
    # 转换残差矩阵为可视化方便的格式
    # 因为残差矩阵形状为 (7n, 1)，需要转换回 (n, 7)
    num_time_steps = len(tau_mes_all) // 7  # 总的时间步数
    
    residuals_matrix = residuals.reshape(num_time_steps, 7)

    # 绘制每个关节的扭矩误差
    # 使用整数索引代替实际时间
    time_vector = np.arange(num_time_steps)

    # 绘图代码调整
    fig1, axs1 = plt.subplots(7, 1, figsize=(10, 20), sharex=True)
    for i in range(7):
        axs1[i].plot(time_vector, residuals_matrix[:, i], label=f'Joint {i+1}')
        axs1[i].set_ylabel('Torque Error')
        axs1[i].legend()

    axs1[-1].set_xlabel('Time Steps')
    plt.suptitle('Torque Prediction Error for Each Joint')
    plt.show()

    # 假设 measured_torque 和 predicted_torque 已经准备好
    # 这通常在数据处理和预测扭矩计算之后
    measured_torque = tau_mes_all.reshape(num_time_steps, 7)  # 确保 tau_mes_all 是正确的形状

    # 转换 predicted_tau 为正确的形状以匹配时间步
    predicted_torque = predicted_tau.reshape(num_time_steps, 7)  # 确保 predicted_tau 是正确的形状
    # 确保时间向量和扭矩数据的长度一致
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
