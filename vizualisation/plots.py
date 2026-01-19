import matplotlib.pyplot as plt


def plot_mse(mse_hist):
    plt.figure()
    plt.plot(mse_hist)
    plt.title("MSE over epochs")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.grid(True)


def plot_pid_params(kp_hist, ki_hist, kd_hist):
    plt.figure()
    plt.plot(kp_hist, label="kp")
    plt.plot(ki_hist, label="ki")
    plt.plot(kd_hist, label="kd")
    plt.title("PID parameters over epochs")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
