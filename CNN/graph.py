import matplotlib.pyplot as plt
from matplotlib import style
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np

style.use("ggplot")
model_name = "model-[14-10-7-5]-[256-256-256-256]-1612774504"

def create_acc_loss_graph(model_name):

    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))
            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

    # accuracies = np.array(accuracies)
    # losses = np.array(losses)
    # val_accs = np.array(val_accs)
    # val_losses = np.array(val_losses)

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)
#
    ax1.plot(times, gaussian_filter1d(accuracies, sigma=10), label="acc")
    ax1.plot(times, gaussian_filter1d(val_accs, sigma=10), label="val_acc")
    ax1.legend(loc=2)
#
    ax2.plot(times, gaussian_filter1d(losses, sigma=10), label="loss")
    ax2.plot(times, gaussian_filter1d(val_losses, sigma=10), label="val_loss")
    ax2.legend(loc=2)

    plt.show()

create_acc_loss_graph(model_name)