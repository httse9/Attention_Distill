import matplotlib.pyplot as plt
from os.path import join

def plot_distillation_linear_layers_loss(losses, path):
    """
    Plot the losses for all linear layers for distillation.

    :param losses: numpy array containing losses of each layer. shape: (#layers, #epochs)
    :param path: path to save the plotted figure.
    :type path: str
    """

    n_epochs = losses.shape[1]
    x = list(range(1, n_epochs + 1))
    for i, layer_loss in enumerate(losses):

        plt.plot(x, layer_loss, label=f"layer{i+1}")
    
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSELoss")
    plt.savefig(join(path, "loss.png"))
    plt.clf()

