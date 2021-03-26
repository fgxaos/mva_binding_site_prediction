### LIBRARIES ###
# Global libraries
import numpy as np

### FUNCTION DEFINITIONS ###
def compare(predictions, truth):
    """Compares two arrays of the same size.

    Args:
        predictions: np.array
                     first array to compare
        truth:       np.array
                     second array to compare
    Returns:
        accuracy: float (in [0, 1])
                  percentage of accurate predictions
    """
    comp = predictions - truth
    return 1 - (np.count_nonzero(comp) / len(predictions))


def init_wandb(lambd, lambd_kern, subseq_length, train_split, kernel):
    """Initializes a wandb display.

    Args:

    """
    wandb.init(project="mva_kernel", reinit=True)
    wandb.config.svm_lambd = lambd
    wandb.config.subseq_length = subseq_length
    wandb.config.train_split = train_split
    wandb.config.kernel = kernel
    if kernel == "spectrum":
        wandb.run.name = "{}-lambd_{}-subseq_len_{}-train_split_{}".format(
            kernel, lambd, subseq_length, train_split
        )
    elif kernel == "substring" or self.kernel_name == "fast_substring":
        wandb.config.lambd_kern = lambd_kern
        wandb.run.name = (
            "{}-lambd_{}-subseq_len_{}-lambd_kern_{}-train_split_{}".format(
                kernel, lambd, subseq_length, lambd_kern, train_split
            )
        )