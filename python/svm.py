### LIBRARIES ###
# Global libraries
import os
from tqdm import tqdm
from functools import partial
from multiprocessing import Process, Pool
import multiprocessing.managers

import cvxpy as cp

import numpy as np

# Custom libraries
from kernels.substring_kernel import SubstringKernel
from kernels.fast_substring_kernel import FastSubstringKernel
from kernels.spectrum_kernel import SpectrumKernel
from utils.errors import KernelError

### MULTIPROCESSING PARAMETRIZATION ###
class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register("np_zeros", np.zeros, multiprocessing.managers.ArrayProxy)

### CLASS DEFINITION ###
class SVM:
    def __init__(
        self, lambd, lambd_kern=0.5, subseq_length=3, kernel="spectrum", parallel=False
    ):
        """Initiates the SVM class.

        Args:
            lambd: float
                lambda to use for the SVM
            lambd_kern: float
                lambda used in the substring kernel
            subseq_length: int
                length of the subsequence to consider in the kernel
            kernel: str
                name of the kernel to use, in ["spectrum", "substring"]
            parallel: Boolean
                Whether to use multiprocessing to compute
        """
        if kernel == "spectrum":
            self.kernel = SpectrumKernel(subseq_length, parallel)
        elif kernel == "substring":
            self.kernel = SubstringKernel(lambd_kern, subseq_length, parallel)
        elif kernel == "fast_substring":
            self.kernel = FastSubstringKernel(lambd_kern, subseq_length, parallel)
        else:
            raise KernelError(kernel)

        self.lambd = lambd
        self.subseq_length = subseq_length
        self.parallel = parallel
        self.kernel_name = kernel

    def solve_optim_problem(self, Y):
        """Finds optimal alpha to solve SVM dual problem.

        Args:
            Y: np.array
               targets in {0, 1} to train the SVM
        """
        # Define problem parameters
        n = Y.shape[0]
        alpha = cp.Variable(n)

        # Replaces the targets values by values in {-1, 1}
        svm_Y = 2 * Y - 1

        sum_term = 2 * alpha @ svm_Y - cp.quad_form(alpha, self.K)
        C = 1 / (2 * n * self.lambd)
        constraints = [0 <= cp.multiply(svm_Y, alpha), cp.multiply(Y, alpha) <= C]

        obj = cp.Maximize(sum_term)
        prob = cp.Problem(obj, constraints)
        _ = prob.solve()
        self.alpha = alpha.value

    def fit(self, X, Y, split_idx, k_path=None):
        """Fits the SVM object to the input data.

        Args:
            X: List[str]
               list of strings to pass to the kernel
            Y: np.array
               targets associated to X, in {0, 1}
            split_idx: int
                if 0, considers the whole dataset as training
                otherwise, only trains with the first `split_idx` elements
            k_path: str
                path to load or save the kernel Gram matrix
        """
        if os.path.exists(k_path):
            # Load the Gram matrix
            K = np.load(k_path)
        else:
            # Compute and save the matrix K
            K = self.kernel.compute_gram(X)
            with open(k_path, "wb") as f:
                np.save(f, K)

        if split_idx == 0:
            self.fit_X = X
            self.K = K
            if self.kernel_name == "spectrum":
                self.kernel.initiate_with_X(X)
            self.solve_optim_problem(Y)
        else:
            self.fit_X = X[:split_idx]
            if self.kernel_name == "spectrum":
                self.kernel.initiate_with_X(X[:split_idx])
            self.K = K[:split_idx, :split_idx]
            self.solve_optim_problem(Y[:split_idx])

    def compute_pred_idx(self, y_pred, args):
        """Computes the prediction of a single input

        Multiprocessing function to handle the computation of
        the prediction, and writing the result in the ArrayProxy.

        Args:
            y_pred: ArrayProxy
                array with the predicted classes
            args: (int, str)
                tuple with the index of a support vector,
                and a string to process with the kernel
        """
        (
            idx,
            X_idx,
        ) = args
        if self.kernel_name == "spectrum":
            phi_x = self.kernel.phi(X_idx)
            pred = self.alpha @ (self.kernel.phi_matrix @ phi_x)
            y_pred[idx] = int(pred >= 0)

        elif self.kernel_name == "substring" or self.kernel_name == "fast_substring":
            for j in range(len(self.fit_X)):
                y_pred[idx] += self.alpha[j] * self.kernel.K(
                    self.subseq_length, self.fit_X[j], X_idx
                )

    def predict(self, X):
        """Predicts the binary class of each element of X.

        Args:
            X: List[str]
                elements whose class we want to predict
        Output:
            y_pred: np.array
                class of each element of X, in {0, 1}
        """
        if self.kernel_name == "spectrum":
            if self.parallel:
                # Initiate multiprocessing
                m = MyManager()
                m.start()
                y_pred = m.np_zeros(len(X), dtype=int)
                pool = Pool(multiprocessing.cpu_count())
                par_func = partial(self.compute_pred_idx, y_pred)
                # Prepare list of operations to execute
                run_list = [(i, X[i]) for i in range(len(X))]
                list_results = pool.map(par_func, run_list)
                # Convert `y_pred` to an array
                y_pred = np.asarray(y_pred).squeeze()

            else:
                y_pred = np.zeros(len(X), dtype=int)
                for idx, X_idx in tqdm(enumerate(X), desc="Predicting the classes..."):
                    phi_x = self.kernel.phi(X_idx)
                    pred = self.alpha @ (self.kernel.phi_matrix @ phi_x)
                    y_pred[idx] = int(pred >= 0)
                y_pred = y_pred.squeeze()

        elif self.kernel_name == "substring" or self.kernel_name == "fast_substring":
            if self.parallel:
                # Initiate multiprocessing
                m = MyManager()
                m.start()
                y_pred = m.np_zeros(len(X), dtype=float)
                pool = Pool(multiprocessing.cpu_count())
                par_func = partial(self.compute_pred_idx, y_pred)
                # Prepare list of operations to execute
                run_list = [(i, X[i]) for i in range(len(X))]
                list_results = pool.map(par_func, run_list)
                # Convert `y_pred` to an array
                y_pred = np.asarray(y_pred).squeeze()
        else:
            raise KernelError(self.kernel_name)

        return y_pred
