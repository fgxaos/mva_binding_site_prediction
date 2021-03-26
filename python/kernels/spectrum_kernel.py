### LIBRARIES ###
import itertools
from functools import partial
from multiprocessing import Process, Pool
import multiprocessing.managers
from tqdm import tqdm

import numpy as np

### MULTIPROCESSING INITIALIZATION ###
class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register("np_zeros", np.zeros, multiprocessing.managers.ArrayProxy)

### CLASS DEFINITION ###
class SpectrumKernel:
    """Implements the Spectrum kernel."""

    def __init__(self, subseq_length, parallel):
        """Initiates the Spectral kernel

        Args:
            subseq_length: int
                length of the subsequences to consider
            parallel: boolean
                whether to use multiprocessing
        """
        self.subseq_length = subseq_length
        self.letters = ["A", "C", "G", "T"]
        self.parallel = parallel

    def initiate_with_X(self, X):
        """Initiates the variables when the Gram matrix is loaded.

        Args:
            X: List[str]
                list of strings which will be considered
        """
        # Compute the different subsequences
        self.compute_str_dict(X)
        self.len_alphabet = len(self.subseq_dict)
        self.phi_matrix = self.compute_phis(X)

    def compute_str_dict(self, list_strings):
        """Computes the alphabet of subsequences.

        Args:
            list_strings: List[str]
                strings to consider to build the dictionary.
        """
        self.subseq_dict = {}
        subseq_idx = 0
        for subseq in itertools.product(self.letters, repeat=self.subseq_length):
            if subseq not in self.subseq_dict:
                self.subseq_dict["".join(subseq)] = subseq_idx
                subseq_idx += 1

    def phi(self, x):
        """Computes phi_u(x) for each possible subsequence.

        Args:
            x: str
                string on which to count the number of occurrences
        Returns:
            phi_x: np.array(len_alphabet)
                value of phi_{subsequence}(x) for each possible subsequence
        """
        phi_x = np.zeros(self.len_alphabet)

        for i in range(0, len(x) - self.subseq_length + 1):
            subseq = x[i : i + self.subseq_length]
            phi_x[self.subseq_dict[subseq]] += 1

        return phi_x

    def compute_phis(self, X, verbose=True):
        """Compute phi_u for each element of X.

        Args:
            X: List[str]
                list of strings to use to compute the phi_u
        Returns:
            phi_matrix: np.array
                matrix with the phi_u values for each element of X
        """
        len_X = len(X)
        phi_matrix = np.zeros((len_X, self.len_alphabet))

        if verbose:
            for i in tqdm(range(len_X), desc="Computing phi matrix"):
                phi_matrix[i] = self.phi(X[i])
        else:
            for i in range(len_X):
                phi_matrix[i] = self.phi(X[i])

        return phi_matrix

    def gram_element(self, gram_matrix, idx):
        """Computes an element of the Gram matrix.

        Args:
            gram_matrix: multiprocessing.managers.ArrayProxy
                array representing the Gram matrix
            idx: (int, int)
                indexes of the Gram matrix to fill
        """
        gram_matrix[idx[0], idx[1]] = self.phi_matrix[idx[0]] @ self.phi_matrix[idx[1]]
        gram_matrix[idx[1], idx[0]] = gram_matrix[idx[0], idx[1]]

    def compute_gram(self, X):
        """Computes the Gram matrix associated with the input X.

        Args:
            X: List[str]
                list of strings to use to compute the Gram matrix
        Returns:
            gram_matrix: np.array
                Gram matrix for the given list of strings
        """
        # First, compute all the possible phi_u(x)
        self.initiate_with_X(X)

        # Then, compute each K_{ij}
        len_X = len(X)
        if self.parallel:
            m = MyManager()
            m.start()
            gram_matrix = m.np_zeros((len_X, len_X), dtype=int)
            pool = Pool(multiprocessing.cpu_count())
            par_func = partial(self.gram_element, gram_matrix)

            for i in tqdm(range(len_X), desc="Computing Gram matrix"):
                run_list = [(i, j) for j in range(i, len_X)]
                _ = pool.map(par_func, run_list)

            self.gram_matrix = np.asarray(gram_matrix)

        else:
            self.gram_matrix = np.zeros((len_X, len_X))
            for i in tqdm(range(len_X), desc="Computing Gram matrix"):
                for j in range(i, len_X):
                    self.gram_matrix[i, j] = self.phi_matrix[i] @ self.phi_matrix[j]
                    self.gram_matrix[j, i] = self.gram_matrix[i, j]

        return self.gram_matrix

    def K(self, x1, x2):
        """Computes the substring kernel between x1 and x2.

        Args:
            x1: str
                first string
            x2: str
                second string
        Returns:
            the substring kernel between the two strings
        """
        phi_matrix = self.compute_phis([x1, x2], verbose=False)
        return phi_matrix[0] @ phi_matrix[1]