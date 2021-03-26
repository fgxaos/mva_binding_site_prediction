### LIBRARIES ###
# Global libraries
from functools import lru_cache, partial
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Pool
import multiprocessing.managers


class MyManager(multiprocessing.managers.BaseManager):
    pass


MyManager.register("np_zeros", np.zeros, multiprocessing.managers.ArrayProxy)

### VARIABLES DEFINITION ###
cache_size = 524288

### CLASS DEFINITION ###
class FastSubstringKernel:
    """Implements the Substring kernel."""

    def __init__(self, lambd, subseq_length, parallel):
        """Initiates the substring kernel.

        Args:
            lambd: float
                lambda parameter in the substring kernel
            subseq_length: int
                length of the subsequence to consider
            parallel: bool
                whether to use multiprocessing
        """
        self.lambd = lambd
        self.subseq_length = subseq_length
        self.parallel = parallel

    @lru_cache(maxsize=cache_size)
    def K(self, n, s, t):
        """Kernel function we want to compute.

        Args:
            n: int
                length of the subsequence
            s: str
                first string
            t: str
                second string
        Returns:
            result: float
                substring kernel between s and t
        """
        if min(len(s), len(t)) < n:
            return 0
        else:
            part_sum = 0
            for j in range(1, len(t)):
                if t[j] == s[-1]:
                    part_sum += self.K1(n - 1, s[:-1], t[:j])
            result = self.K(n, s[:-1], t) + self.lambd ** 2 * part_sum
            return result

    @lru_cache(maxsize=cache_size)
    def K1(self, n, s, t):
        """K' in the original article (intermediate function)

        Args:
            n: int
                length of the subsequence
            s: str
                first string
            t: str
                second string
        Returns:
            result: float
                intermediate value
        """
        if n == 0:
            return 1
        elif min(len(s), len(t)) < n:
            return 0
        else:
            result = self.K2(n, s, t) + self.lambd * self.K1(n, s[:-1], t)
            return result

    @lru_cache(maxsize=cache_size)
    def K2(self, n, s, t):
        """K'' in the original article (intermediate function)

        Args:
            n: int
                length of the subsequence
            s: str
                first string
            t: str
                second string
        Returns:
            result: float
                intermediate value
        """
        if n == 0:
            return 1
        elif min(len(s), len(t)) < n:
            return 0
        else:
            if s[-1] == t[-1]:
                return self.lambd * (
                    self.K2(n, s, t[:-1]) + self.lambd * self.K1(n - 1, s[:-1], t[:-1])
                )
            else:
                return self.lambd * self.K2(n, s, t[:-1])

    @lru_cache(maxsize=cache_size)
    def gram_matrix_element(self, s, t, sdkvalue1, sdkvalue2):
        """Computes an element of the Gram matrix.

        Args:
            s: str
                first string
            t: str
                second string
            sdkvalue1: float
                normalizing parameter associated with s
            sdkvalue2: float
                normalizing parameter associated with t
        Returns:
            value for the element of the Gram matrix corresponding to (s, t)
        """
        if s == t:
            return 1
        else:
            try:
                return self.K(self.subseq_length, s, t) / (sdkvalue1 * sdkvalue2) ** 0.5
            except ZeroDivisionError:
                print(
                    "Maximal subsequence length is less or equal to the sequence minimal length."
                )
                print("Value sdkvalue1: ", sdkvalue1)
                print("Value sdkvalue2: ", sdkvalue2)

    def gram_element(self, gram_matrix, args):
        """Function to compute a single element of the Gram matrix

        Args:
            gram_matrix: multiprocessing.managers.ArrayProxy
                Gram matrix to fill
            args: (int, int, str, str, float)
                indexes of the Gram matrix to compute,
                the associated character sequences,
                and the normalizing values
        """
        i, j, s, t, sdkvalue1, sdkvalue2 = args
        gram_matrix[i, j] = self.gram_matrix_element(s, t, sdkvalue1, sdkvalue2)
        gram_matrix[j, i] = gram_matrix[i, j]

    def compute_gram(self, X):
        """Computes the substring kernel.

        Args:
            X: List[str]
                list of the character sequences to use
        Returns:
            gram_matrix: np.array
            Gram matrix associated to the given character sequences
        """
        len_X = len(X)
        sim_kernel_value = {}
        if self.parallel:
            m = MyManager()
            gram_matrix = m.np_zeros((len_X, len_X), dtype=float)
            pool = Pool(multiprocessing.cpu_count())
            par_func = partial(self.gram_element, gram_matrix)

            for i in tqdm(range(len_X), desc="Computing similar values"):
                sim_kernel_value[i] = self.K(self.subseq_length, X[i], X[i])
            for i in tqdm(range(len_X), desc="Computing different values"):
                run_list = [
                    (i, j, X[i], X[j], sim_kernel_value[i], sim_kernel_value[j])
                    for j in range(len_X)
                ]
                _ = pool.map(par_func, run_list)
            self.gram_matrix = np.asarray(gram_matrix)
        else:
            gram_matrix = np.zeros((len_X, len_X), dtype=float)

            for i in tqdm(range(len_X), desc="Computing similar values"):
                sim_kernel_value[i] = self.K(self.subseq_length, X[i], X[i])
            for i in tqdm(range(len_X), desc="Computing different values"):
                for j in range(i, len_X):
                    gram_matrix[i, j] = self.gram_matrix_element(
                        X[i], X[j], sim_kernel_value[i], sim_kernel_value[j]
                    )
                    gram_matrix[j, i] = gram_matrix[i, j]
            self.gram_matrix = gram_matrix

        return self.gram_matrix
