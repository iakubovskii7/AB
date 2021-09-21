import numpy as np
from numba import jit, prange


def bootstrap_numpy(data, boots):
    """
    Create bootstrap datasets that represent the distribution of the mean.
    Returns a numpy array containing the bootstrap datasets

    Keyword arguments:
    data -- numpy array of systems to boostrap
    boots -- number of bootstrap
    """

    to_return = np.empty(boots)

    for b in range(boots):

        total = 0

        for s in range(data.shape[0]):
            total += data[np.random.randint(0, data.shape[0])]

        to_return[b] = total / data.shape[0]

    return to_return


def bootstrap_numpy(data, boots):
    """
    Create bootstrap datasets that represent the distribution of the mean.
    Returns a numpy array containing the bootstrap datasets

    Keyword arguments:
    data -- numpy array of systems to boostrap
    boots -- number of bootstrap
    """

    to_return = np.empty(boots)

    for b in range(boots):

        total = 0

        for s in range(data.shape[0]):
            total += data[np.random.randint(0, data.shape[0])]

        to_return[b] = total / data.shape[0]

    return to_return


def bootstrap_better_numpy(data, boots):
    """
    Create bootstrap datasets that represent the distribution of the mean.
    Makes better use of built in numpy methods for more efficient sampling
    Returns a numpy array containing the bootstrap datasets

    Keyword arguments:
    data -- numpy array of systems to boostrap
    boots -- number of bootstrap
    """

    to_return = np.empty(boots)

    for b in range(boots):
        total = data[np.random.randint(0, data.shape[0], data.shape[0])].sum()

        to_return[b] = total / data.shape[0]

    return to_return


@jit(nopython=True)
def bootstrap_better_numpy_jit(data, boots):
    """
    Create bootstrap datasets that represent the distribution of the mean.
    Makes better use of built in numpy methods for more efficient sampling
    Returns a numpy array containing the bootstrap datasets

    Keyword arguments:
    data -- numpy array of systems to boostrap
    boots -- number of bootstrap
    """

    to_return = np.empty(boots)

    for b in range(boots):
        total = data[np.random.randint(0, data.shape[0], data.shape[0])].sum()

        to_return[b] = total / data.shape[0]

    return to_return

def resample(n):
    """
    returns a list of size n containing resampled values (integers) between
    0 and n - 1

    @n : number of random integers to generate

    """
    return [np.random.randint(0, n) for i in range(n)]


@jit(nopython=True)
def bootstrap_jit(data, boots):
    """
    Create bootstrap datasets that represent the distribution of the mean.
    Returns a numpy array containing the bootstrap datasets

    Keyword arguments:
    data -- numpy array of systems to boostrap
    boots -- number of bootstrap (default = 1000)
    """

    to_return = np.empty(boots)

    for b in range(boots):

        total = 0.0

        for s in range(data.shape[0]):
            total += data[np.random.randint(0, data.shape[0])]

        to_return[b] = total / data.shape[0]

    return to_return


@jit(nopython=True, parallel=True)
def bootstrap_jit_parallel(data, n_boots=10000):
    """
    Create bootstrap datasets that represent the distribution of the mean.
    Returns quantile q for bootstrap datasets

    Keyword arguments:
    data -- numpy array of systems to boostrap
    boots -- number of bootstrap (default = 1000)
    """

    to_return = np.empty(n_boots)

    for b in prange(n_boots):

        total = 0.0

        for s in range(data.shape[0]):
            total += data[np.random.randint(0, data.shape[0])]
        to_return[b] = total / data.shape[0]
    return to_return



