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


def boostrap_list_comprehension(data, boots):
    """
    Create bootstrap datasets that represent the distribution of the mean.
    Returns a cPython list cntaining the bootstrap datasets

    Keyword arguments:
    data -- numpy array of systems to boostrap
    boots -- number of bootstrap

    """
    return [bootstrap_mean(data) for i in range(boots)]


def bootstrap_mean(data):
    """
    Computes the mean of a bootstrap sample
    """
    return mean(bootstrap_dataset(data, resample(len(data))))


def mean(data):
    """
    Computes the mean of a list of numbers
    @data - list of integers/floats etc.

    """
    # return math.fsum(i for i in data)/len(data)
    # note: math.fsum offers better precision for floating point arthimetic
    # sum(data) is faster!
    return sum(data) / len(data)


def bootstrap_dataset(data, sample):
    """
    returns a list of resampled values

    Keyword Arguments:
    data -- the original scenario data
    sample -- a list of random numbers used to resample from @data

    """
    return [data[i] for i in sample]


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
def bootstrap_jit_parallel(data, boots=10000, q=0.95, log=False):
    """
    Create bootstrap datasets that represent the distribution of the mean.
    Returns quantile q for bootstrap datasets

    Keyword arguments:
    data -- numpy array of systems to boostrap
    boots -- number of bootstrap (default = 1000)
    q -- quantile
    lof -- if False - mean, if True - log of means
    """

    to_return = np.empty(boots)

    for b in prange(boots):

        total = 0.0

        for s in range(data.shape[0]):
            total += data[np.random.randint(0, data.shape[0])]
        if log:
            to_return[b] = np.log(total / data.shape[0]) if total != 0.0 else 0.0
        else:
            to_return[b] = total / data.shape[0]
    quantile = np.quantile(to_return, q)
    return quantile



