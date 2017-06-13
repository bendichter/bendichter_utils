# Little functions that make your life easier
from __future__ import print_function
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.tsatools import lagmat
import pickle
from itertools import islice, tee
import paramiko
from contextlib import contextmanager
from getpass import getpass
import re

def rm_nans(*args):
    """
    Removes indices from all arrays that is a nan in any of them. Prepares arrays for sklearn
    :param args: array1, array2, array3 ...
    :return:
    """
    AA = np.vstack([a if a.ndim == 1 else a.T for a in args])
    nan_ids = np.any(np.isnan(AA), 0)
    return [a[np.logical_not(nan_ids)] for a in args]


def isin_single_interval(tt, tbound, inclusive_left, inclusive_right):
    if inclusive_left:
        left_condition = (tt >= tbound[0])
    else:
        left_condition = (tt >  tbound[0])

    if inclusive_right:
        right_condition = (tt <= tbound[1])
    else:
        right_condition = (tt <  tbound[1])

    return left_condition & right_condition


def isin(tt, tbounds, inclusive_left=False, inclusive_right=False):
    """
    util: Is time inside time window(s)?

    :param tt:      n,    np.array   time counter
    :param tbounds: k, 2  np.array   time windows

    :return:        n, bool          logical indicating if time is in any of the windows
    """
    #check if tbounds in np.array and if not fix it
    tbounds = np.array(tbounds)
    tt = np.array(tt)

    tf = np.zeros(tt.shape, dtype='bool')

    if len(tbounds.shape) is 1:
        tf = isin_single_interval(tt, tbounds, inclusive_left, inclusive_right)
    else:
        for tbound in tbounds:
            tf = tf | isin_single_interval(tt, tbound, inclusive_left, inclusive_right)
    return tf.astype(bool)


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


def is_overlap(time_window,times_window_array):
    """
    Does time_window overlap with the time windows in times_window_array. Used for bad time segments
    :param times: np.array(1,2)
    :param times_array: np.array(x,2)
    :return: TF

    """
    def overlap(tw1, tw2):
        return not ((tw1[1] < tw2[0]) | (tw1[0] > tw2[1]))

    return [overlap(time_window,this_time_window) for this_time_window in times_window_array]


def smooth(x, window_len=11, window='hanning', buffer=True):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[x[(window_len-1)//2:0:-1], x, x[-1:(-window_len-1)//2:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y=np.convolve(w / w.sum(), s, mode='valid')

    return y


def smooth_demo():

    t = np.linspace(-4,4,100)
    x = np.sin(t)
    xn = x + np.random.randn(len(t)) * 0.1
    y = smooth(x)

    ws = 31

    plt.subplot(211)
    plt.plot(np.ones(ws))

    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    for w in windows[1:]:
        eval('plt.plot(np.' + w + '(ws) )')

    plt.axis([0,30,0,1.1])

    plt.legend(windows)
    plt.title("The smoothing windows")
    plt.subplot(212)
    plt.plot(x)
    plt.plot(xn)
    for w in windows:
        plt.plot(smooth(xn,10,w))
    l=['original signal', 'signal with noise']
    l.extend(windows)

    plt.legend(l)
    plt.title("Smoothing a noisy signal")


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def fill_nan(A, kind='cubic'):
    '''
    interpolate to fill nan values
    '''
    import scipy.interpolate

    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = scipy.interpolate.interp1d(inds[good], A[good], bounds_error=False, kind=kind)
    B = np.where(np.isfinite(A), A, f(inds))
    return B


def threshcross(signal, thresh=0, direction='up'):
    """
    Finds the indices where a signal crosses a threshold.
    :param signal: np.array(Nx1)
    :param thresh: double, default=0
    :param direction: str, direction of crosses to detect. {'up'}, 'down', 'both'
    :return:
    """

    over = (signal >= thresh)
    cross = np.diff(over.astype('int'))

    if direction == 'up':
        return np.where(cross > 0)[0] + 1
    elif direction == 'down':
        return np.where(cross < 0)[0] + 1
    elif direction == 'both':
        return np.where(cross != 0)[0] + 1


def lagmatrix(x, y, lead, lag):
    """
    output lagmatrix x which has lead and lag compared to y

    x is TxK, y is TxN
    """

    nlags = lead + lag
    x_lagged = lagmat(x, nlags, trim='both')
    x_lagged = np.concatenate((x[nlags:, :], x_lagged), axis=1)
    y_lagged = y[lead:-lag, :]

    return x_lagged, y_lagged


def fill_nans_prev(data):
    prev = np.nan
    new_data = []
    for dat in data:
        if np.isnan(dat):
            new_data.append(prev)
        else:
            new_data.append(dat)
        prev = new_data[-1]
    return np.array(new_data)


def change_pickle_protocol(filepath, protocol=2):
    """
    Change the protocol that is used for a specific file. Is useful for opening a pandas DataFrame which was created
    in python 3 in python 2
    :param filepath: str
    :param protocol: int, default=2
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def isqrt(x):
    if x < 0:
        raise ValueError('square root not defined for negative numbers')
    n = int(x)
    if n == 0:
        return 0
    a, b = divmod(n.bit_length(), 2)
    x = 2 ** (a + b)
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


def do_lag(x,y,lag):
    if lag > 0:
        x = x[:-lag]
        y = y[lag:]
    elif lag < 0:
        x = x[-lag:]
        y = y[:lag]
    return x, y


def sliding_window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


@contextmanager
def create_ssh(host='dura.cin.ucsf.edu', port=7777, username='bdichter', **kwargs):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        password = getpass()
        ssh.connect(host, port=port, username=username, password=password, **kwargs)
        yield ssh
    finally:
        ssh.close()


def findall(string, sub):
    return [m.start() for m in re.finditer(sub, string)]


def shuffled(x, axis=None):
    """
    Shuffles `x` along dimension.

    Parameters
    ----------
    x: array_like
    axis: None or int or tuple of ints, optional
        The axis or axes along which to shuffle the data. Default=0

    Returns
    -------
    y: ndarray
        Shuffled data
    """
    if axis is None:
        axis = 0

    axis = np.array(axis)  # Needed if axis is int to make iterable

    y = x[:]  # copy data

    for aa in axis:
        y = np.swapaxes(y, 0, aa)
        np.random.shuffle(y)
        y = np.swapaxes(y, aa, 0)

    return y


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def col_corr(X, Y, rm_nans=True):
    """
    Calculate Pearson correlations between respective columns of two matrices.

    equivalent to:
    np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(X.T, Y.T)])

    np.array(corr(X[:, 0], Y[:, 0]), corr(X[:, 1], Y[:, 1]), ...)

    Parameters
    ================
    X: numpy.array(N x K)
    Y:
    rm_nans : bool, optional
        default=True

    Output
    ================
    numpy.array
    """

    if len(X) != len(Y):
        raise ValueError('len of X must equal len of Y')

    if (X.ndim > 1) and (Y.ndim > 1) and (X.shape != Y.shape):
        raise ValueError('if X and Y are matrices, they must be the same shape')

    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    if rm_nans:
        nan_inds = np.any(np.isnan(np.hstack((X, Y))), axis=1)
        X = X[np.logical_not(nan_inds), :]
        Y = Y[np.logical_not(nan_inds), :]

    num = np.sum((X - np.mean(X, axis=0)) * (Y - np.mean(Y, axis=0)), axis=0)
    den = np.sqrt(np.sum((X - np.mean(X, axis=0)) ** 2, axis=0) * np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0))

    return num / den


def groupby(dat, ids):
    uids = np.unique(ids)
    return [(x, dat[ids == x]) for x in uids]


def stratified_bootstrap(labels):
    """
    Performs a single stratified sampling with replacement. Each class
    is sampled individually to preserve the number of samples per class

    Parameters:
    - labels            array_like    List of class labels

    Returns:
    - out               ndarray    List of indices
    """
    labels = np.array(labels)  # Needed for list of strings.

    all_inds_int = []
    for label in np.unique(labels):
        inds_ints = np.where(labels == label)[0]
        all_inds_int.append(np.random.choice(inds_ints, size=(len(inds_ints),)))

    all_inds_int = np.array(all_inds_int)

    return all_inds_int