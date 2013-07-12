import numpy as np

"""
CHALLENGE!!!

Your task is to cythonize one of the two functions: length or downsample

length: calculates the length of a streamline
resample: reduces or increase equidinstantly the number of points on a 
streamline (interpolation)

Make it blazing fast!!
"""


def length(xyz, along=False):
    ''' Euclidean length of track line

    This will give length in mm if tracks are expressed in world coordinates.

    Parameters
    ------------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track
    along : bool, optional
       If True, return array giving cumulative length along track,
       otherwise (default) return scalar giving total length.

    Returns
    ---------
    L : scalar or array shape (N-1,)
       scalar in case of `along` == False, giving total length, array if
       `along` == True, giving cumulative lengths.

    Examples
    --------
    >>> xyz = np.array([[1,1,1],[2,3,4],[0,0,0]])
    >>> expected_lens = np.sqrt([1+2**2+3**2, 2**2+3**2+4**2])
    >>> length(xyz) == expected_lens.sum()
    True
    >>> len_along = length(xyz, along=True)
    >>> np.allclose(len_along, expected_lens.cumsum())
    True
    >>> length([])
    0
    >>> length([[1, 2, 3]])
    0
    >>> length([], along=True)
    array([0])
    '''
    xyz = np.asarray(xyz)
    if xyz.shape[0] < 2:
        if along:
            return np.array([0])
        return 0
    dists = np.sqrt((np.diff(xyz, axis=0) ** 2).sum(axis=1))
    if along:
        return np.cumsum(dists)
    return np.sum(dists)


def _extrap(xyz, cumlen, distance):
    ''' Helper function for extrapolate
    '''
    ind = np.where((cumlen - distance) > 0)[0][0]
    len0 = cumlen[ind - 1]
    len1 = cumlen[ind]
    Ds = distance - len0
    Lambda = Ds / (len1 - len0)
    return Lambda * xyz[ind] + (1 - Lambda) * xyz[ind - 1]


def resample(xyz, n_pols=3):
    ''' downsample or upsample for a specific number of points along the curve/track

    Uses the length of the curve. It works in a similar fashion to
    midpoint and arbitrarypoint but it also reduces the number of segments
    of a track.

    Parameters
    ----------
    xyz : array-like shape (N,3)
       array representing x,y,z of N points in a track
    n_pol : int
       integer representing number of points (poles) we need along the curve.

    Returns
    -------
    xyz2 : array shape (M,3)
       array representing x,y,z of M points that where extrapolated. M
       should be equal to n_pols

    Examples
    --------
    >>> import numpy as np
    >>> # a semi-circle
    >>> theta=np.pi*np.linspace(0,1,100)
    >>> x=np.cos(theta)
    >>> y=np.sin(theta)
    >>> z=0*x
    >>> xyz=np.vstack((x,y,z)).T
    >>> xyz2=resample(xyz,3)
    >>> # a cosine
    >>> x=np.pi*np.linspace(0,1,100)
    >>> y=np.cos(theta)
    >>> z=0*y
    >>> xyz=np.vstack((x,y,z)).T
    >>> xyz2=resample(xyz,3)
    >>> len(xyz2)
    3
    >>> xyz3=resample(xyz,10)
    >>> len(xyz3)
    10
    '''
    xyz = np.asarray(xyz)
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')
    if n_pts == 1:
        return xyz.copy().squeeze()
    cumlen = np.zeros(n_pts)
    cumlen[1:] = length(xyz, along=True)
    step = cumlen[-1] / (n_pols - 1)
    if cumlen[-1] < step:
        raise ValueError('Given number of points n_pols is incorrect. ')
    if n_pols <= 2:
        raise ValueError('Given number of points n_pols needs to be'
                         ' higher than 2. ')

    ar = np.arange(0, cumlen[-1], step)
    if np.abs(ar[-1] - cumlen[-1]) < np.finfo('f4').eps:
        ar = ar[:-1]

    xyz2 = [_extrap(xyz, cumlen, distance) for distance in ar]
    return np.vstack((np.array(xyz2), xyz[-1]))


# Create a streamline
theta = np.pi * np.linspace(0, 1, 100)
x = np.cos(theta)
y = np.sin(theta)
z = 0 * x
streamline = np.vstack((x, y, z)).T

# Resample it
streamline2 = resample(streamline, 60)
# Calculate the length
length(streamline)
