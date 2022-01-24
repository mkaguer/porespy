import numpy as np
from edt import edt
import numba
from porespy.filters import trim_disconnected_blobs, find_trapped_regions
from porespy.filters import find_disconnected_voxels
from porespy.filters import pc_to_satn, satn_to_seq, seq_to_satn
from porespy import settings
from porespy.tools import get_tqdm
from porespy.tools import Results
from porespy.metrics import pc_curve
tqdm = get_tqdm()


__all__ = [
    'drainage',
]


@numba.jit(nopython=True, parallel=False)
def insert_disks_at_points(im, coords, radii, v, smooth=True):  # pragma: no cover
    r"""
    Insert spheres (or disks) of specified radii into an ND-image at given locations.

    This function uses numba to accelerate the process, and does not overwrite
    any existing values (i.e. only writes to locations containing zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    coords : ND-array
        The center point of each sphere/disk in an array of shape
        ``ndim by npts``
    radii : array_like
        The radii of the spheres/disks to add.
    v : scalar
        The value to insert
    smooth : boolean
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.

    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        for i in range(npts):
            r = radii[i]
            s = _make_disk(r, smooth)
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            if (s[a, b] == 1) and (im[x, y] == 0):
                                im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        for i in range(npts):
            r = radii[i]
            s = _make_ball(r, smooth)
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(pt[2]-r, pt[2]+r+1)):
                                if (z >= 0) and (z < zlim):
                                    if (s[a, b, c] == 1) and (im[x, y, z] == 0):
                                        im[x, y, z] = v
    return im


@numba.jit(nopython=True, parallel=False)
def _make_disk(r, smooth=True):  # pragma: no cover
    r"""
    Generate a strel suitable for use in numba nojit function.

    Numba won't allow calls to skimage strel generators so this function
    makes one, also using njit.
    """
    s = np.zeros((2*r+1, 2*r+1), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(2*r+1):
        for j in range(2*r+1):
            if ((i - r)**2 + (j - r)**2)**0.5 <= thresh:
                s[i, j] = 1
    return s


@numba.jit(nopython=True, parallel=False)
def _make_ball(r, smooth=True):  # pragma: no cover
    r"""
    Generate a strel suitable for use in numba nojit function.

    Numba won't allow calls to skimage strel generators so this function
    makes one, also using njit.
    """
    s = np.zeros((2*r+1, 2*r+1, 2*r+1), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(2*r+1):
        for j in range(2*r+1):
            for k in range(2*r+1):
                if ((i - r)**2 + (j - r)**2 + (k - r)**2)**0.5 <= thresh:
                    s[i, j, k] = 1
    return s


def drainage(im, voxel_size, pc=None, inlets=None, outlets=None, residual=None,
             bins=25, delta_rho=1000, g=9.81, sigma=0.072, theta=180):
    r"""
    Simulate drainage using image-based sphere insertion, optionally including
    gravity

    Parameters
    ----------
    im : ndarray
        The image of the porous media with ``True`` values indicating the
        void space.
    voxel_size : float
        The resolution of the image in meters per voxel edge.
    pc : ndarray, optional
        An array containing precomputed capillary pressure values in each
        voxel. If not provided then the Washburn equation is used with the
        provided values of ``sigma`` and ``theta``. If the image is 2D only
        1 principle radii of curvature is included.
    inlets : ndarray (default = x0)
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. See Notes. If not specified it is
        assumed that the invading phase enters from the bottom (x=0).
    outlets : ndarray, optional
        Similar to ``inlets`` except defining the outlets. This image is used
        to assess trapping.  \If not provided then trapping is ignored,
        otherwise a mask indicating which voxels were trapped is included
        amoung the returned data.
    residual : ndarray, optional
        A boolean array indicating the locations of any residual defending
        phase. This is added to the intermediate image prior to trimming
        disconnected clusters, so will create connections to some clusters
        that would otherwise be removed. The residual phase is indicated
        in the final image by ``-np.inf`` values, since there are invaded at
        all applied capillary pressures.
    bins : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given
        then bins will be created between the lowest and highest pressures
        in the ``pc``.  If a list is given, each value in the list is used
        directly in order.
    delta_rho : float (default = 997)
        The density difference between the invading and defending phases.
        Note that if air is displacing water this value should be -997 (1-998).
    g : float (default = 9.81)
        The gravitational constant prevailing for the simulation. The default
        is 9.81. If the domain is on an angle, such as a tilted micromodel,
        this value should be scaled appropriately by the user
        (i.e. g = 9.81 sin(alpha) where alpha is the angle relative to the
        horizonal).  Setting this value to zeor removes any gravity effects.
    sigma : float (default = 0.072)
        The surface tension of the fluid pair. If ``pc`` is provided this is
        ignored.
    theta : float (defaut = 180)
        The contact angle of the sytem in degrees.  If ``pc`` is provded this
        is ignored.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

    ========== ================================================================
    Attribute  Description
    ========== ================================================================
    im_pc      A numpy array with each voxel value indicating the
               capillary pressure at which it was invaded
    im_satn    A numpy array with each voxel value indicating the global
               saturation value at the point it was invaded
    pc         1D array of capillary pressure values that were applied
    swnp       1D array of non-wetting phase saturations for each applied
               value of capillary pressure (``pc``).
    ========== ================================================================

    Notes
    -----
    - The direction of gravity is always towards the x=0 axis
    - This algorithm has only been tested for gravity stabilized
      configurations, meaning the more dense fluid is on the bottom.
      Be sure that ``inlets`` are specified accordingly.

    """
    im = np.array(im, dtype=bool)
    dt = edt(im)
    if pc is None:
        pc = -(im.ndim-1)*sigma*np.cos(np.deg2rad(theta))/(dt*voxel_size)
    pc[~im] = 0  # Remove any infs or nans from pc computation

    # Generate image for correcting entry pressure by gravitational effects
    h = np.ones_like(im, dtype=bool)
    h[0, ...] = False
    h = (edt(h) + 1)*voxel_size   # This could be done quicker using clever logic
    rgh = delta_rho*g*h
    fn = pc + rgh

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True

    if outlets is not None:
        outlets = outlets * im

    if isinstance(bins, int):  # Use values fn for invasion steps
        vmax = pc[pc < np.inf].max()
        vmin = pc[im][pc[im] > -np.inf].min()
        Ps = np.logspace(np.log10(vmin), np.log10(vmax), bins)
    else:
        Ps = bins

    # Initialize empty arrays to accumulate results of each loop
    inv = np.zeros_like(im, dtype=float)
    seeds = np.zeros_like(im, dtype=bool)
    # Deal with any void space trapped behind residual blobs
    mask = None
    if (residual is not None) and (outlets is not None):
        mask = im * (~residual)
        mask = trim_disconnected_blobs(mask, inlets=inlets)
    for p in tqdm(Ps, **settings.tqdm):
        # Find all locations in image invadable at current pressure
        temp = (fn <= p)*im
        # Add residual so that fluid is more easily reconnected
        if residual is not None:
            temp = temp + residual
        # Trim locations not connected to the inlets
        new_seeds = trim_disconnected_blobs(temp, inlets=inlets)
        # Trim locations not connected to the outlet
        if mask is not None:
            new_seeds = new_seeds * mask
        # Isolate only newly found locations to speed up inserting
        temp = new_seeds*(~seeds)
        # Find i,j,k coordinates of new locations
        coords = np.where(temp)
        # Add new locations to list of invaded locations
        seeds += new_seeds
        # Extract the local size of sphere to insert at each new location
        radii = dt[coords].astype(int)
        # Insert spheres are new locations of given radii
        inv = insert_disks_at_points(inv, np.vstack(coords), radii, p, smooth=True)

    # Set uninvaded voxels to inf
    inv[(inv == 0)*im] = np.inf

    # Add residual if given
    if residual is not None:
        inv[residual] = -np.inf

    # Initialize results object
    results = Results()
    trapped = None
    satn = pc_to_satn(pc=inv, im=im)

    if outlets is not None:
        seq = satn_to_seq(satn=satn, im=im)
        trapped = find_trapped_regions(seq=seq, outlets=outlets)
        trapped[seq == -1] = True
        inv[trapped] = np.inf
        if residual is not None:  # Re-add residual to inv
            inv[residual] = -np.inf
        satn = pc_to_satn(pc=inv, im=im)

    results.im_satn = satn
    results.im_pc = inv
    results.im_trapped = trapped

    _pccurve = pc_curve(im=im, pressures=inv)
    results.pc = _pccurve.pc
    results.snwp = _pccurve.snwp

    return results


if __name__ == "__main__":
    import numpy as np
    import porespy as ps
    import matplotlib.pyplot as plt

    # %% Run this cell to regenerate the variables in drainage
    np.random.seed(6)
    bg = 'white'
    im = ps.generators.blobs(shape=[500, 500], porosity=0.7, blobiness=1.5)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    outlets = np.zeros_like(im)
    outlets[-1, :] = True
    pc = None
    lt = ps.filters.local_thickness(im)
    residual = lt > 25
    bins = 25
    voxel_size = 1e-4
    sigma = 0.072
    theta = 180
    delta_rho = 1000
    g = 0


# %%
    if 0:
        drn5 = ps.simulations.drainage(im=im, voxel_size=voxel_size,
                                       inlets=inlets,
                                       residual=residual,
                                       g=g)
        plt.imshow(drn5.im_satn/im, origin='lower')
        # fig, ax = invasion_steps(drn5.im_satn, im)


    # %% Run 4 different drainage simulations
    if 1:
        drn = ps.simulations.drainage(im=im, voxel_size=voxel_size,
                                      inlets=inlets,
                                      g=g)
        drn2 = ps.simulations.drainage(im=im, voxel_size=voxel_size,
                                       inlets=inlets, outlets=outlets,
                                       g=g)
        drn3 = ps.simulations.drainage(im=im, voxel_size=voxel_size,
                                       inlets=inlets,
                                       residual=residual,
                                       g=g)
        drn4 = ps.simulations.drainage(im=im, voxel_size=voxel_size,
                                       inlets=inlets, outlets=outlets,
                                       residual=residual,
                                       g=g)


    # %% Visualize the invasion configurations for each scenario
    if 1:
        fig, ax = plt.subplots(2, 2, facecolor=bg)
        ax[0][0].imshow(drn.im_satn/im, origin='lower')
        ax[0][0].set_title("No trapping, no residual")
        ax[0][1].imshow(drn2.im_satn/im, origin='lower')
        ax[0][1].set_title("With trapping, no residual")
        ax[1][0].imshow(drn3.im_satn/im, origin='lower')
        ax[1][0].set_title("No trapping, with residual")
        ax[1][1].imshow(drn4.im_satn/im, origin='lower')
        ax[1][1].set_title("With trapping, with residual")


    # %% Plot the capillary pressure curves for each scenario
    if 1:
        plt.figure(facecolor=bg)
        ax = plt.axes()
        ax.set_facecolor(bg)
        plt.step(np.log10(drn.pc), drn.snwp, 'b-o', where='post',
                 label="No trapping, no residual")
        plt.step(np.log10(drn2.pc), drn2.snwp, 'r--o', where='post',
                 label="With trapping, no residual")
        plt.step(np.log10(drn3.pc), drn3.snwp, 'g--o', where='post',
                 label="No trapping, with residual")
        plt.step(np.log10(drn4.pc), drn4.snwp, 'm--o', where='post',
                 label="With trapping, with residual")
        plt.legend()

