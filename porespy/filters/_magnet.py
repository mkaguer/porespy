import logging
import numpy as np
import scipy as sp
import scipy.ndimage as spim
import skimage as ski
from edt import edt
import dask.array as da
from skimage.morphology import square, cube
from porespy.tools import _insert_disks_at_points_m, make_contiguous, unpad
from porespy.tools import extend_slice, insert_sphere, Results
from porespy import settings
from porespy.filters import trim_floating_solid
from porespy.filters._snows import _estimate_overlap

logger = logging.getLogger(__name__)


def magnet(im,
           sk=None,
           padding=20,
           parallel=False,
           numba=False,
           voxel_size=1,
           l_max=7,
           boundary_width=3,
           **kwargs):
    r"""
    Perform a Medial Axis Guided Network ExtracTion (MAGNET) on an image of
    porous media. This is a modernized python implementation of an efficient
    network extraction method. First the skeleton of the provided image is
    determined. Padding is added to the image before getting the skeleton to
    help identify boundary pores. The skeleton can be computed in serial or
    parallel modes. Next all the junction points of the skeleton are determined
    by using convolution to find voxels with extra neighbors, as well as
    terminal points on the ends of branches. Pores are then inserted at these
    points. The size of the pores inserted is based on the distance transform
    value at it's junction. This approach results in many long throats so more
    pores are added using a maximum filter along long throats to find openings.
    Throats are found from how the skeleton connects the junction points. To
    ensure an efficient network extraction method, only the most fundamential
    pare and throat properties are calculated.

    Parameters
    ------------
    im : ndarray
        An image of the porous material of interest. It is not necessary to
        trim floating solids beforehand as this is done automatically.
    sk : ndarray
        Optionally provide your own skeleton of the image. If sk is `None` we
        compute the skeleton using `skimage.morphology.skeleton_3d`. This is
        recommended to ensure that the skeleton does not have any shells and is
        well suited for boundary pores. If however your own custom skeleton is
        provided we check the skeleton for shells and throw a warning if shells
        are detected in the skeleton.
    padding : integer
        The amount of padding to add to the image before determining the
        skeleton. This helps determine boundary pores.
    parallel : boolean
        If `False` the skeleton is calculated in serial. This is the default
        mode. However, if `True`, the skeleton is calculated in parallel using
        dask. The default mode is `False`
    numba : boolean
        If `False` pure python is used in `insert_pore_bodies`. However, if set
        to `True` numba is used to speed up the insertion of pores. We
        recommend setting to `True` for large networks. The default mode is
        `False`.
    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be voxel_size-cubed
    l_max : scalar (default = 7)
        The size of the maximum filter used in finding pores along long throats
    boundary_width : integer (default = 0)
        The number of voxels inward from the edge of the image which
        constitutes the boundary. Pores centred within this boundary are
        labelled as boundary pores. 

    Returns
    -------
    net : dict
        A dictionary containing the most important pore and throat size data
        and topological data. These are pore radius, throat radius, pore
        coordinates, and throat connections. The dictionary names use the
        OpenPNM convention (i.e. 'pore.coords', 'throat.conns', 'pore.radius',
        'throat.radius'). Labels for boundary pores and overlapping thraots
        are also returned.
    """
    # get the skeleton
    if sk is None:
        sk = skeleton(im, padding, parallel, **kwargs)
    else:
        _check_skeleton_health(sk)  # FIXME: This doesn't work on 2D
    # find junction points
    pt = analyze_skeleton(sk)
    # distance transform
    if im.ndim == 3:
        im = trim_floating_solid(im, conn=6, surface=True)  # ensure no floating solids
    b = square(3) if im.ndim == 2 else cube(3)
    dt = spim.maximum_filter(edt(im), footprint=b)
    # insert pores at junction points
    fbd = insert_pore_bodies(sk, dt, pt, l_max, numba)
    # convert spheres to network dictionary
    net = spheres_to_network(sk, dt, fbd, pt, voxel_size, boundary_width)
    return net


def analyze_skeleton(sk):
    r"""
    Finds all the junction in a skeleton. It uses convolution to find voxels
    with extra neighbors, as well as terminal points on the ends of branches.

    Parameters
    ------------
    sk : ndarray
        The skeleton of an image (boolean).

    Returns
    -------
    pt : Results object
        A custom object with the following data added as named attributes:
        'juncs'
        An array of ones where all the junction points were found
        'endpts'
        An array of ones where all the endpoints were found
        'juncs_r'
        A boolean array of all junctions after calling reduce_peaks()
    """
    # kernel for convolution
    if sk.ndim == 2:
        a = square(3)
    else:
        a = cube(3)
    # compute convolution directly or via fft, whichever is fastest
    conv = sp.signal.convolve(sk*1.0, a, mode='same', method='auto')
    conv = np.rint(conv).astype(int)  # in case of fft, accuracy is lost
    # find junction points of skeleton
    juncs = (conv >= 4) * sk
    # find endpoints of skeleton
    end_pts = (conv == 2) * sk
    # reduce cluster of junctions to single pixel at centre
    juncs_r = reduce_peaks(juncs)
    # results object
    pt = Results()
    pt.juncs = juncs
    pt.endpts = end_pts
    pt.juncs_r = juncs_r
    return pt


def insert_pore_bodies(sk, dt, pt, l_max=7, numba=False):
    r"""
    Insert spheres at each junction point of the skeleton corresponding to
    the local size. A search for local maximums is performed along throats
    between inserted spheres. Additional spheres are inserted where any local
    maximums are found.

    Parameters
    ------------
    sk : ndarray
        The skeleton of an image (boolean).
    dt : ndarray
        The distance transform of the phase of interest.
    pt : Results object
        A custom object returned from find_junctions()
    l_max: int
        The length of the cubical structuring element to use in the maximum
        filter for inserting pores along long throats
    numba : boolean
        If `True` numba is used to speed up python for loops that are used for
        inserting pore bodies. We recommend setting to `True` for large
        images. The default mode is `False`.

    Returns
    -------
    fbd : Results object
        A custom object with the following data added as named attributes:
        'Ps'
        Inserted spheres corresponding to the local size
        'Ps2'
        Inserted spheres of uniform 'small' size (4 pixels)
        'mx'
        Maximum points along long throats where additional pores are inserted
        'p_coords'
        The coordinates where each sphere is added
        'p_radius'
        The radius of each sphere added
    """
    pts = pt.juncs + pt.endpts
    c = np.vstack(np.where(pts)).astype('float64').T
    Ps = np.zeros_like(pts, dtype=int)
    # Find number of dimensions
    ND = pts.ndim
    # insert spheres at junctions and endpoints
    d = np.insert(c, ND, dt[np.where(pts)].T, axis=1)
    d = np.flip(d[dt[np.where(pts)].T.argsort()], axis=0)
    d = np.delete(d, np.where(d[:, ND] == 0), axis=0)  # FIXME: temporary fix
    d = np.round(d).astype('int64')
    # Speed up for loop using numba
    if numba:
        n = len(d) + 1
        v = np.arange(1, n)
        Ps, p_coords1 = _insert_disks_at_points_m(im=Ps,
                                                  coords=d[:, 0:ND].T,
                                                  radii=d[:, ND],
                                                  v=v,
                                                  overwrite=True)
    # Pure python
    if not numba:
        p_coords = []
        for n, row in enumerate(d):
            coord = row[0:ND]
            if Ps[tuple(coord)] == 0:
                p_coords.append(coord)
                v = n + 1
                insert_sphere(im=Ps, c=coord, r=row[ND], v=v, overwrite=True)
    # Find maximums on long throats
    temp = Ps * np.inf
    mask = np.isnan(temp)
    temp[mask] = 0
    dt2 = spim.gaussian_filter(dt, sigma=0.4)
    temp = temp + dt2 * sk
    b = square(l_max) if ND == 2 else cube(l_max)
    mx = (spim.maximum_filter(temp, footprint=b) == dt2) * sk
    # insert spheres along long throats
    c = np.vstack(np.where(mx)).astype('float64').T
    # insert spheres at local maximums
    d = np.insert(c, ND, dt[np.where(mx)].T, axis=1)
    d = np.flip(d[d[:, ND].argsort()], axis=0)
    d = np.round(d).astype('int64')
    # Speed up for loop using numba
    if numba:
        v = np.arange(n, n + len(d))
        Ps, p_coords2 = _insert_disks_at_points_m(im=Ps,
                                                  coords=d[:, 0:ND].T,
                                                  radii=d[:, ND],
                                                  v=v,
                                                  overwrite=False)
        p_coords1.extend(p_coords2)
        p_coords = np.array(p_coords1)
    # Pure python
    if not numba:
        ss = n + 1
        for n, row in enumerate(d):
            coord = row[0:ND]
            if Ps[tuple(coord)] == 0:
                p_coords.append(coord)
                v = ss+n+1
                insert_sphere(im=Ps, c=coord, r=row[ND], v=v, overwrite=False)
        p_coords = np.array(p_coords)
    # retrieve radius
    p_radius = np.array([dt[tuple(co)] for co in p_coords])
    # make pore numbers sequential
    Ps = make_contiguous(Ps)
    # second image for finding throat connections
    Ps2 = ((pts + mx) > 0) * Ps
    # results object
    fbd = Results()
    fbd.Ps = Ps
    fbd.Ps2 = Ps2
    fbd.mx = mx
    fbd.pts = pts
    fbd.p_coords = p_coords
    fbd.p_radius = p_radius
    return fbd


def spheres_to_network(sk, dt, fbd, pt, voxel_size=1, boundary_width=0):
    r"""
    Assemable a dictionary object containing essential topological and
    geometrical information. The skeleton and an image with spheres already
    inserted is used to determine the essential throat and pore properties:
    throat.conns, throat.radius, pore.coords, and pore.radius. Labels are also
    created for overlapping throats and boundary pores.

    Parameters
    ------------
    sk : ndarray
        The skeleton of an image (boolean).
    dt : ndarray
        The distance transform of an image
    fbd: Results object
        A custom object returned from insert_pore_bodies()
    pt : Results object
        A custom object returned from find_junctions()
    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.
    boundary_width : integer (default = 0)
        The number of voxels inward from the edge of the image which
        constitutes the boundary. Pores centred within this boundary are
        labelled as boundary pores. 

    Returns
    -------
    net : dict
        A dictionary containing the most important pore and throat size data
        and topological data. These are pore radius, throat radius, pore
        coordinates, and throat connections. The dictionary names use the
        OpenPNM convention (i.e. 'pore.coords', 'throat.conns', 'pore.radius',
        'throat.radius'). Labels for boundary pores and overlapping throats
        are also returned.
    """
    # no. of dimensions and shape
    ND = fbd.Ps.ndim
    shape = fbd.Ps.shape
    # identify throat segments
    mx = fbd.mx
    pts = fbd.pts
    throats = (~((pts + mx) > 0)) * sk
    s = spim.generate_binary_structure(ND, ND)
    throats, num_throats = spim.label(throats, structure=s)
    slicess = spim.find_objects(throats)  # Nt by 2
    # initialize throat conns and radius
    t_conns = np.zeros((len(slicess), 2), dtype=int)
    t_radius = np.zeros((len(slicess)), dtype=float)
    # loop through throats to get t_conns and t_radius
    for i in range(num_throats):
        throat_l = i
        ss = extend_slice(slicess[throat_l], shape)
        # get slices
        sub_im_p = fbd.Ps2[ss]
        sub_im_l = throats[ss]
        sub_sk = sk[ss]
        sub_dt = dt[ss]
        throat_im = sub_im_l == i+1
        # dilate throat_im to capture connecting pore indices
        structure = spim.generate_binary_structure(ND, ND)
        im_w_throats_l = spim.binary_dilation(input=throat_im,
                                              structure=structure)
        im_w_throats_l = im_w_throats_l * sub_sk
        im_w_throats_l = im_w_throats_l * sub_im_p
        # throat conns
        Pn_l = np.unique(im_w_throats_l)[1:] - 1
        if np.any(Pn_l):
            Pn_l = Pn_l[0:2]
            t_conns[throat_l, :] = Pn_l
        # throat radius
        throat_dt = throat_im * sub_dt
        t_radius[throat_l] = np.min(throat_dt[throat_dt != 0])
    # remove duplicates (if any) in t_conns
    remove = np.where(t_conns[:, 0] == t_conns[:, 1])
    t_conns = np.delete(t_conns, remove, axis=0)
    t_radius = np.delete(t_radius, remove, axis=0)
    # find overlapping pores
    Nt = len(t_conns)
    P1 = t_conns[:, 0]
    P2 = t_conns[:, 1]
    R1 = fbd.p_radius[P1]
    R2 = fbd.p_radius[P2]
    pt1 = fbd.p_coords[P1]
    pt2 = fbd.p_coords[P2]
    dist = np.linalg.norm(pt1 - pt2, axis=1)
    t_overlapping = dist <= R1 + R2  # FIXME: does not consider lens
    # set throat radius of overlapping pores
    Rs = np.hstack((R1.reshape((Nt, 1)), R2.reshape((Nt, 1))))
    Rmin = np.min(Rs, axis=1)
    t_radius[t_overlapping] = 0.95 * Rmin[t_overlapping]
    # ensure throat radius is smaller than pore radii
    mask = Rmin <= t_radius
    t_radius[mask] = 0.95 * Rmin[mask]
    # sort AND remove smaller duplicate throat
    d = np.insert(t_conns.astype('float'), 2, t_radius, axis=1)
    d = np.insert(d, 3, t_overlapping.astype('float'), axis=1)
    d = np.flip(d[t_radius.argsort()], axis=0)
    t_conns = d[:, 0:2].astype('int')
    t_conns, indices = np.unique(t_conns, axis=0, return_index=True)
    t_radius = d[:, 2][indices]
    t_overlapping = d[:, 3][indices].astype('bool')
    # pore coords
    p_coords = fbd.p_coords.astype('float')
    if ND == 2:  # If 2D, add 0's in 3rd dimension
        Np = np.amax(fbd.Ps)
        p_coords = np.vstack((p_coords.T, np.zeros((Np, )))).T
    # create network dictionary
    net = {}
    net['throat.conns'] = t_conns
    net['pore.coords'] = p_coords * voxel_size
    net['throat.radius'] = t_radius * voxel_size
    net['pore.radius'] = fbd.p_radius * voxel_size
    net['throat.overlapping'] = t_overlapping
    bw = boundary_width * voxel_size
    net['pore.xmin'] = p_coords[:, 0] <= 0 + bw
    net['pore.xmax'] = p_coords[:, 0] >= shape[0] - 1 - bw
    net['pore.ymin'] = p_coords[:, 1] <= 0 + bw
    net['pore.ymax'] = p_coords[:, 1] >= shape[1] - 1 - bw
    if ND == 3:
        net['pore.zmin'] = p_coords[:, 2] <= 0 + bw
        net['pore.zmax'] = p_coords[:, 2] >= shape[2] - 1 - bw
    return net


def reduce_peaks(peaks):
    r"""
    Any peaks that are broad or elongated are replaced with a single voxel
    that is located at the center of mass of the original voxels.

    Parameters
    ----------
    peaks : ndarray
        An image containing ``True`` values indicating peaks in the
        distance transform

    Returns
    -------
    image : ndarray
        An array with the same number of isolated peaks as the original
        image, but fewer total ``True`` voxels.

    Notes
    -----
    The center of mass of a group of voxels is used as the new single
    voxel, so if the group has an odd shape (like a horse shoe), the new
    voxel may *not* lie on top of the original set.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/reduce_peaks.html>`_
    to view online example.

    """
    if peaks.ndim == 2:
        strel = square
    else:
        strel = cube
    markers, N = spim.label(input=peaks, structure=strel(3))
    inds = spim.measurements.center_of_mass(
        input=peaks, labels=markers, index=np.arange(1, N + 1)
    )
    inds = np.floor(inds).astype(int)
    # Centroid may not be on old pixel, so create a new peaks image
    peaks_new = np.zeros_like(peaks, dtype=bool)
    peaks_new[tuple(inds.T)] = True
    return peaks_new


def skeleton_parallel(im, divs, overlap=None, cores=None):
    r"""
    Performs `skimage.morphology.skeleton_3d` in parallel using dask

    Parameters
    ----------
    im : ndarray
        A binary image of porous media with 'True' values indicating
        phase of interest.
    divs : ndarray
        The number of divisions in each dimension used for chunking the image
        (e.g. [2, 2, 4])
    overlap : float (optional)
        The amount of overlap to apply between chunks.  If not provided it
        will be estiamted using ``porespy.tools.estimate_overlap`` with
        ``mode='dt'``.
    cores : int or None
        Number of cores that will be used to parallel process all domains.
        If ``None`` then all cores will be used but user can specify any
        integer values to control the memory usage.  Setting value to 1
        will effectively process the chunks in serial to minimize memory
        usage.

    Returns
    -------
    sk : ndarray
        Skeleton of image

    """
    if overlap is None:
        overlap = _estimate_overlap(im, mode='dt') * 2
    if cores is None:
        cores = settings.ncores
    depth = {}
    for i in range(im.ndim):
        depth[i] = np.round(overlap).astype(int)
    chunk_shape = (np.array(im.shape) / np.array(divs)).astype(int)
    skel = da.from_array(im, chunks=chunk_shape)
    skel = da.overlap.overlap(skel, depth=depth, boundary='none')
    skel = skel.map_blocks(ski.morphology.skeletonize_3d)
    skel = da.overlap.trim_internal(skel, depth, boundary='none')
    skel = skel.compute(num_workers=cores).astype(bool)
    return skel


def skeleton(im, padding=20, parallel=False, **kwargs):
    r"""
    This helper function adds padding to an image before the skeleton is
    determined. This is useful for determining boundary pores in conjunction
    with MAGNET. This method uses `skimage.morphology.skeleton_3d` in either
    serial or parallel. This function also ensures no shells in the final
    skeleton by removing any floating solids in the original image. Not only is
    the skeleton returned but also the image with floating solids removed.

    Parameters
    ----------
    im : ndarray
        A binary image of porous media with 'True' values indicating
        phase of interest.
    padding : integer
        The amount of padding to add to the image before determining the
        skeleton.
    parallel : boolean
        If `False` the skeleton is calculated in serial. This is the default
        mode. However, if `True`, the skeleton is calculated in parallel using
        dask.

    Returns
    -------
    sk : ndarray
        Skeleton of image

    """
    # trim floating solid
    if im.ndim == 3:
        im = trim_floating_solid(im, conn=6, surface=True)
    # add pading
    padded = np.pad(im, padding, mode='reflect')
    # perform skeleton
    if parallel is False:  # serial
        sk = ski.morphology.skeletonize_3d(padded).astype('bool')
    if parallel is True:
        sk = skeleton_parallel(padded, **kwargs)
    # remove padding
    sk = unpad(sk, pad_width=padding)
    return sk


def _check_skeleton_health(sk):
    r"""
    This function checks the health of the skeleton by looking for any shells.

    Parameters
    ----------
    sk : ndarray
        The skeleton of an image

    Returns
    -------
    N_shells : int
        The number of shells detected in the skeleton. If any shells are
        detected a warning is triggered.
    """
    _, N = spim.label(input=~sk.astype('bool'))
    N_shells = N - 1
    if N_shells > 0:
        logger.warning(f"{N_shells} shells were detected in the skeleton. "
                       "Trim floating solids using: "
                       "porespy.filters.trim_floating_solid()")
    return N_shells


if __name__ == "__main__":
    '''
    Simulation using MAGNET extraction

    '''
    import matplotlib.pyplot as plt
    import porespy as ps
    import skimage as ski
    import numpy as np
    import openpnm as op
    from openpnm.algorithms._solution import SteadyStateSolution, SolutionContainer
    
    np.random.seed(1)
    
    # Define images of spheres
    Dp = 39
    resolution = 1e-6
    im2d = ps.generators.blobs([100, 100], porosity=0.6, blobiness=2)
    im1 = ps.generators.lattice_spheres([247, 247, 247], r=10, spacing=19, offset=9)
    im2 = ps.generators.lattice_spheres([273, 273, 273], r=20, spacing=39, offset=19)
    
    # import Blunt images
    name = 'Berea'
    resolution = 5.345e-6
    shape = [400, 400, 400]

    raw = np.fromfile(name + '.raw', dtype=np.uint8)
    imb = (raw.reshape(shape[0], shape[1], shape[2]))
    imb = imb == 0;
    imb = ps.filters.trim_floating_solid(imb, conn=6)
    
    # Choose im
    im = imb
    
    # take the skeleton
    sk = ski.morphology.skeletonize_3d(im).astype('bool')
    # analyze skeleton
    pt = analyze_skeleton(sk)
    # insert pore bodies
    b = square(3) if im.ndim == 2 else cube(3)
    dt = spim.maximum_filter(edt(im), footprint=b)
    fbd = insert_pore_bodies(sk=sk, dt=dt, pt=pt, l_max=5)
    # get network
    net = spheres_to_network(sk, dt, fbd, pt, voxel_size=resolution, boundary_width=10)
    
    if im.ndim == 2:
        # inspect by plotting
        plt.figure(1)
        plt.imshow(~im + sk + pt.endpts*10 + pt.juncs*5)
        plt.axis('off')
        plt.figure(2)
        plt.imshow(~im + sk + pt.endpts*10 + pt.juncs*10 + fbd.Ps.astype('bool')*5)
        plt.axis('off')
    
    net = op.io.network_from_porespy(net)
    net['pore.diameter'] = net['pore.radius']*2
    net['throat.diameter'] = net['throat.radius']*2

    h = op.utils.check_network_health(net)
    op.topotools.trim(net, pores=np.append(h['disconnected_pores'], h['isolated_pores']))
    h = op.utils.check_network_health(net)
    print(h)

    geo = op.models.collections.geometry.cubes_and_cuboids.copy()
    del geo['pore.diameter'], geo['throat.diameter']
    net['pore.diameter'] = net['pore.diameter'].copy()
    net['throat.diameter'] = net['throat.diameter'].copy()
    net.add_model_collection(geo)
    net.regenerate_models()

    phase = op.phase.Phase(network=net)
    phase['pore.viscosity'] = 1e-3

    phys = op.models.collections.physics.basic.copy()
    phase.add_model_collection(phys)
    phase.regenerate_models()

    Pin = 5e-6
    Pout = 0
    A = (im.shape[0]*im.shape[1]) * resolution**2
    L = im.shape[2] * resolution
    mu = phase['pore.viscosity'].max()

    # label boundary pores
    xmin = net.pores('xmin')
    xmax = net.pores('xmax')
    flow_x = op.algorithms.StokesFlow(network=net, phase=phase)
    flow_x.set_value_BC(pores=xmin, values=Pin)
    flow_x.set_value_BC(pores=xmax, values=Pout)
    flow_x.run()

    Q_x = flow_x.rate(pores=xmin, mode='group')[0]
    K_x = Q_x * L * mu / (A * (Pin - Pout))
    print(f'K_x is: {K_x/0.98e-12*1000:.2f} mD')

    ymin = net.pores('ymin')
    ymax = net.pores('ymax')
    flow_y = op.algorithms.StokesFlow(network=net, phase=phase)
    flow_y.set_value_BC(pores=ymin, values=Pin)
    flow_y.set_value_BC(pores=ymax, values=Pout)
    flow_y.run()

    Q_y = flow_y.rate(pores=ymin, mode='group')[0]
    K_y = Q_y * L * mu / (A * (Pin - Pout))
    print(f'K_y is: {K_y/0.98e-12*1000:.2f} mD')

    zmin = net.pores('zmin')
    zmax = net.pores('zmax')
    flow_z = op.algorithms.StokesFlow(network=net, phase=phase)
    flow_z.set_value_BC(pores=zmin, values=Pin)
    flow_z.set_value_BC(pores=zmax, values=Pout)
    flow_z.run()

    Q_z = flow_z.rate(pores=zmin, mode='group')[0]
    K_z = Q_z * L * mu / (A * (Pin - Pout))
    print(f'K_z is: {K_z/0.98e-12*1000:.2f} mD')

    K = np.average([K_x, K_y, K_z])
    print(f'K is: {K/0.98e-12*1000:.2f} mD')

    epsilon = np.sum(im)/np.product(im.shape)
    phi = 1
    Dp = Dp*resolution
    Kc = phi**2*epsilon**3*Dp**2/180/(1-epsilon)**2
    print(f'Carmen-Kozeny K is: {Kc/0.98e-12*1000:.2f} mD')

    phase = op.phase.Water(network=net)
    alg = op.algorithms.FickianDiffusion(network=net, phase=phase)
    alg['pore.concentration'] = np.zeros(net.Np)
    alg.soln = SolutionContainer()
    alg.soln['pore.concentration'] = SteadyStateSolution(np.zeros(net.Np))
    ps.io.to_stl(~sk, 'sk')
    ps.io.to_stl(im, 'im')
    net['pore.coords'] += 10*resolution
    flow_x['pore.pressure'] = np.zeros(net.Np)
    flow_y['pore.pressure'] = np.zeros(net.Np)
    flow_z['pore.pressure'] = np.zeros(net.Np)
    flow_x.soln['pore.pressure'] = SteadyStateSolution(np.zeros(net.Np))
    flow_y.soln['pore.pressure'] = SteadyStateSolution(np.zeros(net.Np))
    flow_z.soln['pore.pressure'] = SteadyStateSolution(np.zeros(net.Np))
    proj = net.project
    op.io.project_to_xdmf(proj, filename='network')
    # op.io.project_to_vtk(proj, filename='net')
    
    
    

    