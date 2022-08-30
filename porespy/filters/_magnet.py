import numpy as np
import scipy.ndimage as spim
from edt import edt
from skimage.morphology import square, cube
from skimage.segmentation import find_boundaries
from porespy import settings
from porespy.tools import insert_sphere, make_contiguous, get_tqdm
from porespy.tools import extend_slice, Results
tqdm = get_tqdm()


def find_junctions(sk, asmask=True):
    r"""
    find all the junctions in the skelton. It uses convolution to fine voxels
    with extra neighbors, as well as terminal points on the ends of branches.
    parameters
    ------------
    sk : ndarray
        The skeleton of an image (boolean).
    asmask: bool
        if 'False' the function returns the transpose of junctions.
    Returns
    -------
    pt : ndarray
        the junctions in the skelton.
    """
    if sk.ndim == 2:
        a = square(3)
    else:
        a = cube(3)
    conv = spim.convolve(sk*1.0, a)
    pt = (conv >= 4)*sk
    pt += (conv == 2)*sk
    pt = reduce_peaks(pt)
    if not asmask:
        pt = np.vstack(np.where(pt)).T
    return pt


def find_pore_bodies(im, sk, pt, dt):
    r"""
    Insert spheres at each junction point of the skeleton corresponding to
    the local size. A sphere is not inserted at junctions with local size less
    than three. Additionally, distance transform is performed relative to
    the both solid and pores, then any locations that are equal to the values
    in the original distance transform are selected to insert a new pore.
    parameters
    ------------
    im : ndarray
        The image of the pore space
    sk : ndarray
        The skeleton of an image (boolean).
    pt : ndarray
        The junctions in the skeleton.
    dt : ndarray
        The distance transform of the phase of interest.
    Returns
    -------
    fbd : Results object
        A custom object with the following data added as named attributes:
        'Ps'
        Inserted spheres at each junction point corresponding to the local size
        'Ps2'
        Inserted spheres at each junction point AND where throats were long
    """
    c = np.vstack(np.where(pt)).T
    Ps = np.zeros_like(pt, dtype=int)
    if pt.ndim == 2:
        d = np.insert(c, 2, dt[np.where(pt)].T, axis=1)
        d = np.flip(d[d[:, 2].argsort()], axis=0)
        # delete junctions with dt < 3
        d = np.delete(d, np.where(d[:, 2] < 3), axis=0)
        # place n where there is a pore in the empty image Ps
        for n, (i, j, k) in enumerate(d):
            if Ps[i, j] == 0:
                insert_sphere(im=Ps, c=np.hstack((i, j)), r=dt[i, j]/1., v=n+1,
                              overwrite=True)
        b = square(7)
    else:
        d = np.insert(c, 3, dt[np.where(pt)].T, axis=1)
        d = np.flip(d[d[:, 3].argsort()], axis=0)
        # delete junctions with dt < 3
        d = np.delete(d, np.where(d[:, 3] < 3), axis=0)
        for n, (i, j, k, l) in enumerate(d):
            if Ps[i, j, k] == 0:
                insert_sphere(im=Ps, c=np.hstack((i, j, k)), r=dt[i, j, k]/1., v=n+1,
                              overwrite=True)
        b = cube(7)
    # Find maximums on long throats
    temp = Ps * np.inf
    mask = np.isnan(temp)
    temp[mask] = 0
    temp = temp + dt * sk
    mx = (spim.maximum_filter(temp, footprint=b) == dt) * (~(Ps > 0)) * sk
    # insert spheres along long throats
    c = np.vstack(np.where(mx)).T
    Ps1_number = n
    Ps2 = Ps.copy()
    if mx.ndim == 2:
        d = np.insert(c, 2, dt[np.where(mx)].T, axis=1)
        d = np.flip(d[d[:, 2].argsort()], axis=0)
        for n, (i, j, k) in enumerate(d):
            if Ps2[i, j] == 0:
                ss = n + Ps1_number + 1
                insert_sphere(im=Ps2, c=np.hstack((i, j)), r=dt[i, j]/1.,
                              v=ss+1, overwrite=False)
    else:
        d = np.insert(c, 3, dt[np.where(mx)].T, axis=1)
        d = np.flip(d[d[:, 3].argsort()], axis=0)
        for n, (i, j, k, l) in enumerate(d):
            ss = n + Ps1_number + 1
            insert_sphere(im=Ps2, c=np.hstack((i, j, k)), r=dt[i, j, k]/1.,
                          v=ss+1, overwrite=False)
    fbd = Results()
    fbd.Ps = Ps
    fbd.Ps2 = Ps2
    return fbd


def find_throat_skeleton(im, sk, fbd):
    r"""
    Identify throat segments corresponding to the overlapping pores by
    finding the border of each region, then finding the skeleton that
    overlaps this border.
    parameters
    ------------
    im : ndarray
        The image of the pore space
    sk : ndarray
        The skeleton of an image (boolean).
    fbd: ndarray
        Inserted spheres using find_pore_bodies function.
    Returns
    -------
    ts : Results object
        A custom object with the following data added as named attributes:
        'internal_throats'
        throats connecting overlapped pores
        'throats'
        throats connecting non-overlapped pores
    """
    # Throat segmentation
    bd1 = find_boundaries(fbd, mode='inner')
    bd2 = find_boundaries(fbd > 0, mode='inner')
    temp = bd1 * sk
    internal_throats = temp * (~bd2)
    throats = sk * (~(fbd > 0))
    ts = Results()
    ts.internal_throats = internal_throats
    ts.throats = throats
    ts.all = internal_throats*3.0 + throats*3.0 + bd2*1.0 + bd1*1.0 + (~im)*0.5
    return ts


def spheres_to_network(im, sk, fbd, throats, voxel_size=1):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.
    parameters
    ------------
    im : ndarray
        The image of the pore space
    sk : ndarray
        The skeleton of an image (boolean).
    fbd: ndarray
        Inserted spheres using find_pore_bodies function.
    throats : ndarray
        Segmented throats using find_throat_skeleton function.
    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.
    Returns
    -------
    net : dict
        A dictionary containing all the pore and throat size data, as well as
        the network topological information.  The dictionary names use the
        OpenPNM convention (i.e. 'pore.coords', 'throat.conns').
    """
    if len(fbd.shape) == 3:
        s = spim.generate_binary_structure(3, 2)
    else:
        s = spim.generate_binary_structure(2, 2)
    throats, num_throats = spim.label(throats, structure=s)
    slicess = spim.find_objects(throats)  # Nt by 2
    t_conns = np.zeros((len(slicess), 2), dtype=int)  # initialize
    t_coords = []
    phases = (fbd + throats > 0).astype(int)
    dt = edt(phases == 1)
    # Add distane transform for more than 2 phases
    for i in range(2, phases.max()+1):
        dt += edt(phases == i)
    fbd = make_contiguous(fbd)
    slices = spim.find_objects(fbd)
    # Initialize arrays
    Ps = np.arange(1, np.amax(fbd)+1)  # check that this works!!
    Np = np.size(Ps)
    p_coords_cm = np.zeros((Np, fbd.ndim), dtype=float)
    p_coords_dt = np.zeros((Np, fbd.ndim), dtype=float)
    p_coords_dt_global = np.zeros((Np, fbd.ndim), dtype=float)
    p_volume = np.zeros((Np, ), dtype=float)
    p_dia_local = np.zeros((Np, ), dtype=float)
    p_dia_global = np.zeros((Np, ), dtype=float)
    p_area_surf = np.zeros((Np, ), dtype=int)
    p_label = np.zeros((Np, ), dtype=int)
    p_phase = np.zeros((Np, ), dtype=int)
    tqdm = get_tqdm()

    for i in range(num_throats):
        throat_l = i
        if slicess[throat_l] is None:
            continue
        ss = extend_slice(slicess[throat_l], fbd.shape)
        sub_im_p = fbd[ss]
        sub_im_l = throats[ss]
        throat_im = sub_im_l == i+1
        padded_mask = np.pad(throat_im, pad_width=1, mode='constant')
        if len(fbd.shape) == 3:
            structure = spim.generate_binary_structure(3, 2)
        else:
            structure = spim.generate_binary_structure(2, 2)
        sub_sk = sk[ss]
        im_w_throats_l = spim.binary_dilation(input=throat_im,
                                              structure=structure)
        im_w_throats_l = im_w_throats_l * sub_sk
        im_w_throats_l = im_w_throats_l * sub_im_p
        Pn_l = np.unique(im_w_throats_l)[1:] - 1
        if np.any(Pn_l):
            Pn_l = Pn_l[0:2]
            t_conns[throat_l, :] = Pn_l
        for j in Pn_l:
            vx = np.where(im_w_throats_l == (j + 1))
            s_offset = np.array([i.start for i in ss])
            t_inds = tuple([i+j for i, j in zip(vx, s_offset)])
            temp = np.where(dt[t_inds] == np.amax(dt[t_inds]))[0][0]
            t_coords.append(tuple([t_inds[k][temp] for k in range(im.ndim)]))
    for i in tqdm(Ps, **settings.tqdm):
        pore = i - 1
        if slices[pore] is None:
            continue
        s = extend_slice(slices[pore], fbd.shape)
        sub_im = fbd[s]
        sub_dt = dt[s]
        pore_im = sub_im == i
        padded_mask = np.pad(pore_im, pad_width=1, mode='constant')
        pore_dt = edt(padded_mask)
        s_offset = np.array([i.start for i in s])
        p_label[pore] = i
        p_coords_cm[pore, :] = spim.center_of_mass(pore_im) + s_offset
        temp = np.vstack(np.where(pore_dt == pore_dt.max()))[:, 0]
        p_coords_dt[pore, :] = temp + s_offset
        p_phase[pore] = (phases[s]*pore_im).max()
        temp = np.vstack(np.where(sub_dt == sub_dt.max()))[:, 0]
        p_coords_dt_global[pore, :] = temp + s_offset
        p_volume[pore] = np.sum(pore_im)
        p_dia_local[pore] = 2*np.amax(pore_dt)
        p_dia_global[pore] = 2*np.amax(sub_dt)
        p_area_surf[pore] = np.sum(pore_dt == 1)
    # Clean up values
    p_coords = p_coords_cm
    if im.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = np.vstack((p_coords_cm.T, np.zeros((Np, )))).T
    # create network dictionary
    net = {}
    ND = im.ndim
    # Define all the fundamental stuff
    net['throat.conns'] = np.array(t_conns)
    net['pore.coords'] = np.array(p_coords)*voxel_size
    net['pore.all'] = np.ones_like(net['pore.coords'][:, 0], dtype=bool)
    net['pore.region_label'] = np.array(p_label)
    net['pore.phase'] = np.array(p_phase, dtype=int)
    V = np.copy(p_volume)*(voxel_size**ND)
    net['pore.region_volume'] = V  # This will be an area if image is 2D
    f = 3/4 if ND == 3 else 1.0
    net['pore.equivalent_diameter'] = 2*(V/np.pi * f)**(1/ND)
    # Extract the geometric stuff
    net['pore.local_peak'] = np.copy(p_coords_dt)*voxel_size
    net['pore.global_peak'] = np.copy(p_coords_dt_global)*voxel_size
    net['pore.geometric_centroid'] = np.copy(p_coords_cm)*voxel_size
    net['pore.volume'] = np.copy(p_volume)*(voxel_size**ND)
    net['pore.surface_area'] = np.copy(p_area_surf)*(voxel_size**2)
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
