import numpy as np
import scipy as sp
import scipy.ndimage as spim
from edt import edt
from skimage.morphology import square, cube
from porespy.tools import insert_sphere, make_contiguous, get_tqdm
from porespy.tools import extend_slice, Results


def find_junctions(sk):
    r"""
    find all the junctions in the skelton. It uses convolution to find voxels
    with extra neighbors, as well as terminal points on the ends of branches.
    parameters
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


def find_pore_bodies(sk, dt, pt):
    r"""
    Insert spheres at each junction point of the skeleton corresponding to
    the local size. A search for local maximums is performed along throats
    between inserted spheres. Additional spheres are inserted where any local
    maximums are found.
    parameters
    ------------
    sk : ndarray
        The skeleton of an image (boolean).
    dt : ndarray
        The distance transform of the phase of interest.
    pt : Results object
        A custom object returned from find_junctions()
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
    """
    mask = (pt.endpts * dt) >= 3  # remove endpoints with dt < 3
    pt = pt.juncs_r + pt.endpts * mask
    c = np.vstack(np.where(pt)).T
    Ps = np.zeros_like(pt, dtype=int)
    # initialize p_coords
    p_coords = []
    # insert spheres at junctions and endpoints
    d = np.insert(c, pt.ndim, dt[np.where(pt)].T, axis=1)
    d = np.flip(d[dt[np.where(pt)].T.argsort()], axis=0)
    # place n where there is a pore in the empty image Ps
    for n, row in enumerate(d):
        coords = tuple(row[0:pt.ndim])
        if Ps[coords] == 0:
            p_coords.append(coords)  # best to record p_coords here
            insert_sphere(im=Ps, c=np.hstack(coords), r=dt[coords]/1., v=n+1,
                          overwrite=True)
    # Find maximums on long throats
    temp = Ps * np.inf
    mask = np.isnan(temp)
    temp[mask] = 0
    temp = temp + dt * sk
    b = square(7) if pt.ndim == 2 else cube(7)
    mx = (spim.maximum_filter(temp, footprint=b) == dt) * (~(Ps > 0)) * sk
    mx = reduce_peaks(mx)
    # remove mx with dt < 3
    mask = (mx * dt) >= 3
    mx = mx * mask
    # insert spheres along long throats
    c = np.vstack(np.where(mx)).T
    Ps1_number = n
    # insert spheres at local maximums
    d = np.insert(c, mx.ndim, dt[np.where(mx)].T, axis=1)
    d = np.flip(d[d[:, mx.ndim].argsort()], axis=0)
    for n, row in enumerate(d):
        coords = tuple(row[0:mx.ndim])
        if Ps[coords] == 0:
            p_coords.append(coords)  # continue to record p_coords
            ss = n + Ps1_number + 1
            insert_sphere(im=Ps, c=np.hstack(coords), r=dt[coords]/1.,
                          v=ss+1, overwrite=False)
    # make pore numbers sequential
    Ps = make_contiguous(Ps)
    # second image for finding throat connections
    Ps2 = ((pt + mx) > 0) * Ps
    f = square(4) if Ps.ndim == 2 else cube(4)
    Ps2 = spim.maximum_filter(Ps2, footprint=f)
    # results object
    fbd = Results()
    fbd.Ps = Ps
    fbd.Ps2 = Ps2
    fbd.mx = mx
    fbd.p_coords = np.array(p_coords)
    return fbd


def find_throat_skeleton(sk, pt, fbd):
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
        'throats'
        The skeleton segmented by having all junctions and local maximum points
        removed
    """
    # segment sk into throats
    mx = fbd.mx
    pt = pt.juncs
    throats = (~((pt + mx) > 0)) * sk
    # results object
    ts = Results()
    ts.throats = throats
    return ts


def spheres_to_network(sk, fbd, ts, voxel_size=1):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.
    parameters
    ------------
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
    # create structuring element
    s = spim.generate_binary_structure(fbd.Ps.ndim, 2)
    throats, num_throats = spim.label(ts.throats, structure=s)
    slicess = spim.find_objects(throats)  # Nt by 2
    t_conns = np.zeros((len(slicess), 2), dtype=int)  # initialize
    # Initialize arrays
    Ps = np.arange(1, np.amax(fbd.Ps)+1)  # check that this works!!
    Np = np.size(Ps)
    p_volume = np.zeros((Np, ), dtype=float)
    for i in range(num_throats):
        throat_l = i
        if slicess[throat_l] is None:
            continue
        ss = extend_slice(slicess[throat_l], fbd.Ps2.shape)
        sub_im_p = fbd.Ps2[ss]
        sub_im_l = throats[ss]
        throat_im = sub_im_l == i+1
        if len(fbd.Ps2.shape) == 3:
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
    # remove duplicates in t_conns
    remove = np.where(t_conns[:, 0] == t_conns[:, 1])
    np.delete(t_conns, remove, axis=0)
    # pore volume
    _, counts = np.unique(fbd.Ps, return_counts=True)
    p_volume = counts[1:]
    # pore coords
    p_coords = fbd.p_coords
    if fbd.Ps.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = np.vstack((p_coords.T, np.zeros((Np, )))).T
    # create network dictionary
    net = {}
    ND = fbd.Ps.ndim
    # Define all the fundamentals for a pore network
    net['throat.conns'] = t_conns
    net['pore.coords'] = p_coords * voxel_size
    # Define geometric information
    V = np.copy(p_volume)*(voxel_size**ND)
    net['pore.volume'] = V
    f = 3/4 if ND == 3 else 1.0
    net['pore.equivalent_diameter'] = 2*(V/np.pi * f)**(1/ND)
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


if __name__ == "__main__":

    '''

    Simulation using MAGNET extraction

    '''

    # import packages
    import porespy as ps
    import openpnm as op
    import skimage as ski
    import matplotlib.pyplot as plt
    import time

    ps.visualization.set_mpl_style()
    np.random.seed(10)

    # %% Generate a test image
    im = ps.generators.blobs(shape=[400, 400], porosity=0.7, blobiness=2)
    plt.figure(1)
    plt.imshow(np.flip(im.T, axis=0))
    plt.axis('off')
    # Parameters
    mu = 1  # Pa s
    Pin = 1  # 10kPa/m
    Pout = 0
    Delta_P = Pin - Pout
    A = 1e-2
    L = 1e-5

    # %% MAGNET Extraction
    start_m = time.time()
    # create skeleton
    sk = ski.morphology.skeletonize_3d(im)/255
    # find all the junctions
    pt = ps.filters.find_junctions(sk)
    # distance transform
    dt = edt(im)
    # find pore bodies
    fbd = ps.filters.find_pore_bodies(sk, dt, pt)
    # throat segmentation
    ts = ps.filters.find_throat_skeleton(sk, pt, fbd)
    # create network object
    checkpoint_m = time.time()
    net = ps.filters.spheres_to_network(sk, fbd, ts, voxel_size=1)

    end_m = time.time()
    print('MAGNET Extraction Complete')
    net_m = op.network.from_porespy(net)

    # visualize MAGNET network
    plt.figure(2)
    fig, ax = plt.subplots(figsize=[5, 5]);
    slice_m = im.T
    ax.imshow(slice_m, cmap=plt.cm.bone)
    op.visualization.plot_coordinates(ax=fig,
                                      network=net_m,
                                      size_by=net_m["pore.equivalent_diameter"],
                                      color_by=net_m["pore.equivalent_diameter"],
                                      markersize=200)
    op.visualization.plot_connections(network=net_m, ax=fig)
    ax.axis("off");
    print('Visualization Complete')

    # find surface pores
    op.topotools.find_surface_pores(net_m)
    surface_pores = net_m['pore.surface']

    # label left, right, back, and front pores
    [a, b] = [30, 375]
    net_m['pore.left'] = np.zeros(net_m.Np, dtype=bool)
    net_m['pore.left'][surface_pores] = net_m['pore.coords'][surface_pores][:, 0] < a
    net_m['pore.right'] = np.zeros(net_m.Np, dtype=bool)
    net_m['pore.right'][surface_pores] = net_m['pore.coords'][surface_pores][:, 0] > b

    # Move boundary pores to edge of domain
    # left
    left = net_m.pores('left')
    temp = np.zeros(net_m.Np)
    temp[left] = net_m['pore.coords'][:, 0][left]
    net_m['pore.coords'][:, 0] -= temp  # don't index pore.coords
    # right
    right = net_m.pores('right')
    temp = np.zeros(net_m.Np)
    temp[right] = net_m['pore.coords'][:, 0][right] - im.shape[0]
    net_m['pore.coords'][:, 0] -= temp  # don't index pore.coords

    # network health
    h = op.utils.check_network_health(net_m)
    dis_pores = np.zeros(net_m.Np, dtype=bool)
    dis_pores[h['disconnected_pores']] = True
    net_m['pore.disconnected_pores'] = dis_pores
    Ps_trim = h['disconnected_pores']
    Ts_trim = np.append(h['duplicate_throats'], h['looped_throats'])
    op.topotools.trim(net_m, pores=Ps_trim, throats=Ts_trim)

    # visualize MAGNET network
    plt.figure(3)
    fig, ax = plt.subplots(figsize=[5, 5]);
    slice_m = im.T
    ax.imshow(slice_m, cmap=plt.cm.bone)
    op.visualization.plot_coordinates(ax=fig,
                                      network=net_m,
                                      size_by=net_m["pore.equivalent_diameter"],
                                      color_by=net_m["pore.left"],
                                      markersize=200)
    op.visualization.plot_connections(network=net_m, ax=fig)
    ax.axis("off");
    print('Visualization Complete')

    # %% Run Stokes Flow algorithm on extracted network
    # collection of geometry models, delete pore.diameter and pore.volume models
    geo = op.models.collections.geometry.spheres_and_cylinders
    del geo['pore.diameter'], geo['pore.volume']
    # set pore.diameter
    net_m['pore.diameter'] = net_m['pore.equivalent_diameter'].copy()
    # add geometry models to network
    net_m.add_model_collection(geo)
    net_m.regenerate_models()

    # phase
    phase_m = op.phase.Phase(network=net_m)
    phase_m['pore.viscosity'] = mu

    # add physics models to geometry
    phys = op.models.collections.physics.basic
    phase_m.add_model_collection(phys)
    phase_m.regenerate_models()

    # Stokes flow algorithm on MAGNET network
    inlet_m = net_m.pores('left')
    outlet_m = net_m.pores('right')
    flow_m = op.algorithms.StokesFlow(network=net_m, phase=phase_m)
    flow_m.set_value_BC(pores=inlet_m, values=Pin)
    flow_m.set_value_BC(pores=outlet_m, values=Pout)
    flow_m.run()
    print('MAGNET Simulation Complete')

    # Calculate permeability from MAGNET extraction
    Q_m = flow_m.rate(pores=inlet_m, mode='group')[0]
    K_m = Q_m * L * mu / (A * Delta_P)
    print(f'MAGNET K is: {K_m/0.98e-12*1000:.2f} mD')

    # Calculate extraction times and output
    time_m = end_m - start_m
    time_s2n = end_m - checkpoint_m
    print(f'MAGNET extraction time is: {time_m:.2f} s')
    print(f'Spheres to Network time is: {time_s2n:.2f} s')