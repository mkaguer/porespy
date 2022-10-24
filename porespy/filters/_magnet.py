import numpy as np
import scipy as sp
import scipy.ndimage as spim
from edt import edt
from skimage.morphology import square, cube
from porespy.tools import _insert_disk_at_points, make_contiguous
from porespy.tools import extend_slice, Results
import time
import dask.array as da
from skimage.morphology import skeletonize_3d
from porespy import settings
from porespy.filters._snows import _estimate_overlap


def magnet(im, sk=None, voxel_size=1, l_max=7):
    r"""
    find all the junctions in the skelton. It uses convolution to find voxels
    with extra neighbors, as well as terminal points on the ends of branches.
    parameters
    ------------
    im : ndarray
        An image of the porous material of interest
    sk : ndarray
        The skeleton of an image (boolean).
    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be voxel_size-cubed
    Returns
    -------
    net : dict
        A dictionary containing all the pore and throat size data, as well as
        the network topological information.  The dictionary names use the
        OpenPNM convention (i.e. 'pore.coords', 'throat.conns').
    """
    if sk is None:
        sk = ski.morphology.skeletonize_3d(im)/255
    # find junction points
    pt = ps.filters.find_junctions(sk)
    # distance transform
    dt = edt(im)
    # insert pores at junction points
    fbd = ps.filters.find_pore_bodies(sk, dt, pt, l_max)
    # find throat skeleton
    ts = ps.filters.find_throat_skeleton(sk, pt, fbd)
    # convert spheres to network dictionary
    net = ps.filters.spheres_to_network(sk, fbd, ts, voxel_size=voxel_size)
    return net
    

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


def find_pore_bodies(sk, dt, pt, l_max=7):
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
    l_max: int
        The length of the cubical structuring element to use in the maximum
        filter for inserting pores along long throats
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
    mask = (pt.endpts * dt) >= 3  # remove endpoints with dt < 3
    pts = pt.juncs_r + pt.endpts * mask
    c = np.vstack(np.where(pts)).T
    Ps = np.zeros_like(pts, dtype=int)
    # initialize p_coords
    p_coords = []
    p_radius = []
    # Find number of dimensions
    ND = pts.ndim
    # insert spheres at junctions and endpoints
    d = np.insert(c, ND, dt[np.where(pts)].T, axis=1)
    d = np.flip(d[dt[np.where(pts)].T.argsort()], axis=0)
    d = np.delete(d, np.where(d[:, ND] == 0), axis=0)  # FIXME: temporary fix
    # place n where there is a pore in the empty image Ps
    for n, row in enumerate(d):
        coords = tuple(row[0:ND])
        if Ps[coords] == 0:
            p_coords.append(coords)  # best to record p_coords here
            p_radius.append(dt[coords])  # and p_radius here
            _insert_disk_at_points(im=Ps,
                                   coords=np.hstack(coords).reshape((ND, 1)),
                                   r=round(dt[coords]),
                                   v=n+1,
                                   overwrite=True)
    # Find maximums on long throats
    temp = Ps * np.inf
    mask = np.isnan(temp)
    temp[mask] = 0
    temp = temp + dt * sk
    b = square(l_max) if ND == 2 else cube(l_max)
    mx = (spim.maximum_filter(temp, footprint=b) == dt) * (~(Ps > 0)) * sk
    # mx = reduce_peaks(mx)
    # remove mx with dt < 3
    mask = (mx * dt) >= 3
    mx = mx * mask
    # insert spheres along long throats
    c = np.vstack(np.where(mx)).T
    Ps1_number = n
    # insert spheres at local maximums
    d = np.insert(c, ND, dt[np.where(mx)].T, axis=1)
    d = np.flip(d[d[:, ND].argsort()], axis=0)
    for n, row in enumerate(d):
        coords = tuple(row[0:ND])
        if Ps[coords] == 0:
            p_coords.append(coords)  # continue to record p_coords
            p_radius.append(dt[coords])  # and p_radius here
            ss = n + Ps1_number + 1
            _insert_disk_at_points(im=Ps,
                                   coords=np.hstack(coords).reshape((ND, 1)),
                                   r=round(dt[coords]),
                                   v=ss+1,
                                   overwrite=False)
    # make pore numbers sequential
    Ps = make_contiguous(Ps)
    # second image for finding throat connections
    Ps2 = ((pts + mx) > 0) * Ps
    f = square(4) if ND == 2 else cube(4)
    Ps2 = spim.maximum_filter(Ps2, footprint=f)
    # results object
    fbd = Results()
    fbd.Ps = Ps
    fbd.Ps2 = Ps2
    fbd.mx = mx
    fbd.p_coords = np.array(p_coords)
    fbd.p_radius = np.array(p_radius)
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
    s = spim.generate_binary_structure(fbd.Ps.ndim, fbd.Ps.ndim)
    throats, num_throats = spim.label(ts.throats, structure=s)
    slicess = spim.find_objects(throats)  # Nt by 2
    t_conns = np.zeros((len(slicess), 2), dtype=int)  # initialize
    # Initialize arrays
    Ps = np.arange(1, np.amax(fbd.Ps)+1)  # check that this works!!
    Np = np.size(Ps)
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
    t_conns = np.delete(t_conns, remove, axis=0)
    # pore coords
    p_coords = fbd.p_coords.astype('float')
    if fbd.Ps.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = np.vstack((p_coords.T, np.zeros((Np, )))).T
    # create network dictionary
    net = {}
    # Define all the fundamentals for a pore network
    net['throat.conns'] = t_conns
    net['pore.coords'] = p_coords * voxel_size
    # Define geometric information
    # add radius
    net['pore.radius'] = fbd.p_radius * voxel_size
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


def skeleton_parallel(im, overlap=None, cores=None):
    r"""
    Performs `skimage.morphology.skeleton_3d` in parallel using dask

    Parameters
    ----------
    im : ndarray
        A binary image of porous media with 'True' values indicating
        phase of interest.
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
        overlap = _estimate_overlap(im, mode='dt') # FIXME: perform dt once!
    if cores is None:
        cores = settings.ncores
    divs = cores
    depth = {}
    for i in range(im.ndim):
        depth[i] = np.round(overlap).astype(int)
    chunk_shape = (np.array(im.shape) / np.array(divs)).astype(int)
    skel = da.from_array(im, chunks=chunk_shape)
    skel = da.overlap.overlap(skel, depth=depth, boundary='none')
    skel = skel.map_blocks(skeletonize_3d)
    skel = da.overlap.trim_internal(skel, depth, boundary='none')
    skel = skel.compute(num_workers=cores).astype(bool)
    return skel


if __name__ == "__main__":
    '''
    import porespy as ps
    import matplotlib.pyplot as plt
    import time
    np.random.seed(10)
    im = ps.generators.blobs([128, 128, 128], blobiness=0.7, porosity=0.65)
    im = ps.filters.trim_floating_solid(im)
    # perform skeleton_3d
    t0 = time.time()
    sk = skeletonize_3d(im).astype(bool)
    print(f"Elapsed time (skimage): {time.time() - t0:.2f} s")
    # perform skeleton_3d in parallel
    t0 = time.time()
    sk_p = skeleton_parallel(im)
    print(f"Elapsed time (parallel): {time.time() - t0:.2f} s")
    # plot
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    if im.ndim == 3:
        ax[0].imshow(sk[:, :, 50], origin="lower")
        ax[1].imshow(sk_p[:, :, 50], origin="lower")
    else:
        ax[0].imshow(sk, origin="lower")
        ax[1].imshow(sk_p, origin="lower")
    N_ones = sk.sum()
    N_ones_not_matched = (sk != sk_p).sum()
    accuracy = (1 - N_ones_not_matched / N_ones)
    print(f"Accuracy: {accuracy*100:.9f} %")
    '''
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

    twod = False

    # %% Generate a test image
    if twod:
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
        [a, b] = [30, 375]
        voxel_size = 1

    if not twod:
        im = ps.generators.blobs(shape=[256, 256, 256], porosity=0.7, blobiness=2)
        # trim floating solid
        im = ps.filters.trim_floating_solid(im, conn=6)
        # Parameters
        x, y, z = 2.25e-6, 2.25e-6, 2.25e-6
        mu = 1e-3  # Pa s
        Pin = 22.5  # 10kPa/m
        Pout = 0
        Delta_P = Pin - Pout
        A = (y * 1000) * (z * 1000)
        L = x * 1000
        voxel_size = x
        a = x * 30
        b = x * 225

    # %% MAGNET Extraction
    start_m = time.time()
    net = magnet(im, voxel_size=voxel_size, l_max=7)
    end_m = time.time()
    print('MAGNET Extraction Complete')

    net_m = op.io.network_from_porespy(net)
    net_m['pore.diameter'] = net_m['pore.radius'] * 2

    # network health
    h = op.utils.check_network_health(net_m)
    dis_pores = np.zeros(net_m.Np, dtype=bool)
    dis_pores[h['disconnected_pores']] = True
    net_m['pore.disconnected_pores'] = dis_pores
    Ps_trim = h['disconnected_pores']
    Ts_trim = np.append(h['duplicate_throats'], h['looped_throats'])
    op.topotools.trim(net_m, pores=Ps_trim, throats=Ts_trim)

    # visualize MAGNET network
    if twod:
        plt.figure(2)
        fig, ax = plt.subplots(figsize=[5, 5]);
        slice_m = im.T
        ax.imshow(slice_m, cmap=plt.cm.bone)
        op.visualization.plot_coordinates(ax=fig,
                                          network=net_m,
                                          size_by=net_m["pore.diameter"],
                                          color_by=net_m["pore.diameter"],
                                          markersize=200)
        op.visualization.plot_connections(network=net_m, ax=fig)
        ax.axis("off");
        print('Visualization Complete')

    # find surface pores
    op.topotools.find_surface_pores(net_m)
    surface_pores = net_m['pore.surface']

    # label left, right, back, and front pores
    net_m['pore.left'] = np.zeros(net_m.Np, dtype=bool)
    net_m['pore.left'][surface_pores] = net_m['pore.coords'][surface_pores][:, 0] < a
    net_m['pore.right'] = np.zeros(net_m.Np, dtype=bool)
    net_m['pore.right'][surface_pores] = net_m['pore.coords'][surface_pores][:, 0] > b
    '''
    # add boundary pores
    left = net_m.pores(['left', 'surface'], mode='and')
    right = net_m.pores(['right', 'surface'], mode='and')        
    op.topotools.add_boundary_pores(net_m,
                                    pores=left,
                                    move_to=[0, None, None],
                                    apply_label='left_boundary')
    op.topotools.add_boundary_pores(net_m,
                                    pores=right,
                                    move_to=[im.shape[0]*voxel_size, None, None],
                                    apply_label='right_boundary')

    # assign parent radius to cloned pores
    left_boundary = net_m.pores('left_boundary')
    right_boundary = net_m.pores('right_boundary')
    net_m['pore.radius'][left_boundary] = net_m['pore.radius'][left]
    net_m['pore.radius'][right_boundary] = net_m['pore.radius'][right]
    net_m['pore.diameter'] = net_m['pore.radius'] * 2
    '''
    # visualize MAGNET network
    if twod:
        plt.figure(3)
        fig, ax = plt.subplots(figsize=[5, 5]);
        slice_m = im.T
        ax.imshow(slice_m, cmap=plt.cm.bone)
        op.visualization.plot_coordinates(ax=fig,
                                          network=net_m,
                                          size_by=net_m["pore.diameter"],
                                          color_by=net_m["pore.left"],
                                          markersize=200)
        op.visualization.plot_connections(network=net_m, ax=fig)
        ax.axis("off");
        print('Visualization Complete')

    # %% Run Stokes Flow algorithm on extracted network
    # collection of geometry models, delete pore.diameter and pore.volume models
    geo = op.models.collections.geometry.cones_and_cylinders
    del geo['pore.diameter'], geo['pore.volume']
    # set pore.diameter
    net_m['pore.diameter'] = net_m['pore.diameter'].copy()
    # add geometry models to network
    net_m.add_model_collection(geo)
    net_m.regenerate_models()
    
    # throat diameter of boundary pores
    # left_throats = net_m.find_neighbor_throats(left_boundary)
    # net_m['throat.diameter'][left_throats] = net_m['pore.diameter'][left_boundary]
    # right_throats = net_m.find_neighbor_throats(right_boundary)
    # net_m['throat.diameter'][right_throats] = net_m['pore.diameter'][right_boundary]
    # net_m.regenerate_models(exclude=['throat.diameter'])
    
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
    print(f'MAGNET extraction time is: {time_m:.2f} s')
    
    net_m['throat.radius'] = net_m['throat.diameter']/2
    project = net_m.project
    op.io.project_to_vtk(project, filename='magnet-256')
    op.io.project_to_xdmf(project, filename='magnet-256')
