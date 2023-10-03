import logging
import numpy as np
import scipy.ndimage as spim
import scipy.signal as spsg
import skimage as ski
from edt import edt
import dask.array as da
from skimage.morphology import square, cube, skeletonize, skeletonize_3d
from porespy.generators import borders
from porespy.tools import (
    make_contiguous,
    unpad,
    extract_subsection,
    ps_disk,
    ps_round,
    extend_slice,
    insert_sphere,
    Results,
    _make_disk,
    _make_ball,
)
from porespy import settings
from porespy.filters import (
    trim_floating_solid,
    fill_blind_pores,
)
from porespy.filters._snows import _estimate_overlap
from scipy.ndimage import maximum_position
from numba import njit


logger = logging.getLogger(__name__)


__all__ = [
    'magnet',
    'reduce_points',
    'merge_nearby_pores',
    'padded_image',
    'skeleton_parallel',
    'skeleton',
    'spheres_to_network',
    'insert_pore_bodies',
    'analyze_skeleton',
    'pad_faces_for_skeletonization',
    'skeletonize_magnet',
    'find_throat_points',
]


def magnet(im,
           sk=None,
           endpoints=True,
           padding=None,
           parallel=False,
           surface=False,
           voxel_size=1,
           l_max=7,
           boundary_width=3,
           **kwargs):
    r"""
    Perform a Medial Axis Guided Network ExtracTion (MAGNET) on an image of
    porous media. This is a modernized python implementation of a classical
    network extraction method. First, the skeleton of the provided image is
    determined. The skeleton can be computed in serial or parallel modes and
    padding can be optionally added to the image prior to taking the skeleton
    for help in identifying boundary pores. Next, all the junction points of
    the skeleton are determined by using convolution to find voxels with extra
    neighbors, as well as terminal points on the ends of branches. Pores are
    then inserted at these points. The size of the pores inserted is based on
    the distance transform value at it's junction. This approach results in
    many long throats so more pores are added using a maximum filter along long
    throats to find openings. To ensure an efficient network extraction method,
    only the most fundamential pore and throat properties are returned.

    Parameters
    ------------
    im : ndarray
        An image of the porous material of interest. Be careful of floating
        solids in the 3D image as this will result in a hollow shell after
        taking the skeleton. Floating solids are removed from the image by
        default prior to taking the skeleton.
    sk : ndarray
        Optionally provide your own skeleton of the image. If `sk` is `None` the
        skeleton is computed using `skimage.morphology.skeleton_3d`.  A check
        is made to ensure no shells are found in the resulting skeleton.
    endpoints : boolean
        If `True` pores are inserted at endpoints as well as junction points.
        This is the default mode. If `False`, endpoints are ignored. This is
        useful for flow simulations where endpoints are essentially dead ends
        where no flow occurs.
    padding : integer
        The amount of padding to add to the image before determining the
        skeleton. Padding helps coerce the skeleton to the edge of the image
        for easy determination of boundary pores. If 'None', no padding is
        added to the image. WARNING: this feature may create skeleton clusters
        near the edge of the image resulting in disconnected pores.
    surface : boolean
        If `False` disconnected solid at the surface of the image is NOT
        trimmed. This is the default mode. However, if `True`, disconnected
        solid at the surface of the image is trimmed.
    parallel : boolean
        If `False` the skeleton is calculated in serial. This is the default
        mode. However, if `True`, the skeleton is calculated in parallel using
        chunking in Dask. The default mode is `False`
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
        'throat.radius'). Labels for boundary pores and overlapping throats
        are also returned.
    sk : ndarray
        The skeleton of the image is also returned.
    """
    # get the skeleton
    if sk is None:
        sk, im = skeleton(im, padding, surface, parallel, **kwargs)
    else:
        if im.ndim == 3:
            _check_skeleton_health(sk.astype('bool'))
    # find junction and terminal points
    pt = analyze_skeleton(sk)
    if not endpoints:
        pts = pt.juncs
    else:
        pts = pt.juncs + pt.endpts
    # take the distance transform
    dt = edt(im)
    # b = square(3) if im.ndim == 2 else cube(3)
    # dt = spim.maximum_filter(dt, footprint=b)
    # insert pores at junction points
    fbd = insert_pore_bodies(sk, dt, pts, l_max)
    # convert spheres to network dictionary
    net = spheres_to_network(sk, dt, fbd, pt, voxel_size, boundary_width)
    return net, sk


def analyze_skeleton(sk):
    r"""
    Finds all the junction and end points in a skeleton.

    It uses convolution to find voxels with extra neighbors, as well as terminal
    points on the ends of branches with fewer neighbors.

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
    """
    # kernel for convolution
    if sk.ndim == 2:
        a = square(3)
    else:
        a = cube(3)
    # compute convolution directly or via fft, whichever is fastest
    conv = spsg.convolve(sk*1.0, a, mode='same', method='auto')
    conv = np.rint(conv).astype(int)  # in case of fft, accuracy is lost
    # find junction points of skeleton
    juncs = (conv >= 4) * sk
    # find endpoints of skeleton
    endpts = (conv == 2) * sk
    # results object
    pt = Results()
    pt.juncs = juncs
    pt.endpts = endpts
    return pt


def insert_pore_bodies(sk, dt, pt, l_max=7, numba=False):
    r"""
    Insert spheres at each junction and/or terminal points of the skeleton
    corresponding to the local size. A search for local maximums is performed
    along throats between inserted spheres. Additional spheres are inserted
    where any local maximums are found.

    Parameters
    ------------
    sk : ndarray
        The skeleton of an image (boolean).
    dt : ndarray
        The distance transform of the phase of interest.
    pt : ndarray
        The junction and/or terminal points of the skeleton at which to insert
        pores.
    l_max: int
        The length of the cubical structuring element to use in the maximum
        filter for inserting pores along long throats

    Returns
    -------
    fbd : Results object
        A custom object with the following data as named attributes:
        'Ps'
        Inserted spheres corresponding to the local size
        'pts'
        Points labeled corresponding to pore index
        'p_coords'
        The coordinates where each pore body is added
        'p_radius'
        The radius of each pore added
    """
    # label and reduce points
    ND = pt.ndim
    s = spim.generate_binary_structure(ND, ND)
    pts1, Np = spim.label(pt, structure=s)
    pts2 = reduce_points(pts1, dt)
    # find coords of junction/endpoints and sort
    c = np.vstack(np.where(pts2)).astype(int).T
    c = np.insert(c, ND, pts2[np.where(pts2)].T, axis=1)
    c = c[pts2[np.where(pts2)].T.argsort()]
    # insert pore bodies at points
    Ps = np.zeros_like(pts1, dtype=int)
    p_coords = []
    for n, row in enumerate(c):
        coord = row[0:ND]
        radius = np.round(dt[tuple(coord)])
        insert_sphere(im=Ps, c=coord, r=radius, v=1, overwrite=False)
        p_coords.append(coord)
    # Find maximums along long throats
    temp = Ps * np.inf
    mask = np.isnan(temp)
    temp[mask] = 0
    temp = temp + dt * sk
    if l_max is not None:
        b = square(l_max) if ND == 2 else cube(l_max)
        mx = (spim.maximum_filter(temp, footprint=b) == dt) * sk
        # label and reduce maximum points
        mx, _ = spim.label(mx, structure=s)
        mx = reduce_points(mx, dt)
        # remove any maximum points that lie on junction cluster!
        mx[pts1 > 0] = 0
        mx = make_contiguous(mx)  # make contiguous again
        # find coords of mx points and sort
        c = np.vstack(np.where(mx)).astype(int).T
        c = np.insert(c, ND, mx[np.where(mx)].T, axis=1)
        c = c[mx[np.where(mx)].T.argsort()]
        # add mx coords to list of coords
        p_coords = p_coords + list(c[:, 0:ND])
        # assign index to mx
        mx[mx > 0] += n + 1
        # merge mx with junction/endpoints
        pts1 = pts1 + mx
    # results object
    fbd = Results()
    fbd.Ps = Ps
    fbd.pts = pts1
    fbd.p_coords = np.array(p_coords)
    fbd.p_radius = np.array([dt[tuple(co)] for co in p_coords])
    return fbd


def merge_nearby_pores(network, Lmax):
    r"""
    Merges sets of pores that are within a given distance of each other

    Parameters
    ----------
    network : dict
        The OpenPNM network object
    Lmax : scalar
        Any pores within this distance of each other will be merged

    Returns
    -------
    network : dict
        The OpenPNM network with the pores merged

    Notes
    -----
    - This works even if the pores are not topologically connected
    - The *new* pore takes on the average values of the ones that are being merged,
      including average coordinates, sizes etc.  Labels are all set to False.
    - Throats connected to the pores that are being merged are kept and rejoined
      to the *new* pore, so keep all their original properties. This includes length
      which might change slightly.
    """
    from openpnm.topotools import extend, trim, bond_percolation
    from openpnm.models.network import pore_to_pore_distance
    L = pore_to_pore_distance(network)
    clusters = bond_percolation(network, L <= Lmax)
    labels = np.unique(clusters.site_labels)
    cluster_num = {v: [] for v in labels}
    for n, v in enumerate(clusters.site_labels):
        if v >= 0:
            cluster_num[v].append(n)
    _ = cluster_num.pop(-1, None)
    Np = network.Np
    props = network.props(element='pore')
    props.remove('pore.coords')
    for i, Ps in enumerate(cluster_num.values()):
        crds = np.mean(network['pore.coords'][Ps], axis=0)
        extend(network, pore_coords=[crds])
        for prop in props:
            network[prop][-1] = np.mean(network[prop][Ps])
        Ts = network.find_neighbor_throats(pores=Ps, mode='xor')
        conns = network.conns[Ts, :]
        mask = np.isin(conns, Ps)
        conns[mask] = Np + i
        network['throat.conns'][Ts, :] = np.sort(conns, axis=1)
    Ps = np.where(clusters.site_labels >= 0)[0]
    trim(network=network, pores=Ps)
    return network


def pad_faces_for_skeletonization(im, pad_width=5, r=3):
    r"""
    Pad faces of domain with solid with holes to force skeleton to edge of image

    Parameters
    ----------
    im : ndarray
        The boolean image of the porous media with `True` value indicating the
        void phase.
    pad_width : int or list
        This is passed to the `numpy.pad` function so refer to that method for
        details.
    r : int
        The radius of the holes to create.

    Returns
    -------
    im_padded : ndarray
        A image with solid on all sides that has holes at the local peaks of the
        distance transform.  Applying a skeletonization on this image will force
        the skeleton to draw branches to the edge of the image.

    """
    dt = edt(im)
    faces = borders(im.shape, mode='faces')
    mx = im * faces * (spim.maximum_filter(dt*faces, size=3) == dt)
    mx = np.pad(mx, pad_width, mode='edge')
    mx = spim.binary_dilation(mx, structure=ps_round(r, im.ndim, False))
    im_new = np.pad(im, pad_width, mode='constant', constant_values=False)
    im_new = im_new + mx
    return im_new


def spheres_to_network(sk, dt, fbd, pt, voxel_size=1, boundary_width=3):
    r"""
    Assemble a dictionary object containing essential topological and
    geometrical data for a pore network. The information is retrieved from the
    skeleton and a labelled image of corresponding junction, endpoints, and
    maximums. The essential throat and pore properties are throat connections,
    throat radii, pore coordinates, and pore radii. Labels are also created
    for overlapping throats and boundary pores.

    Parameters
    ------------
    sk : ndarray
        The skeleton of an image (boolean).
    dt : ndarray
        The distance transform of an image
    fbd: Results object
        A custom object returned from insert_pore_bodies()
    pt : Results object
        A custom object returned from analyze_skeleton()
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
    pts = fbd.pts
    throats = ~(pts > 0) * sk
    # label throats
    s = spim.generate_binary_structure(ND, ND)
    throats, num_throats = spim.label(throats, structure=s)
    # get slicess of throats
    slicess = spim.find_objects(throats)  # Nt by 2
    # initialize throat conns and radius
    t_conns = np.zeros((len(slicess), 2), dtype=int)
    t_radius = np.zeros((len(slicess)), dtype=float)
    # loop through throats to get t_conns and t_radius
    for i in range(num_throats):
        throat_l = i
        ss = extend_slice(slicess[throat_l], shape)
        # get slices
        sub_im_p = fbd.pts[ss]
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
            t_conns[throat_l, :] = Pn_l
        # throat radius
        throat_dt = throat_im * sub_dt
        t_radius[throat_l] = np.average(throat_dt[throat_dt != 0])
    # remove [0,0] case for isolated one pixel long throat
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
    t_overlapping = dist <= R1 + R2
    # set throat radius of overlapping pores
    Rs = np.hstack((R1.reshape((Nt, 1)), R2.reshape((Nt, 1))))
    Rmin = np.min(Rs, axis=1)
    t_radius[t_overlapping] = 0.99 * Rmin[t_overlapping]
    # ensure throat radius is smaller than pore radii
    mask = Rmin <= t_radius
    t_radius[mask] = 0.99 * Rmin[mask]
    # pore coords
    p_coords = fbd.p_coords.astype('float')
    if ND == 2:  # If 2D, add 0's in 3rd dimension
        Np = np.amax(fbd.pts)
        p_coords = np.vstack((p_coords.T, np.zeros((Np, )))).T
    # create network dictionary
    net = {}
    net['throat.conns'] = t_conns
    net['pore.coords'] = p_coords * voxel_size
    net['throat.radius'] = t_radius * voxel_size
    net['pore.radius'] = fbd.p_radius * voxel_size
    net['throat.overlapping'] = t_overlapping
    bw = boundary_width
    net['pore.xmin'] = p_coords[:, 0] <= 0 + bw
    net['pore.xmax'] = p_coords[:, 0] >= shape[0] - 1 - bw
    net['pore.ymin'] = p_coords[:, 1] <= 0 + bw
    net['pore.ymax'] = p_coords[:, 1] >= shape[1] - 1 - bw
    if ND == 3:
        net['pore.zmin'] = p_coords[:, 2] <= 0 + bw
        net['pore.zmax'] = p_coords[:, 2] >= shape[2] - 1 - bw
    return net


def skeleton(im, padding=None, surface=False, parallel=False, **kwargs):
    r"""
    Takes the skeleton of an image. This function ensures that no shells are
    found in the resulting skeleton by trimming floating solids from the image
    beforehand. The skeleton is taken using Lee's method as available in
    scikit-image. Padding can be optionally added to the image prior to taking
    the skeleton for easy determination of boundary pores. For faster
    skeletonization, a parallel mode is available.

    Parameters
    ----------
    im : ndarray
        A binary image of porous media with 'True' values indicating phase of
        interest.
    padding : integer
        The amount of padding to add to the image before determining the
        skeleton. Padding helps coerce the skeleton to the edge of the image
        for easy determination of boundary pores. If 'None', no padding is
        added to the image.
    surface : boolean
        If `False` disconnected solid at the surface of the image is NOT
        trimmed. This is the default mode. However, if `True`, disconnected
        solid at the surface of the image is trimmed.
    parallel : boolean
        If `False` the skeleton is calculated in serial. This is the default
        mode. However, if `True`, the skeleton is calculated in parallel using
        chunking in Dask.

    Returns
    -------
    sk : ndarray
        Skeleton of image
    """
    # add padding
    if padding is not None:
        im = padded_image(im, padding=padding)
    # trim floating solid from 3D images
    if im.ndim == 3:
        im = trim_floating_solid(im, conn=6, surface=surface)
    # perform skeleton
    if parallel is False:  # serial
        sk = ski.morphology.skeletonize_3d(im).astype('bool')
    if parallel is True:  # parallel
        sk = skeleton_parallel(im, **kwargs)
    # remove padding
    if padding is not None:
        sk = unpad(sk, pad_width=padding)
        im = unpad(im, pad_width=padding)
    return sk, im


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


def padded_image(im, padding=20):
    r"""
    Pads the image prior to taking the skeleton so that the resulting skeleton
    extends to the edge of the image. This is ideal for finding boundary pores.
    For 2D images, `edge` mode is used to apply padding to the image. For 3D
    images, the skeleton of each face is taken and small disks are inserted at
    junction and endpoints. The original image is then padded in such a way
    that small holes form from the inserted disks.

    Parameters
    ----------
    im : ndarray
        An image of the porous material of interest. Make sure to trim floating
        solids beforehand.
    padding : int
        The amount of padding to add to each face of the image prior to taking
        the skeleton.

    Returns
    -------
    padded : ndarray
        The original image with padding applied for skeletonization.
    """
    if im.ndim == 2:
        padded = np.pad(im, pad_width=padding, mode='edge')
    if im.ndim == 3:
        # extract faces
        xmin = im[0, :, :]
        ymin = im[:, 0, :]
        zmin = im[:, :, 0]
        xmax = im[-1, :, :]
        ymax = im[:, -1, :]
        zmax = im[:, :, -1]
        # take skeleton of each face and insert disks at junctions/endpoints
        faces=[]
        for face in [xmin, xmax, ymin, ymax, zmin, zmax]:
            temp = np.pad(face, padding, mode='edge')
            sk = ski.morphology.skeletonize_3d(temp).astype('bool')  # skeleton
            sk = extract_subsection(sk, im.shape)  # really handy function
            pt = analyze_skeleton(sk)  # find junction and endpoints
            dt = edt(face)
            # find centres where to insert disks
            pts = pt.juncs + pt.endpts
            s = spim.generate_binary_structure(2, 2)
            pts, _ = spim.label(pts, structure=s)
            centres = reduce_points(pts, dt)
            # insert disks at junction and endpoints
            disk = ps_disk(r=5, smooth=True)
            face = spim.binary_dilation(centres, structure=disk)
            face = np.pad(face, pad_width=1)
            faces.append(face)
        # pad original image by 1 voxel so new faces can be added
        padded = np.pad(im, pad_width=1)
        # add new faces to padded image
        padded[0, :, :] = faces[0]
        padded[-1, :, :] = faces[1]
        padded[:, 0, :] = faces[2]
        padded[:, -1, :] = faces[3]
        padded[:, :, 0] = faces[4]
        padded[:, :, -1] = faces[5]
        # pad image full amount using `edge` mode
        padded = np.pad(padded, pad_width=padding-1, mode='edge')
    return padded


def reduce_points(points, dt):
    r"""
    Clusters of junction points are reduced to a single voxel, whereby the new
    voxel, corresponds to the one that has the largest distance transform value
    from within the original cluster. This method, unlike reduce_peaks, ensures
    that the new voxel lies on the original set.

    Parameters
    ----------
    points : ndarray
        A lablled image containing clusters of junction points to be reduced.

    dt : ndarray
        The distance transform of the original image.

    Returns
    -------
    image : ndarray
        An array with the same number of isolated junction points as the
        original image, but without clustering.
    """
    reduced_pts = np.zeros_like(points, dtype=int)
    # find position of maximums by labelled cluster
    max_coords = maximum_position(dt, points, range(1, np.max(points)+1))
    # Get row and column coordinates within each cluster
    x = [pos[0] for pos in max_coords]
    y = [pos[1] for pos in max_coords]
    # Set the pixels at the maximum coordinates to the cluster labels
    if points.ndim==2:
        reduced_pts[x, y] = points[x, y]
    else:
        z = [pos[2] for pos in max_coords]
        reduced_pts[x, y, z] = points[x, y, z]
    return reduced_pts


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
    sk = np.pad(sk, 1)  # pad by 1 void voxel to avoid false warning
    _, N = spim.label(input=~sk.astype('bool'))
    N_shells = N - 1
    if N_shells > 0:
        logger.warning(f"{N_shells} shells were detected in the skeleton. "
                       "Trim floating solids using: "
                       "porespy.filters.trim_floating_solid()")
    return N_shells


@njit(parallel=False)
def _insert_disks_at_points_m(im, coords, radii, v, smooth=True,
                              overwrite=False):  # pragma: no cover
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
    smooth : boolean, optional
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.
    overwrite : boolean, optional
        If ``True`` then the inserted spheres overwrite existing values.  The
        default is ``False``.

    """
    p_coords = []
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        for i in range(npts):
            r = radii[i]
            s = _make_disk(r, smooth)
            pt = coords[:, i]
            if im[pt[0], pt[1]] == 0:
                p_coords.append(pt)
                for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                    if (x >= 0) and (x < xlim):
                        for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                            if (y >= 0) and (y < ylim):
                                if s[a, b] == 1:
                                    if overwrite or (im[x, y] == 0):
                                        im[x, y] = v[i]
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        for i in range(npts):
            r = radii[i]
            s = _make_ball(r, smooth)
            pt = coords[:, i]
            if im[pt[0], pt[1], pt[2]] == 0:
                p_coords.append(pt)
                for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                    if (x >= 0) and (x < xlim):
                        for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                            if (y >= 0) and (y < ylim):
                                for c, z in enumerate(range(pt[2]-r, pt[2]+r+1)):
                                    if (z >= 0) and (z < zlim):
                                        if s[a, b, c] == 1:
                                            if overwrite or (im[x, y, z] == 0):
                                                im[x, y, z] = v[i]
    return im, p_coords


if __name__ == "__main__":
    '''
    Simulation using MAGNET extraction

    '''
    import matplotlib.pyplot as plt
    import porespy as ps
    import openpnm as op
    np.random.seed(10)

    twod = False
    export = False
    res = 1e-6

    # Define 2D image
    im2d = ps.generators.blobs([100, 100], porosity=0.6, blobiness=2)
    im2d = ps.filters.fill_blind_pores(im2d, conn=8, surface=True)

    # Define 3D image
    im3d = ps.generators.blobs([100, 100, 100], porosity=0.25, blobiness=1)
    im3d = ps.filters.fill_blind_pores(im3d, conn=26, surface=True)
    im3d = ps.filters.trim_floating_solid(im3d, conn=6, surface=False)

    if twod:
        im = im2d
    else:
        im = im3d

    print(f'porosity: {np.sum(im)/np.product(im.shape)*100}')

    # MAGNET
    net, sk = magnet(im, endpoints=True, voxel_size=res, l_max=7, boundary_width=3)

    # import network to openpnm
    net = op.io.network_from_porespy(net)
    net['pore.diameter'] = net['pore.radius']*2
    net['throat.diameter'] = net['throat.radius']*2

    # check network health
    h = op.utils.check_network_health(net)
    op.topotools.trim(net, pores=np.append(h['disconnected_pores'], h['isolated_pores']))
    h = op.utils.check_network_health(net)
    print(h)

    # visualize MAGNET network if 2d
    if twod:
        plt.figure(1)
        fig, ax = plt.subplots(figsize=[5, 5]);
        slice_m = im.T
        ax.imshow(slice_m, cmap=plt.cm.bone)
        op.visualization.plot_coordinates(ax=fig,
                                          network=net,
                                          size_by=net["pore.diameter"],
                                          color_by=net["pore.ymax"],
                                          markersize=200)
        op.visualization.plot_connections(network=net, ax=fig)
        ax.axis("off");
        print('Visualization Complete')

    # add geometry models
    geo = op.models.collections.geometry.cubes_and_cuboids.copy()
    del geo['pore.diameter'], geo['throat.diameter']
    net['pore.diameter'] = net['pore.diameter'].copy()
    net['throat.diameter'] = net['throat.diameter'].copy()
    net.add_model_collection(geo)
    net.regenerate_models()

    # add phase
    phase = op.phase.Phase(network=net)
    phase['pore.viscosity'] = 1e-3

    # physics
    phys = op.models.collections.physics.basic.copy()
    phase.add_model_collection(phys)
    phase.regenerate_models()

    # stokes flow simulation to estimate permeability
    Pin = 5e-6
    Pout = 0
    A = (im.shape[0]*im.shape[1]) * res**2
    L = im.shape[1] * res
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

    if im.ndim == 3:
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

    # number of pore vs. skeleton clusters in network
    from scipy.sparse import csgraph as csg
    am = net.create_adjacency_matrix(fmt='coo', triu=True)
    N, cluster_num = csg.connected_components(am, directed=False)
    print('Pore clusters:', N)
    # number of skeleton pieces
    b = square(3) if im.ndim == 2 else cube(3)
    _, N = spim.label(input=sk.astype('bool'), structure=b)
    print('Skeleton clusters:', N)

    # export
    if export:
        ps.io.to_stl(~sk, 'sk')
        ps.io.to_stl(im, 'im')
        net['pore.coords'] += 10*res
        proj = net.project
        op.io.project_to_xdmf(proj, filename='network')
