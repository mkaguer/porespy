from porespy.filters import trim_floating_solid, flood_func, region_size
from porespy.filters._snows import _estimate_overlap
from porespy.tools import ps_rect, Results, extend_slice, make_contiguous
from porespy.tools import _insert_disks_at_points, ps_round, extract_subsection
from porespy.generators import borders
from skimage.morphology import square, cube
from scipy.ndimage import maximum_position
from skfmm import distance
from porespy import settings
from edt import edt
import skimage as ski
import dask.array as da
import scipy.ndimage as spim
import scipy.signal as spsg
import logging
import numpy as np
from porespy.tools import get_tqdm

tqdm = get_tqdm()
logger = logging.getLogger(__name__)


def magnet(im,
           sk=None,
           parallel=False,
           surface=False,
           voxel_size=1,
           l_max=7,
           throat_junctions=None,
           **kwargs):
    r"""
    Perform a Medial Axis Guided Network ExtracTion (MAGNET) on an image of
    porous media.

    This is a modernized python implementation of a classical
    network extraction method. First, the skeleton of the provided image is
    determined. The skeleton can be computed in serial or parallel modes.
    Next, all the junction points of the skeleton are determined by using
    convolution including terminal points on the ends of branches. ClustersPores are
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
    surface : boolean
        If `False` disconnected solid at the surface of the image is NOT
        trimmed. This is the default mode. However, if `True`, disconnected
        solid at the surface of the image is trimmed. This is NOT applied when
        im is 2d.
    parallel : boolean
        If `False` the skeleton is calculated in serial. This is the default
        mode. However, if `True`, the skeleton is calculated in parallel using
        chunking in Dask. The default mode is `False`
    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be voxel_size-cubed
    l_max : scalar (default = 7)
        The size of the maximum filter used in finding junction along long
        throats. This argument is only used when throat_junctions is set to
        "maximum filter" mode.
    throat_junctions : str
        The mode to use when finding throat junctions. The options are "maximum
        filter" or "fast marching". If None is given, then throat junctions are
        not found (this is the default).

    Returns
    -------
    net : dict
        A dictionary containing the most important pore and throat size data
        and topological data.
    sk : ndarray
        The skeleton of the image is also returned.
    """
    # get the skeleton
    if sk is None:
        sk, im = skeleton(im, surface, parallel, **kwargs)  # take skeleton
    else:
        if im.ndim == 3:
            _check_skeleton_health(sk.astype('bool'))
    # take distance transform
    dt = edt(im)
    # find junctions
    fj = find_junctions(sk)
    juncs = fj.juncs + fj.endpts
    juncs = merge_nearby_juncs(sk, juncs, dt)  # FIXME: merge juncs AND endpts?
    # find throats
    throats = (~juncs) * sk
    # find throat junctions
    if throat_junctions is not None:
        mode = throat_junctions
        ftj = find_throat_junctions(im, juncs, throats, dt, l_max, mode)
        # add throat juncs to juncs
        juncs = ftj.new_juncs.astype('bool') + juncs
        # get new throats
        throats = ftj.new_throats
    # get network from junctions
    net = junctions_to_network(sk, juncs, throats, dt, voxel_size)
    return net, sk


def skeleton(im, surface=False, parallel=False, **kwargs):
    r"""
    Takes the skeleton of an image. This function ensures that no shells are
    found in the resulting skeleton by trimming floating solids from the image
    beforehand and by checking for shells after taking the skeleton. The
    skeleton is taken using Lee's method as available in scikit-image. For
    faster skeletonization, a parallel mode is available.

    Parameters
    ----------
    im : ndarray
        A binary image of porous media with 'True' values indicating phase of
        interest.
    surface : boolean
        If `False` disconnected solid at the surface of the image is NOT
        trimmed. This is the default mode. However, if `True`, disconnected
        solid at the surface of the image is trimmed. Note that disconnected
        solids are NOT removed if a 2D image is passed.
    parallel : boolean
        If `False` the skeleton is calculated in serial. This is the default
        mode. However, if `True`, the skeleton is calculated in parallel using
        chunking in Dask.

    Returns
    -------
    sk : ndarray
        Skeleton of image
    im : ndarray
        The image used to take the skeleton, the same as the input image except
        for floating solids removed if the image supplied is 3D
    """
    # trim floating solid from 3D images
    if im.ndim == 3:
        im = trim_floating_solid(im, conn=6, surface=surface)
    # perform skeleton
    if parallel is False:  # serial
        sk = ski.morphology.skeletonize_3d(im).astype('bool')
    if parallel is True:  # parallel
        sk = skeleton_parallel(im, **kwargs)
    if im.ndim == 3:
        _check_skeleton_health(sk.astype('bool'))
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


def find_junctions(sk):
    r"""
    Finds all junctions and endpoints in a skeleton.

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


def find_throat_junctions(im,
                          juncs,
                          throats,
                          dt=None,
                          l_max=7,
                          mode="fast marching"):
    r"""
    Finds local peaks on the throat segments of a skeleton large enough to be
    considered junctions.

    Parameters
    ----------
    im : ndarray
        A boolean array with `True` values indicating the void phase (or phase
        of interest).
    juncs : ndarray
        An ndarray the same shape as `im` with clusters of junction voxels
        uniquely labelled (1...Np).  If a boolean array is provided then a
        cluster labeling is performed with full cubic connectivity.
    throats : ndarray
        An ndarray the same shape as `im` with clusters of throat voxels
        uniquely labelled (1...Nt). If a boolean array is provided then a
        cluster labeling is performed with full cubic connectivity.
    dt : ndarray (optional)
        The distance transform of the image. This is used to find local peaks
        on the segments defined by `throats`. If these local peaks are sufficiently
        high and spaced apart from each other they are considered throat junctions.
        If not provided it will be computed from `im`.
    l_max: int
        The length of the cubical structuring element to use in the maximum
        filter, if that mode is specified.
    mode : string {'maximum filter' | 'fast marching' }
        Specifies how to find throat junctions.

    Returns
    -------
    results : dataclass
        A dataclass-like object with the following named attributes:

        =============== =============================================================
        Atribute        Description
        =============== =============================================================
        new_juncs       The newly identified junctions on long throat segments. These
                        are labelled starting from the 1+ the maximum in `pores`
        juncs           The original juncs image with labels applied (if original
                        `pores` image was given as a `bool` array.
        new_throats     The new throat segments after dividing them at the newly
                        found junction locations.
        =============== =============================================================
    """
    # Parse input args
    if dt is None:
        dt = edt(im)
    strel = ps_rect(3, ndim=juncs.ndim)
    if juncs.dtype == bool:
        juncs = spim.label(juncs > 0, structure=strel)[0]
    if throats.dtype == bool:
        throats = spim.label(throats > 0, structure=strel)[0]
    if mode == "maximum filter":
        # reduce clusters to pore centers
        ct = juncs_to_pore_centers(juncs, dt)
        # find coords of pore centers and radii
        coords = np.vstack(np.where(ct)).astype(int)
        radii = dt[np.where(ct)].astype(int)
        # insert spheres
        Ps = np.zeros_like(ct, dtype=int)
        Ps = _insert_disks_at_points(Ps, coords, radii, v=1)
        # Find maximums along long throats
        temp = Ps * np.inf
        mask = np.isnan(temp)
        temp[mask] = 0
        temp = temp + dt * sk
        b = square(l_max) if ct.ndim == 2 else cube(l_max)
        mx = (spim.maximum_filter(temp, footprint=b) == dt) * sk
        mx = juncs_to_pore_centers(mx, dt)
        # remove maximum points that lie on junction cluster!
        mx[juncs > 0] = 0
        mx = make_contiguous(mx)  # make contiguous again
        # set new_juncs equal to mx
        new_juncs = mx
    if mode == "fast marching":
        new_juncs = np.zeros_like(juncs, dtype=bool)
        slices = spim.find_objects(throats)
        for i, s in enumerate(tqdm(slices)):
            sx = extend_slice(s, juncs.shape, pad=1)
            im_sub = throats[sx] == (i + 1)
            # Get starting point for fmm as pore with highest index number
            # fmm requires full connectivity so must dilate im_sub
            phi = spim.binary_dilation(im_sub, structure=strel)
            tmp = juncs[sx]*phi
            start = np.where(tmp == tmp.max())
            # Convert to masked array to confine fmm to throat segment
            phi = np.ma.array(phi, mask=phi == 0)
            phi[start] = 0
            dist = np.array(distance(phi))*im_sub  # Convert from masked to ndarray
            # Obtain indices into segment
            ind = np.argsort(dist[im_sub])
            # Analyze dt profile to find significant peaks
            line_profile = dt[sx][im_sub][ind]
            pk = spsg.find_peaks(
                line_profile,
                prominence=1,
                distance=max(1, line_profile.min()),
            )
            # Add peak(s) to new_juncs image
            hits = dist[im_sub][ind][pk[0]]
            for d in hits:
                new_juncs[sx] += (dist == d)
        # label new_juncs
        new_juncs = spim.label(new_juncs, structure=strel)[0]
    # Remove peaks from original throat image and re-label
    new_throats = spim.label(throats*(new_juncs == 0), structure=strel)[0]
    # increment new_juncs by labels in original pores
    new_juncs[new_juncs > 0] += juncs.max()
    results = Results()
    results.new_juncs = new_juncs
    results.juncs = juncs
    results.new_throats = new_throats
    return results


def merge_nearby_juncs(sk, juncs, dt=3):
    r"""
    Merges nearby junctions found in the skeleton

    Parameters
    ----------
    sk : ndarray
        A boolean image of the skeleton of the phase of interest
    juncs : ndarray
        A boolean array the same shape as `sk` with `True` values indicating
        the junction points of the skeleton.
    dt : ndarray or int, optional
        The distance transform of the phase of interest. If dt is a scalar,
        then a hard threshold is used to determine "near" junctions.

    Returns
    -------
    results : dataclass
        A `Results` object with images of `pores` and `throats` each containing
        the labelled clusters of connected voxels.
    """
    strel = ps_rect(3, sk.ndim)
    labels = spim.label(sk*~juncs, structure=strel)[0]
    sizes = region_size(labels)
    # Add voxels from skeleton to junctions if they are too close to each other
    if isinstance(dt, (int, float)):  # If dt is a scalar, use hard threshold
        juncs += (sizes <= dt)*(labels > 0)
    else:  # If dt is proper dt, threshold each voxel specifically
        # Division by root(ndim) limits range since size of cluster is not quite
        # equal to distance between end points since size does not account for
        # diagonally oriented or windy segements.
        dists = flood_func(dt, np.amin, labels=labels) / (sk.ndim)**0.5
        juncs += (sizes <= dists)*(labels > 0)

    return juncs


def juncs_to_pore_centers(juncs, dt):
    r"""
    Finds pore centers from an image of junctions. To do this, clusters of
    junction points are reduced to a single voxel, whereby the new voxel,
    corresponds to the one that has the largest distance transform value from
    within the original cluster. This method, ensures that the 'pore centre'
    lies on the original set of voxels.

    Parameters
    ----------
    juncs : ndarray
        An ndarray the same shape as `dt` with clusters of junction voxels
        uniquely labelled (1...Np).  If a boolean array is provided then a
        cluster labeling is performed with full cubic connectivity.

    dt : ndarray
        The distance transform of the original image.

    Returns
    -------
    pc : ndarray
        The resulting pore centres labelled
    """
    # cubic structuring element, full connectivity
    strel = ps_rect(3, ndim=juncs.ndim)
    if juncs.dtype == bool:
        juncs = spim.label(juncs > 0, structure=strel)[0]
    # initialize reduced juncs
    reduced_juncs = np.zeros_like(juncs, dtype=int)
    # find position of maximums by labelled cluster
    max_coords = maximum_position(dt, juncs, range(1, np.max(juncs)+1))
    # Get row and column coordinates within each cluster
    x = [pos[0] for pos in max_coords]
    y = [pos[1] for pos in max_coords]
    # Set the pixels at the maximum coordinates to the cluster labels
    if juncs.ndim==2:
        reduced_juncs[x, y] = juncs[x, y]
    else:
        z = [pos[2] for pos in max_coords]
        reduced_juncs[x, y, z] = juncs[x, y, z]
    return reduced_juncs


def junctions_to_network(sk, juncs, throats, dt, voxel_size=1):
    r"""
    Assemble a dictionary object containing essential topological and
    geometrical data for a pore network. The information is retrieved from the
    distance transform, an image of labelled junctions, and an image of
    labelled throats.

    Parameters
    ------------
    sk : ndarray
        A boolean image of the skeleton of the phase of interest
    juncs : ndarray
        An ndarray the same shape as `im` with clusters of junction voxels
        uniquely labelled (1...Np).  If a boolean array is provided then a
        cluster labeling is performed with full cubic connectivity.
    throats : ndarray
        An ndarray the same shape as `im` with clusters of throat voxels
        uniquely labelled (1...Nt). If a boolean array is provided then a
        cluster labeling is performed with full cubic connectivity.
    dt : ndarray (optional)
        The distance transform of the image.
    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.

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
    # Parse input args
    strel = ps_rect(3, ndim=juncs.ndim)
    if juncs.dtype == bool:
        juncs = spim.label(juncs > 0, structure=strel)[0]
    if throats.dtype == bool:
        throats = spim.label(throats > 0, structure=strel)[0]
    # get slicess of throats
    slices = spim.find_objects(throats)  # Nt by 2
    # initialize throat conns and radius
    Nt = len(slices)
    t_conns = np.zeros((Nt, 2), dtype=int)
    t_radius = np.zeros((Nt), dtype=float)
    # loop through throats to get t_conns and t_radius
    for throat in range(Nt):
        ss = extend_slice(slices[throat], throats.shape)
        # get slices
        sub_juncs = juncs[ss]  # sub_im_p
        sub_throats = throats[ss]  # sub_im_l
        sub_sk = sk[ss]
        sub_dt = dt[ss]
        throat_im = sub_throats == throat+1
        # dilate throat_im to capture connecting pore indices
        throat_im_dilated = spim.binary_dilation(throat_im, strel)
        throat_im_dilated = throat_im_dilated * sub_sk
        throat_im_dilated = throat_im_dilated * sub_juncs
        # throat conns
        Pn_l = np.unique(throat_im_dilated)[1:] - 1
        # FIXME: I removed if statement. Is that okay?
        t_conns[throat, :] = Pn_l
        # throat radius
        throat_dt = throat_im * sub_dt
        # FIXME: Integrate R^4 and assume square cross-section!
        t_radius[throat] = (np.average(throat_dt[throat_dt != 0]**4))**(1/4)
    # FIXME: was it okay to remove remove?
    # FIXME: did not set overlapping throat radius to min of neighbour pores
    # find pore coords
    Np = juncs.max()
    ct = juncs_to_pore_centers(juncs, dt)  # pore centres!
    p_coords = np.vstack(np.where(ct)).astype(float).T
    p_coords = np.insert(p_coords, juncs.ndim, ct[np.where(ct)], axis=1)
    p_coords = p_coords[ct[np.where(ct)].T.argsort()]
    p_coords = p_coords[:, 0:juncs.ndim]
    if p_coords.shape[1] == 2:  # If 2D, add zeros in 3rd column
        p_coords = np.hstack((p_coords, np.zeros((Np, 1))))
    # find pore radius
    p_radius = dt[np.where(ct)].reshape((Np, 1)).astype(float)
    p_radius = np.insert(p_radius, 1, ct[np.where(ct)], axis=1)
    p_radius = p_radius[ct[np.where(ct)].T.argsort()]
    p_radius = p_radius[:, 0].reshape(Np)
    # create network dictionary
    net = {}
    net['throat.conns'] = t_conns
    net['pore.coords'] = p_coords * voxel_size
    net['throat.radius'] = t_radius * voxel_size
    net['pore.radius'] = p_radius * voxel_size
    net['pore.index'] = np.arange(0, Np)
    return net


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


def skeletonize_magnet2(im):
    r"""
    Performs a skeletonization but first deals with the image boundaries correctly

    Parameters
    ----------
    im : ndarray
        The boolean image with `True` values indicating the phase of interest

    Returns
    -------
    sk : ndarray
        A boolean image with `True` values indicating the skeleton.
    """
    if im.ndim == 2:
        pw = 5
        im = ps.filters.fill_blind_pores(im, conn=8, surface=True)
        shape = np.array(im.shape)
        im = np.pad(im, pad_width=pw, mode='edge')
        im = np.pad(im, pad_width=shape, mode='symmetric')
        sk = ski.morphology.skeletonize_3d(im) > 0
        sk = extract_subsection(sk, shape)
        return sk
    else:
        shape = np.array(im.shape)  # Save for later
        dt3D = edt(im)
        # Tidy-up image so skeleton is clean
        im2 = ps.filters.fill_blind_pores(im, conn=26, surface=True)
        im2 = trim_floating_solid(im2, conn=6)
        # Add one layer to outside where holes will be defined
        im2 = np.pad(im2, 1, mode='edge')
        # This is needed for later since numpy is getting harder and harder to
        # deal with using indexing
        inds = np.arange(im2.size).reshape(im2.shape)
        # strel = ps_rect(w=1, ndim=2)  # This defines the hole size
        # Extract skeleton of each face, find junctions, and put holes on outer
        # layer of im2 at each one
        for face in [(0, 1), (0, im2.shape[0]),
                     (1, 1), (1, im2.shape[1]),
                     (2, 1), (2, im2.shape[2])]:
            s = []
            for ax in range(im2.ndim):
                if face[0] == ax:
                    s.append(slice(face[1]-1, face[1]))
                else:
                    s.append(slice(0, im2.shape[ax]))
            im_face = im[tuple(s)].squeeze()
            dt = spim.gaussian_filter(dt3D[tuple(s)].squeeze(), sigma=0.4)
            peaks = im_face*(spim.maximum_filter(dt, size=5) == dt)
            # # Dilate junctions and endpoints to create larger 'thru-holes'
            # juncs_dil = spim.binary_dilation(peaks, strel)
            # Insert image of holes onto corresponding face of im2
            np.put(im2, inds[tuple(s)].flatten(), peaks.flatten())
        # Extend the faces to convert holes into tunnels
        im2 = np.pad(im2, 20, mode='edge')
        # Perform skeletonization
        sk = ski.morphology.skeletonize_3d(im2) > 0
        # Extract the original 'center' of the image prior to padding
        sk = extract_subsection(sk, shape)
        return sk


def partition_skeleton(sk, juncs, dt):
    r"""
    Divides skeleton into pore and throat voxels given junctions

    Parameters
    ----------
    sk : ndarray
        A boolean image of the skeleton of the phase of interest
    juncs : ndarray
        A boolean array the same shape as `sk` with `True` values indicating the
        junction points at which the skeleton will be divided.
    dt : ndarray
        The distance transform of the phase of interest

    Returns
    -------
    results : dataclass
        A `Results` object with images of `pores` and `throats` each containing
        the labelled clusters of connected voxels.
    """
    strel = ps_rect(3, sk.ndim)
    labels = spim.label(sk*~juncs, structure=strel)[0]
    sizes = region_size(labels)
    # Add voxels from skeleton to junctions if they are too close to each other
    if isinstance(dt, (int, float)):  # If dt is a scalar, use hard threshold
        juncs += (sizes <= dt)*(labels > 0)
    else:  # If dt is proper dt, threshold each voxel specifically
        # Division by root(ndim) limits range since size of cluster is not quite
        # equal to distance between end points since size does not account for
        # diagonally oriented or windy segements.
        dists = flood_func(dt, np.amin, labels=labels) / (sk.ndim)**0.5
        juncs += (sizes <= dists)*(labels > 0)
    # Label the surviving pieces of skeleton as throats
    throats = spim.label(sk*~juncs, structure=strel)[0]
    pores = spim.label(juncs, structure=strel)[0]
    return pores, throats


def sk_to_network(pores, throats, dt):
    # Find conns
    dil = spim.binary_dilation(pores > 0, structure=ps_rect(w=3, ndim=pores.ndim))
    pores = flood_func(pores, np.amax, spim.label(dil)[0]).astype(int)
    joints = (throats > 0)*(pores > 0)
    pts = np.where(joints)
    P1 = np.inf*np.ones(pts[0].size)
    P2 = -np.inf*np.ones(pts[0].size)
    np.minimum.at(P1, throats[pts], pores[pts])
    np.maximum.at(P2, throats[pts], pores[pts])
    mask = np.isfinite(P1) * np.isfinite(P2)
    conns = np.vstack((P1[mask], P2[mask])).T.astype(int) - 1
    Tradii = -np.ones(conns.shape[0])
    slices = spim.find_objects(throats)
    for i, s in enumerate(tqdm(slices)):
        im_sub = throats[s] == (i + 1)
        Rs = dt[s][im_sub]
        # Tradii[i] = np.median(Rs)
        Tradii[i] = np.amin(Rs)
    # Now do pores
    Pradii = -np.ones(pores.max())
    index = -np.ones(pores.max(), dtype=int)
    im_ind = np.arange(0, dt.size).reshape(dt.shape)
    slices = spim.find_objects(pores)
    for i, s in enumerate(tqdm(slices)):
        Pradii[i] = dt[s].max()
        index[i] = im_ind[s][dt[s] == Pradii[i]][0]
    coords = np.vstack(np.unravel_index(index, dt.shape)).T
    if dt.ndim == 2:
        coords = np.vstack(
            (coords[:, 0], coords[:, 1], np.zeros_like(coords[:, 0]))).T
    d = {}
    d['pore.coords'] = coords
    d['throat.conns'] = conns
    d['throat.diameter'] = 2*Tradii
    d['pore.diameter'] = 2*Pradii
    return d


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


if __name__ == "__main__":
    '''
    Simulation using MAGNET extraction

    '''
    import matplotlib.pyplot as plt
    import porespy as ps
    import openpnm as op
    import numpy as np
    np.random.seed(10)

    # Define 2D image
    im2 = ps.generators.blobs([100, 100], porosity=0.6, blobiness=2)
    im2 = ps.filters.fill_blind_pores(im2, conn=8, surface=True)

    # Define 3D image
    im3 = ps.generators.blobs([100, 100, 100], porosity=0.25, blobiness=1)
    im3 = ps.filters.fill_blind_pores(im3, conn=26, surface=True)
    im3 = ps.filters.trim_floating_solid(im3, conn=6, surface=False)

    im = im2

    # plot
    if im.ndim == 2:
        plt.figure(1)
        plt.imshow(im)

    # MAGNET Steps
    net, sk = magnet(im,
                     sk=None,
                     parallel=False,
                     surface=False,
                     voxel_size=1,
                     l_max=7,
                     throat_junctions="fast marching")

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
    if im.ndim == 2:
        plt.figure(2)
        fig, ax = plt.subplots(figsize=[5, 5]);
        slice_m = im.T
        ax.imshow(slice_m, cmap=plt.cm.bone)
        op.visualization.plot_coordinates(ax=fig,
                                          network=net,
                                          size_by=net["pore.diameter"],
                                          color_by=net["pore.diameter"],
                                          markersize=200)
        op.visualization.plot_connections(network=net, ax=fig)
        ax.axis("off");
        print('Visualization Complete')

    # number of pore vs. skeleton clusters in network
    from scipy.sparse import csgraph as csg
    am = net.create_adjacency_matrix(fmt='coo', triu=True)
    Np, cluster_num = csg.connected_components(am, directed=False)
    print('Pore clusters:', Np)
    # number of skeleton pieces
    b = square(3) if im.ndim == 2 else cube(3)
    _, Ns = spim.label(input=sk.astype('bool'), structure=b)
    print('Skeleton clusters:', Ns)
