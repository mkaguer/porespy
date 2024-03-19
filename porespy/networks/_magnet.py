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
        ftj = find_throat_junctions(im, sk, juncs, throats, dt, l_max, mode)
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
                          sk,
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
        temp = temp + dt * sk  # FIXME: should sk be an argument?
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
    # FIXME: clean up returned diameters
    t_inscribed_diameter = np.zeros((Nt), dtype=float)
    t_max_diameter = np.zeros((Nt), dtype=float)
    t_min_diameter = np.zeros((Nt), dtype=float)
    t_avg_diameter = np.zeros((Nt), dtype=float)
    t_equivalent_diameter = np.zeros((Nt), dtype=float)
    t_length = np.zeros((Nt), dtype=float)
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
        t_inscribed_diameter[throat] = (np.average(throat_dt[throat_dt != 0]**4))**(1/4)*2
        radii = throat_dt[throat_dt != 0]
        F_approx = sum(1/(2*radii)**4)
        t_equivalent_diameter[throat] = (len(radii)/F_approx)**(1/4)
        t_min_diameter[throat] = np.min(throat_dt[throat_dt != 0])*2
        t_max_diameter[throat] = np.max(throat_dt[throat_dt != 0])*2
        t_avg_diameter[throat] = np.average(throat_dt[throat_dt != 0])*2
        t_length[throat] = len(throat_dt[throat_dt != 0])
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
    p_diameter = p_radius * 2
    # clipped diameters
    V_p = p_diameter**3  # volume of a cube!
    p_diameter_clipped = (6*V_p/np.pi)**(1/3)
    V_t_max = t_max_diameter**2*t_length
    t_max_diameter_clipped = (4*V_t_max/np.pi/t_length)**(1/2)
    V_t_min = t_min_diameter**2*t_length
    t_min_diameter_clipped = (4*V_t_min/np.pi/t_length)**(1/2)
    V_t_avg = t_avg_diameter**2*t_length
    t_avg_diameter_clipped = (4*V_t_avg/np.pi/t_length)**(1/2)
    V_t_inscribed = t_inscribed_diameter**2*t_length
    t_inscribed_diameter_clipped = (4*V_t_inscribed/np.pi/t_length)**(1/2)
    V_t_equivalent = t_equivalent_diameter**2*t_length
    t_equivalent_diameter_clipped = (4*V_t_equivalent/np.pi/t_length)**(1/2)
    # create network dictionary
    net = {}
    net['throat.conns'] = t_conns
    net['pore.coords'] = p_coords * voxel_size
    net['throat.actual_length'] = t_length * voxel_size
    net['throat.inscribed_diameter'] = t_inscribed_diameter * voxel_size
    net['throat.equivalent_diameter'] = t_equivalent_diameter * voxel_size
    net['throat.max_diameter'] = t_max_diameter * voxel_size
    net['throat.min_diameter'] = t_min_diameter * voxel_size
    net['throat.avg_diameter'] = t_avg_diameter * voxel_size
    net['throat.inscribed_diameter_clipped'] = t_inscribed_diameter_clipped * voxel_size
    net['throat.equivalent_diameter_clipped'] = t_equivalent_diameter_clipped * voxel_size
    net['throat.max_diameter_clipped'] = t_max_diameter_clipped * voxel_size
    net['throat.min_diameter_clipped'] = t_min_diameter_clipped * voxel_size
    net['throat.avg_diameter_clipped'] = t_avg_diameter_clipped * voxel_size
    net['pore.inscribed_diameter'] = p_diameter  * voxel_size
    net['pore.clipped_diameter'] = p_diameter_clipped * voxel_size
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


def _get_normal(sk, throats):
    r"""
    This function returns the normal along each point in the skeleton that is
    also in 'throats'. Throats can be clusters of throat voxels or a single
    voxel along each throat; wherever the user would like the normal to be
    measured.

    Parameters
    ----------
    sk : ndarray
        The skeleton of an image
    throats : ndarray
        An ndarray the same shape as `im` with clusters of throat voxels
        uniquely labelled (1...Nt). If a boolean array is provided then a
        cluster labeling is performed with full cubic connectivity.

    Returns
    -------
    n : ndarray
        The unit normal vector at each non-zero voxel in throats
    coords : ndarray
        The coordinates at which each unit normal is found, sorted by label
        in throats
    """
    # label throats if not already labelled
    strel = ps_rect(3, ndim=sk.ndim)
    if throats.dtype == bool:
        throats = spim.label(throats > 0, structure=strel)[0]
    n_throat_nodes = np.sum(throats > 0)
    # find neighbour voxels on sk
    neighbour_coords = np.zeros((n_throat_nodes, 2, sk.ndim))
    sk_pad = np.pad(sk, pad_width=1)  # pad skeleton to handle edges
    # get coordinates, sorted by throat label
    labels = throats[np.where(throats > 0)]
    sort = np.argsort(labels)
    coord = np.array(np.where(throats > 0)).T
    coord = coord[sort]
    coord += 1  # add 1 because of padding
    # the following dilation assumes each node has two neighbours
    # mask is False when 1st neighbour is found
    mask = np.ones(n_throat_nodes, dtype=bool)
    if sk.ndim == 2:
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == 0 and j == 0:
                    pass
                else:
                    x = coord[:, 0] + i
                    y = coord[:, 1] + j
                    coords = np.array([x-1, y-1]).T
                    is_1st = sk_pad[x, y] * mask  # first neighbour
                    neighbour_coords[:, 0, :][is_1st] = coords[is_1st]
                    is_2nd = sk_pad[x, y] * (~mask)  # second neighbour
                    neighbour_coords[:, 1, :][is_2nd] = coords[is_2nd]
                    # update mask
                    mask[is_1st] = False
    if sk.ndim == 3:
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if i == 0 and j == 0 and k == 0:
                        pass
                    else:
                        x = coord[:, 0] + i
                        y = coord[:, 1] + j
                        z = coord[:, 2] + k
                        coords = np.array([x-1, y-1, z-1]).T
                        is_1st = sk_pad[x, y, z] * mask  # first neighbour
                        neighbour_coords[:, 0, :][is_1st] = coords[is_1st]
                        is_2nd = sk_pad[x, y, z] * (~mask)  # second neighbour
                        neighbour_coords[:, 1, :][is_2nd] = coords[is_2nd]
                        # update mask
                        mask[is_1st] = False
    # calculate normal from neighbour coords!
    n = np.diff(neighbour_coords, axis=1).reshape((n_throat_nodes, sk.ndim))
    # make normal a unit normal
    norm = np.linalg.norm(n, axis=1).reshape(n_throat_nodes, 1)
    n = n/norm
    coord -= 1

    return n, coord


def _cartesian_to_spherical(n):
    r"""
    This function converts from cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    n : ndarray, N by ndim
        The unit vector in cartesian coordinates x, y, and z
        (if three dimensions)

    Returns
    -------
    angles : N by 2
        The theta and phi of the unit vector where theta is measured from the
        positive x-axis between 0 and 2pi and phi is measurd from the positive
        z-axis between 0 and pi. Theta is in the first column and Phi is in the
        second column.
    """
    # calculate theta and phi of normal
    x = n[:, 0]
    y = n[:, 1]
    theta = np.arctan2(y, x)  # arctan2 is from 0 to pi
    theta[theta<0] += 2*np.pi  # make range from 0 to 2*pi
    if n.shape[1] == 3:
        z = n[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arccos(z/r)  # arccos is from 0 to pi
    else:
        phi = np.pi/2  # if 2D assume z = 0 plane
    angles = np.zeros((n.shape[0], 2))
    angles[:, 0] = theta
    angles[:, 1] = phi

    return angles


def walk(im,
         normals,
         coords,
         n_walkers,
         step_size=0.5,
         max_n_steps=None):
    r"""
    This function converts from cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    im : ndarray (boolean)
        Image of porous material where True indicates the void phase and False
        indicates solid
    normals : ndarray, N by im.ndim
        The unit vectors that are perpendicular to the direction that the
        walkers must take. The length of this array must equal the length of
        coords.
    coords : ndarray, N by im.ndim
        The coordinates from which the walkers are to start walking from. These
        coordinates correspond with the normals and as such the length of this
        array must equal the length of normals.
    n_walkers: int (default=10)
        The number of walkers to start from each voxel in throats. If 2D image
        is passed then 2 walkers is used automatically since there are only
        two directions perpendicular to normal in 2D.
    step_size : float
        The size of each step to take. This is equivalent to r in spherical
        coordinates.
    max_n_steps : int (optional)
        The maximum number of steps to take. The default is None, in which case
        there is no maximum and the walk will stop when ALL walkers have
        reached solid.

    Returns
    -------
    path: ndarray N_steps by N_walkers by 6
        The array used to keep track of the positions or path that each walker
        takes is returned. Upon each step, this array is incremented by one in
        the first axis. The second axis corresponds to the number of total
        walkers while the third axis contains labels, x-coord, y-coord,
        z-coord, theta, and phi in that order. The labels given to each walker
        are asigned acoording to their starting coordinate. For example, a
        label 1 is assigned to all walkers starting at the first coordinate
        given in coords.
        
    """
    # parse arguments
    if max_n_steps is None:
        max_n_steps = np.inf
    if im.ndim == 2:
        n_walkers = 2  # overwrite n_walkers to two if 2D
    # build 'path' array for walkers
    n_throat_nodes = coords.shape[0]
    path = np.zeros((1, n_throat_nodes, 6))  # label, x, y, z, theta, phi
    path[0, :, 0] = np.arange(1, n_throat_nodes+1)
    path[0, :, 1:(im.ndim+1)] = coords
    # duplicate path by n_walkers along second axis
    path = np.tile(path, (1, n_walkers, 1))
    # get theta and phi of normals
    angles = _cartesian_to_spherical(normals)
    thetan = angles[:, 0]  # theta of normal
    phin = angles[:, 1]  # phi of normal
    for w in range(n_walkers):
        if im.ndim == 2:
            # In 2d we have two walkers: (theta + pi/2) and (theta - pi/2)
            theta = angles[:, 0] + (-1)**w*np.pi/2
            phi = angles[:, 1]
        if im.ndim == 3:
            theta = w/n_walkers*2*np.pi
            # take dot product with normal to get phi2
            phi = np.arctan(-1/np.tan(phin)/np.cos(thetan-theta))
            phi[phin == 0] = np.pi/2  # b/c tan(0) is zero
            phi[phi < 0] += np.pi  # fix so phi is from 0 to pi
        # add theta and phi to path for walker w
        path[0, n_throat_nodes*w:n_throat_nodes*(w+1), 4] = theta
        path[0, n_throat_nodes*w:n_throat_nodes*(w+1), 5] = phi
    # start walk
    i = 0  # initialize count of walkers reaching solid
    r = step_size
    step = 1
    # pad image
    im_pad = np.pad(im, pad_width=1)
    # get total no of walkers
    n_walkers_total = n_throat_nodes * n_walkers
    while i < n_walkers_total:
        # retrieve old coords
        x_old = path[step-1, :, 1]
        y_old = path[step-1, :, 2]
        z_old = path[step-1, :, 3]
        # check if void, add one for padding
        x = np.round(x_old+1).astype(int)  # round and take integer
        y = np.round(y_old+1).astype(int)
        z = np.round(z_old+1).astype(int)
        if im.ndim == 2:
            is_void = im_pad[x, y]
        else:
            is_void = im_pad[x, y, z]
        # calculate step in each direction
        delta_x = r*np.sin(path[step-1, :, 5])*np.cos(path[step-1, :, 4])
        delta_y = r*np.sin(path[step-1, :, 5])*np.sin(path[step-1, :, 4])
        delta_z = r*np.cos(path[step-1, :, 5])
        # calculate new coords
        x_new = delta_x + x_old
        y_new = delta_y + y_old
        z_new = delta_z + z_old
        # create a new row in rw...
        new_step = np.zeros_like(path[step-1:, :, :])
        new_step[0, :, :] = path[step-1, :, :].copy()
        new_step[0, :, 1][is_void] = x_new[is_void]
        new_step[0, :, 2][is_void] = y_new[is_void]
        new_step[0, :, 3][is_void] = z_new[is_void]
        path = np.vstack((path, new_step))
        if step == max_n_steps:
            print('Maximum number of steps reached, ending walk')
            break
        # update step counter
        step += 1
        # update number of walkers that have reached solid
        i = np.sum(~is_void)

    return path


def get_throat_area(im,
                    sk,
                    throats,
                    n_walkers=10,
                    step_size=0.5,
                    max_n_steps=None):
    r"""
    This function returns the cross-sectional acrea of throats. The 

    Parameters
    ----------
    im : ndarray (boolean)
        Image of porous material where True indicates the void phase and False
        indicates solid
    sk : ndarray
        The skeleton of an image
    throats : ndarray
        An ndarray the same shape as `im` with clusters of throat voxels
        uniquely labelled (1...Nt). If a boolean array is provided then a
        cluster labeling is performed with full cubic connectivity. Walkers are
        sent from each voxel in throats. Therefore, reducing clusters of throat
        voxels beforehand could help reduce computation time but consequently
        will reduce the amount of imformation available. It is important that
        each voxel specified in this image has exactly two neighbours on the
        skeleton!
    n_walkers: int (default=10)
        The number of walkers to start from each voxel in throats. If 2D image
        is passed then 2 walkers is used automatically since there are only
        two directions perpendicular to normal in 2D.
    step_size : float (default=0.5)
        The size of each step to take. This is equivalent to r in spherical
        coordinates.
    max_n_steps : int (optional)
        The maximum number of steps to take. The default is None, in which case
        there is no maximum and the walk will stop when ALL walkers have
        reached solid.

    Returns
    -------
    throat_area : ndarray
        The original throats image with labelled voxels overwritten by their
        measured throat areas.

    """
    # parse arguments
    if max_n_steps is None:
        max_n_steps = np.inf
    if im.ndim == 2:
        n_walkers = 2  # overwrite n_walkers to two if 2D
    # label throats if not already labelled
    strel = ps_rect(3, ndim=im.ndim)
    if throats.dtype == bool:
        throats = spim.label(throats > 0, structure=strel)[0]
    # get normals
    normals, coords = _get_normal(sk, throats)
    # perform walk
    path = walk(im, normals, coords, n_walkers, step_size, max_n_steps)
    # sort array by throat label again
    I = np.argsort(path[0, :, 0], kind='stable')
    path = path[:, I, :]
    # calculate throat area
    throat_area = np.zeros_like(throats, dtype=float)
    # number of throats
    n_throat_nodes = np.sum(throats > 0)
    for n in range(n_throat_nodes):
        coord1 = path[-1, n*n_walkers:(n+1)*n_walkers, 1:4]
        coord2 = path[0, n*n_walkers:(n+1)*n_walkers, 1:4]
        r = np.sum((coord1 - coord2)**2, axis=1)**(1/2)
        if im.ndim==2:
            area = r[0] + r[1]  # cross-section length is area in 2D
            x, y, z = coord2[0].astype(int)
            throat_area[x, y] = area
        else:
            r1 = r
            r2 = np.roll(r1, 1)
            angle = 2*np.pi/n_walkers
            area = np.sum(r1*r2*np.sin(angle)/2)
            x, y, z = coord2[0].astype(int)
            throat_area[x, y, z] = area
    
    return throat_area


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
    '''
    # MAGNET Steps
    net, sk = magnet(im,
                     sk=None,
                     parallel=False,
                     surface=False,
                     voxel_size=1,
                     l_max=7,
                     throat_junctions="fast marching")
    '''
    # take the skeleton
    sk, im = skeleton(im, surface=False, parallel=False)
    # take distance transform
    dt = edt(im)
    # find junctions
    fj = find_junctions(sk)
    juncs = fj.juncs + fj.endpts
    juncs = merge_nearby_juncs(sk, juncs, dt)
    # find throats
    throats = (~juncs) * sk
    # find throat junctions
    ftj = find_throat_junctions(im, sk, juncs, throats, dt, l_max=7, mode="fast marching")
    # add throat juncs to juncs
    juncs = ftj.new_juncs.astype('bool') + juncs
    # get new throats
    throats = ftj.new_throats
    # use random walk to get throat area
    dt_inv = 1/spim.gaussian_filter(dt, sigma=0.4)
    throat_nodes = juncs_to_pore_centers(throats, dt_inv)  # find throat node at minimums, doesn't work well for endpoints!
    n_walkers=10
    throat_area = get_throat_area(im, sk, throat_nodes, n_walkers)
    print(f'Total throat area: {np.sum(throat_area)}')
    # on im3: 28609.85937008391
    # 30774.378561650945
    xx
    # get network from junctions
    net = junctions_to_network(sk, juncs, throats, dt, voxel_size=1)

    # import network to openpnm
    net = op.io.network_from_porespy(net)
    net['pore.diameter'] = net['pore.inscribed_diameter']
    net['throat.diameter'] = net['throat.avg_diameter']

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
    
    '''
    for j in range(10):
        # j = 9
        theta1 = angles[j, 0]
        phi1 = angles[j, 1]
        x1 =  np.sin(phi1)*np.cos(theta1)
        y1 = np.sin(phi1)*np.sin(theta1)
        z1 = np.cos(phi1)
        coord1 = np.array([x1, y1, z1])
        theta2 = path[0, 10*(j):10*(j+1), 4]
        phi2 = path[0, 10*(j):10*(j+1), 5]
        x2 =  np.sin(phi2)*np.cos(theta2)
        y2 = np.sin(phi2)*np.sin(theta2)
        z2 = np.cos(phi2)
        for i in range(10):
            coord2 = np.array([x2[i], y2[i], z2[i]])
            print(np.dot(coord1, coord2))
    '''
