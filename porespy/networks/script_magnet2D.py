import porespy as ps
import numpy as np
import scipy.ndimage as spim
import matplotlib.pyplot as plt
import scipy.signal as spsg
from edt import edt
from skimage.morphology import skeletonize_3d
from copy import copy
from skfmm import distance
from porespy.tools import (
    _insert_disks_at_points_parallel,
    extract_subsection,
    extend_slice,
    ps_rect,
    ps_round,
)
from porespy.filters import (
    region_size,
    fill_blind_pores,
    trim_floating_solid,
    flood_func,
)
from _magnet import analyze_skeleton


cm = copy(plt.cm.turbo)
cm.set_under('darkgrey')


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
        im = fill_blind_pores(im, conn=8, surface=True)
        shape = np.array(im.shape)
        im = np.pad(im, pad_width=pw, mode='edge')
        im = np.pad(im, pad_width=shape, mode='symmetric')
        sk = skeletonize_3d(im) > 0
        sk = extract_subsection(sk, shape)
        return sk
    else:
        shape = np.array(im.shape)  # Save for later
        # Tidy-up image so skeleton is clean
        im2 = fill_blind_pores(im, conn=26, surface=True)
        im2 = trim_floating_solid(im2, conn=6)
        # Add one layer to outside where holes will be defined
        im2 = np.pad(im2, 1, mode='edge')
        # This is needed for later since numpy is getting harder and harder to
        # deal with using indexing
        inds = np.arange(im2.size).reshape(im2.shape)
        strel = ps_rect(w=5, ndim=2)  # This defines the hole size
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
            imtmp = im2[tuple(s)].squeeze()  # Extract 2D face
            sk2d = skeletonize_3d(imtmp) > 0  # Skeletonize it
            juncs, ends = analyze_skeleton(sk2d)  # Extract juncs and ends
            # This function merges junctions, defines throats, and labels them
            dt = edt(imtmp)
            pores, throats = partition_skeleton(sk2d, juncs + ends, dt)
            # This is a new function for finding 'throat nodes'
            new_juncs = find_throat_junctions(
                im=imtmp, pores=pores, throats=throats, dt=dt)
            # Dilate junctions and endpoints to create larger 'thru-holes'
            juncs_dil = spim.binary_dilation((pores > 0) + new_juncs, strel)
            # Insert image of holes onto corresponding face of im2
            np.put(im2, inds[tuple(s)].flatten(), juncs_dil.flatten())
        # Extend the faces to convert holes into tunnels
        im2 = np.pad(im2, 20, mode='edge')
        # Perform skeletonization
        sk = skeletonize_3d(im2) > 0
        # Extract the original 'center' of the image prior to padding
        sk = extract_subsection(sk, shape)
        return sk


def find_throat_junctions(im, pores, throats, dt=None):
    r"""
    Finds local peaks on the throat segments of a skeleton large enough to be
    considered junctions.

    Parameters
    ----------
    im : ndarray
        A boolean array with `True` values indicating the void phase (or phase
        of interest).
    pores : ndarray
        An ndarray the same shape as `im` with clusters of pore voxels
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

    Returns
    -------
    juncs : ndarray
        A boolean array with `True` values indicating voxels on the `throat`
        segments which are actually junctions.
    """
    # Parse input args
    if dt is None:
        dt = edt(im)
    strel = ps_rect(3, ndim=pores.ndim)
    if pores.dtype == bool:
        pores = spim.label(pores > 0, structure=strel)[0]
    if throats.dtype == bool:
        throats = spim.label(throats > 0, structure=strel)[0]
    new_pores = np.zeros_like(pores, dtype=bool)
    slices = spim.find_objects(throats)
    for i, s in enumerate(slices):
        sx = extend_slice(s, pores.shape, pad=1)
        im_sub = throats[sx] == (i + 1)
        # Get starting point for fmm as pore with highest index number
        # fmm requires full connectivity so must dilate im_sub
        phi = spim.binary_dilation(im_sub, structure=strel)
        tmp = pores[sx]*phi
        start = np.where(tmp == tmp.max())
        # Convert to masked array to confine fmm to throat segment
        phi = np.ma.array(phi, mask=phi == 0)
        phi[start] = 0
        dist = np.array(distance(phi))*im_sub  # Convert from masked to ndarray
        # Obtain indices into segment
        ind = np.argsort(dist[im_sub])
        # Analyze dt profile to find significant peaks
        line_profile = dt[sx][im_sub][ind]
        pk = spsg.find_peaks(line_profile, prominence=1, distance=line_profile.min())
        # Add peak(s) to new_pores image
        hits = dist[im_sub][ind][pk[0]]
        for d in hits:
            new_pores[sx] += (dist == d)
    return new_pores


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
    dil = spim.binary_dilation(pores > 0, structure=ps_rect(w=3, ndim=im.ndim))
    pores = ps.filters.flood_func(pores, np.amax, spim.label(dil)[0]).astype(int)
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
    for i, s in enumerate(slices):
        im_sub = throats[s] == (i + 1)
        Rs = dt[s][im_sub]
        # Tradii[i] = np.median(Rs)
        Tradii[i] = np.amin(Rs)
    # Now do pores
    Pradii = -np.ones(pores.max())
    index = -np.ones(pores.max(), dtype=int)
    im_ind = np.arange(0, im.size).reshape(im.shape)
    slices = spim.find_objects(pores)
    for i, s in enumerate(slices):
        Pradii[i] = dt[s].max()
        index[i] = im_ind[s][dt[s] == Pradii[i]][0]
    coords = np.vstack(np.unravel_index(index, im.shape)).T
    if dt.ndim == 2:
        coords = np.vstack(
            (coords[:, 0], coords[:, 1], np.zeros_like(coords[:, 0]))).T
    d = {}
    d['pore.coords'] = coords
    d['throat.conns'] = conns
    d['throat.diameter'] = 2*Tradii
    d['pore.diameter'] = 2*Pradii
    return d


# %%
im = ps.generators.blobs([200, 200], porosity=0.7, blobiness=2, seed=0)
dt = edt(im)
sk = skeletonize_magnet2(im)
juncs, ends = analyze_skeleton(sk)
pores, throats = partition_skeleton(sk, juncs + ends, dt)
new_juncs = find_throat_junctions(im=im, pores=pores, throats=throats, dt=dt)
pores, throats = partition_skeleton(sk, juncs + new_juncs + ends, dt=dt)
net = sk_to_network(pores, throats, dt)


# %%
import openpnm as op
pn = op.network.Network()
pn.update(net)


# %%
h = op.visualization.plot_connections(
    network=pn,
    size_by=pn['throat.diameter'],
    color_by=pn.Ts,
    linewidth=50,
    cmap=cm,
)
h = op.visualization.plot_coordinates(
    network=pn,
    size_by=pn['pore.diameter']**2,
    color_by=pn['pore.diameter'],
    alpha=0.85,
    linewidth=3,
    edgecolor='k',
    markersize=8000,
    ax=h,
    zorder=2,
)
fig, ax = plt.gcf(), plt.gca()
ax.imshow(((pores + throats)/im).T, cmap=cm, vmin=0.01, interpolation='none')
fig.set_size_inches([12, 12])
# plt.savefig('magnet_with_throats.png', bbox_inches='tight')












