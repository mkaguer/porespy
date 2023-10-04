import porespy as ps
import numpy as np
import scipy.ndimage as spim
import matplotlib.pyplot as plt
from edt import edt
from skimage.morphology import skeletonize, skeletonize_3d
from porespy.tools import _insert_disks_at_points_parallel, unpad, Results
from porespy.filters import region_size, fill_blind_pores
from skimage.morphology import square, cube
import scipy.signal as spsg
from copy import copy
from skfmm import distance


cm = copy(plt.cm.turbo)
cm.set_under('darkgrey')


def analyze_skeleton(sk):
    a = square(3) if sk.ndim == 2 else cube(3)
    conv = spsg.convolve(sk*1.0, a, mode='same', method='auto')
    conv = np.rint(conv).astype(int)  # in case of fft, accuracy is lost
    juncs = (conv >= 4) * sk
    endpts = (conv == 2) * sk
    pt = Results()
    pt.juncs = juncs
    pt.endpts = endpts
    return pt


def skeletonize_magnet(im):
    if im.ndim == 2:
        im = fill_blind_pores(im, conn=8, surface=True)
        shape = np.array(im.shape)
        im = np.pad(im, pad_width=5, mode='edge')
        im = np.pad(im, pad_width=shape, mode='symmetric')
        sk = skeletonize_3d(im) > 0
        sk = unpad(sk, shape+5)
        return sk
    else:
        raise Exception("Not implemented in 3D yet")


def draw_spheres(sites, sizes):
    coords = np.where(sites)
    if isinstance(sizes, (float, int)):
        radii = (round(sizes)*np.ones_like(coords[0])).astype(int)
    else:
        sizes = np.around(sizes, decimals=0).astype(int)
        radii = sizes[coords]
    spheres = np.zeros_like(sites)
    spheres = _insert_disks_at_points_parallel(
        im=spheres, coords=np.vstack(coords), radii=radii, v=True)
    return spheres


def find_throat_junctions(im, pores, throats, dt=None):
    if dt is None:
        dt = edt(im)
    new_pores = np.zeros_like(pores, dtype=bool)
    slices = spim.find_objects(throats)
    for i, s in enumerate(slices):
        im_sub = throats[s] == (i + 1)
        # Get starting point for fmm as pore with highest index number
        tmp = pores[s]*im_sub
        start = np.where(tmp == tmp.max())
        # fmm requires full connectivity so must dilate im_sub
        phi = spim.binary_dilation(im_sub)
        # Convert to masked array to confine fmm to throat segment
        phi = np.ma.array(phi, mask=phi == 0)
        phi[start] = 0
        dist = np.array(distance(phi))*im_sub  # Convert from masked to ndarray
        # Obtain indices into segment
        ind = np.argsort(dist[im_sub])
        # Analyze dt profile to find significant peaks
        line_profile = (im_sub*dt[s])[im_sub][ind]  # Mind F#$&
        pk = spsg.find_peaks(line_profile, prominence=1, distance=line_profile.min())
        # Add peak(s) to new_pores image
        hits = dist[im_sub][ind][pk[0]]
        for d in hits:
            new_pores[s] += (dist == d)
    return new_pores


def sk_to_pores_and_throats(sk, juncs, ends=None, t=5):
    strel = ps.tools.ps_rect(3, im.ndim)
    labels = spim.label(sk*~juncs, structure=strel)[0]
    sizes = region_size(labels + 1)  # Add 1 so void has label for next line
    juncs += (sizes <= t)  # Void size will be huge so will not be found by <= test
    throats = spim.label(sk*~juncs, structure=strel)[0]
    juncs_dil = draw_spheres(juncs > 0, sizes=2)
    tmp = juncs_dil + ends if ends is not None else juncs_dil
    pores = spim.label(tmp, structure=strel)[0]
    return pores, throats


def find_conns(pores, throats):
    joints = (throats > 0)*(pores > 0)
    pts = np.where(joints)
    P1 = np.inf*np.ones(pts[0].size)
    P2 = -np.inf*np.ones(pts[0].size)
    np.minimum.at(P1, throats[pts], pores[pts])
    np.maximum.at(P2, throats[pts], pores[pts])
    mask = np.isfinite(P1) * np.isfinite(P2)
    conns = np.vstack((P1[mask], P2[mask])).T.astype(int)  # Need to -1 eventually
    return conns


def find_coords(pores, dt):
    radii = -np.ones(pores.max())
    index = -np.ones(pores.max(), dtype=int)
    im_ind = np.arange(0, im.size).reshape(im.shape)
    slices = spim.find_objects(pores)
    for i, s in enumerate(slices):
        radii[i] = dt[s].max()
        index[i] = im_ind[s][dt[s] == radii[i]][0]
    coords = np.vstack(np.unravel_index(index, im.shape)).T
    if dt.ndim == 2:
        coords = np.vstack((coords[:, 0], coords[:, 1], np.zeros_like(coords[:, 0]))).T
    return coords, radii


# %%
im = ps.generators.blobs([600, 600], porosity=0.7, blobiness=2, seed=0)
dt = edt(im)
sk = skeletonize_magnet(im)
juncs, ends = analyze_skeleton(sk)
pores, throats = sk_to_pores_and_throats(sk, juncs, ends)
new_juncs = find_throat_junctions(im=im, pores=pores, throats=throats, dt=dt)
pores, throats = sk_to_pores_and_throats(sk, juncs + new_juncs, ends)
conns = find_conns(pores, throats)
coords, radii = find_coords(pores, dt)

# %%
import openpnm as op
pn = op.network.Network()
pn['pore.coords'] = coords
pn['throat.conns'] = conns - 1
pn['pore.diameter'] = radii*2
h = op.visualization.plot_connections(pn, color_by=pn.Ts, linewidth=5, cmap=cm)
Ps = pn.num_neighbors(pn.Ps) <= 2
h = op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter']**2, color='none', linewidth=3, edgecolor='k', markersize=1500, ax=h, zorder=2)
fig, ax = plt.gcf(), plt.gca()
throats_temp = throats * (pores == 0)
ax.imshow(((pores + throats_temp)/im).T, cmap=cm, vmin=0.01, interpolation='none')
# ax.imshow(((sk*dt)/im).T, cmap=cm, vmin=0.01, interpolation='none')












