import porespy as ps
import numpy as np
import scipy.ndimage as spim
import matplotlib.pyplot as plt
from edt import edt
from copy import copy
from porespy.tools import (
    ps_rect,
)
from _magnet import (
    analyze_skeleton,
    skeletonize_magnet2,
    find_throat_junctions,
    partition_skeleton,
    sk_to_network,
)


cm = copy(plt.cm.turbo)
cm.set_under('darkgrey')


def sk_to_network2(pores, throats, dt):
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


# %%
im = ps.generators.blobs([200, 200], porosity=0.7, blobiness=2, seed=0)
dt = edt(im)
sk = skeletonize_magnet2(im)
juncs, ends = analyze_skeleton(sk)
pores, throats = partition_skeleton(sk, juncs + ends, dt)
new_juncs, pores, new_throats = find_throat_junctions(im=im, pores=pores, throats=throats, dt=dt)
pores, throats = partition_skeleton(sk, (juncs + new_juncs + ends) > 0, dt=dt)
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












