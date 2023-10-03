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


def find_throat_points(im, sk, l_max=5):
    dt = edt(im, parallel=8)
    dt = spim.gaussian_filter(dt, sigma=0.5)
    skdt = sk*dt
    b = square(l_max) if sk.ndim == 2 else cube(l_max)
    mx = (spim.maximum_filter(skdt, footprint=b) == skdt) * sk
    return mx


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
    r"""
    Inserts spheres of variable size
    """
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


im = ps.generators.blobs([400, 400], porosity=0.7, blobiness=3, seed=0)
# Obtain skeleton with proper handling of the boundaries
sk = skeletonize_magnet(im)
# Extract junctions and endpoints from skeleton
jct = analyze_skeleton(sk)
# Identify pieces of skeleton smaller than threshold and merge junctions
labels = spim.label(sk*~jct.juncs, structure=ps.tools.ps_rect(3, im.ndim))[0]
sizes = region_size(labels + 1)  # Add 1 so void has label for next line
jct.juncs += (sizes <= 5)  # Void size will be huge so will not be found by <= test
# Dilate juctions and add endpoints
juncs = draw_spheres(jct.juncs, sizes=2)
juncs += jct.endpts
# Label pore blobs (dilated junctions) and throats segments
pores = spim.label(juncs)[0]
throats = spim.label(sk*~jct.juncs, structure=ps.tools.ps_rect(3, im.ndim))[0]
# Find overlap between pore blobs and throat segments
joints = (throats > 0)*(pores > 0)
fig, ax = plt.subplots()
ax.imshow((pores + throats)/im)

# Extract conns array
pts = np.where(joints)
P1 = np.inf*np.ones(pts[0].size)
P2 = -np.inf*np.ones(pts[0].size)
np.minimum.at(P1, throats[pts], pores[pts])
np.maximum.at(P2, throats[pts], pores[pts])
mask = np.isfinite(P1) * np.isfinite(P2)
conns = np.vstack((P1[mask], P2[mask])).T.astype(int)  # Need to -1 eventually




















