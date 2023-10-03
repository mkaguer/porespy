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
    sizes = np.around(sizes, decimals=0).astype(int)
    radii = sizes[coords]
    spheres = np.zeros_like(sites)
    spheres = _insert_disks_at_points_parallel(
        im=spheres, coords=np.vstack(coords), radii=radii, v=True)
    return spheres


im = ps.generators.blobs([800, 800], porosity=0.7, blobiness=3)
# im = ~ps.generators.random_spheres([400, 400], r=10, clearance=20, edges='extended')
dt = np.around(edt(im), decimals=0).astype(int)
# Obtain skeleton with proper handling of the boundaries
sk2 = skeletonize_magnet(im)
# Extract junctions and endpoints from skeleton
jct = analyze_skeleton(sk2)
# Identify pieces of skeleton smaller than some threshold and call them junctions
labels = spim.label(sk2*~jct.juncs, structure=ps.tools.ps_rect(3, im.ndim))[0]
labels = region_size(labels + 1)  # Add 1 so void has label for next line
jct.juncs += (labels <= 5)  # Void size will be huge so will not be found by <
# Draw spheres at each junction point
spheres = draw_spheres(sites=jct.endpts + jct.juncs, sizes=dt)
# Draw slighly larger spheres to use as a mask below
spheres1 = draw_spheres(sites=jct.endpts + jct.juncs, sizes=dt+2)
# Find throat junctions, aided by removing skeleton from under dilated spheres
jct.throats = find_throat_points(im=im, sk=sk2*~spheres, l_max=7)
# Remove any throat junctions that are too close to pore bodies
jct.throats = jct.throats*(~spheres1)
# Draw spheres on each surviving throat junction
spheres2 = draw_spheres(sites=jct.throats, sizes=dt)
# Now merge the two sets of spheres and visulize
temp = np.zeros_like(im, dtype=int)
temp[jct.endpts] = 3
temp[jct.juncs] = 4
temp[jct.throats] = 5
plt.imshow((spheres + spheres2*2.0)*(temp == 0)/im)
