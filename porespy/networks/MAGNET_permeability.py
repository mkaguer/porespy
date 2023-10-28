import pandas as pd
import porespy as ps
import openpnm as op
import numpy as np
from skimage.morphology import square, cube
import scipy.ndimage as spim

np.random.seed(10)

export = False

fill_blind_pores = True
padding = None
l_max=7
bw = 3
endpoints=True
# names = ['Berea', 'C1', 'C2', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'A1']
names = ['Berea']
resolution = [5.35e-6]
# resolution = [5.35e-6, 2.85e-6, 5.35e-6, 8.68e-6, 4.96e-6, 9.10e-6, 8.96e-6, 4e-6, 5.1e-6, 4.8e-6, 4.89e-6, 3.4e-6, 3.85e-6]
# data = np.zeros((4, len(names)))
Kx, Ky, Kz, Kavg = [], [], [], []
porosity = []
for i, name in enumerate(names):
    # import Blunt images
    name = names[i]
    print(name)
    res = resolution[i]
    raw = np.fromfile(name + '.raw', dtype=np.uint8)
    shape = np.ceil(len(raw)**(1/3)).astype('int')
    imb = (raw.reshape(shape, shape, shape))
    imb = imb == 0
    # imb = imb[:100, :100, :100]
    imb = ps.generators.blobs([400, 400, 400], porosity=0.2, blobiness=3, seed=0)
    if fill_blind_pores:
        imb = ps.filters.fill_blind_pores(imb, conn=6, surface=True)
    imb = ps.filters.trim_floating_solid(imb, conn=6, surface=True)

    im = imb
    porosity.append(np.sum(im)/np.product(im.shape)*100)
    print(f'porosity: {np.sum(im)/np.product(im.shape)*100}')

    # get skeleton
    sk = ps.networks.skeletonize_magnet2(im)
    labels, N = spim.label(sk, structure=ps.tools.ps_rect(3, 3))

    # MAGNET
    net, sk = ps.networks.magnet(
        im=im,
        sk=None,
        padding=padding,
        endpoints=endpoints,
        voxel_size=res,
        l_max=l_max,
        boundary_width=bw,
    )

    # import network to openpnm
    net = op.io.network_from_porespy(net)
    net['pore.diameter'] = net['pore.radius']*2
    net['throat.diameter'] = net['throat.radius']*2

    # check network health
    h = op.utils.check_network_health(net)
    op.topotools.trim(net, pores=np.append(h['disconnected_pores'], h['isolated_pores']))
    h = op.utils.check_network_health(net)

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
    K_x = Q_x * L * mu / (A * (Pin - Pout))/0.98e-12*1000
    print(f'K_x is: {K_x:.2f} mD')

    ymin = net.pores('ymin')
    ymax = net.pores('ymax')
    flow_y = op.algorithms.StokesFlow(network=net, phase=phase)
    flow_y.set_value_BC(pores=ymin, values=Pin)
    flow_y.set_value_BC(pores=ymax, values=Pout)
    flow_y.run()

    Q_y = flow_y.rate(pores=ymin, mode='group')[0]
    K_y = Q_y * L * mu / (A * (Pin - Pout))/0.98e-12*1000
    print(f'K_y is: {K_y:.2f} mD')

    zmin = net.pores('zmin')
    zmax = net.pores('zmax')
    flow_z = op.algorithms.StokesFlow(network=net, phase=phase)
    flow_z.set_value_BC(pores=zmin, values=Pin)
    flow_z.set_value_BC(pores=zmax, values=Pout)
    flow_z.run()

    Q_z = flow_z.rate(pores=zmin, mode='group')[0]
    K_z = Q_z * L * mu / (A * (Pin - Pout))/0.98e-12*1000
    print(f'K_z is: {K_z:.2f} mD')

    K = np.average([K_x, K_y, K_z])
    print(f'K is: {K:.2f} mD')

    # number of pore vs. skeleton clusters in network
    from scipy.sparse import csgraph as csg
    am = net.create_adjacency_matrix(fmt='coo', triu=True)
    Np, cluster_num = csg.connected_components(am, directed=False)
    print('Pore clusters:', Np)
    # number of skeleton pieces
    b = square(3) if im.ndim == 2 else cube(3)
    _, Ns = spim.label(input=sk.astype('bool'), structure=b)
    print('Skeleton clusters:', Ns)

    # append data
    Kx.append(K_x)
    Ky.append(K_y)
    Kz.append(K_z)
    Kavg.append(K)

    # export
    if export:
        ps.io.to_stl(~sk, name + '-sk')
        ps.io.to_stl(im, name + '-im')
        net['pore.coords'] = net['pore.coords']/res
        net['pore.diameter'] = net['pore.diameter']/res
        net['throat.radius'] = net['throat.radius']/res
        net['pore.coords'] += 10  #*res
        proj = net.project
        op.io.project_to_xdmf(proj, filename=name + '-network')

'''
# create and export data frame
data = {
        "Name": names,
        "Resolution": resolution,
        "boundary_width": bw,
        "l_max": l_max,
        "padding": str(padding),
        "endpoints": endpoints,
        "porosity": porosity,
        "K_x": Kx,
        "K_y": Ky,
        "K_z": Kz,
        "K": Kavg
        }
df = pd.DataFrame(data=data)
df.to_csv('MAGNET Permeability.csv', index=False)
'''
