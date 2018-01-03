from arl_para.data.data_models import *
import copy
from arl_para.Others.utils import *
import collections
from typing import List, Union
from arl_para.test.Utils import compute_baseline_index

def phaserotate_visibility_para(vis: visibility_for_para, newphasecentre: SkyCoord, tangent=True, inverse=False) -> visibility_for_para:
    """
        并行版
    :param vis: Visibility to be rotated
    :param newphasecentre:
    :param tangent: Stay on the same tangent plane? (True)
    :param inverse: Actually do the opposite
    :return: Visibility
    """
    l, m, n = skycoord_to_lmn(newphasecentre, vis.phasecentre)
    # No significant change?
    if numpy.abs(n) > 1e-15:

        # Make a new copy
        newvis = copy.deepcopy(vis)

        phasor = simulate_point(newvis.data['uvw'], l, m)

        if inverse:
            for pol in range(vis.npol):
                newvis.data['vis'][..., pol] *= phasor
        else:
            for pol in range(vis.npol):
                newvis.data['vis'][..., pol] *= numpy.conj(phasor)

        if not tangent:
            if inverse:
                xyz = uvw_to_xyz(vis.data['uvw'], ha=-newvis.phasecentre.ra.rad, dec=newvis.phasecentre.dec.rad)
                newvis.data['uvw'][...] = \
                    xyz_to_uvw(xyz, ha=-newphasecentre.ra.rad, dec=newphasecentre.dec.rad)[...]
            else:
                # This is the original (non-inverse) code
                xyz = uvw_to_xyz(newvis.data['uvw'], ha=-newvis.phasecentre.ra.rad, dec=newvis.phasecentre.dec.rad)
                newvis.data['uvw'][...] = xyz_to_uvw(xyz, ha=-newphasecentre.ra.rad, dec=newphasecentre.dec.rad)[
                    ...]
            newvis.phasecentre = newphasecentre
        return newvis
    else:
        return vis

def predict_skycomponent_visibility_para(vis: visibility_for_para, sc):
    '''
        并行版 predict_sky_component_visibility
    :param viss:
    :param sc:
    :return:
    '''
    if not isinstance(sc, collections.Iterable):
        sc = [sc]

    for comp in sc:
        l, m, n = skycoord_to_lmn(comp.direction, vis.phasecentre)
        phasor = simulate_point(vis.data['uvw'], l, m)
        for ivis in range(vis.nvis):
            for pol in range(vis.npol):
                vis.data['vis'][ivis, pol] += comp.flux[vis.keys[ivis]['channel'], pol] * phasor[ivis]

    return vis

def sum_visibility_in_one_facet_pol(viss: Union[List[visibility_for_para]]) -> visibility_for_para:
    npol = 0
    nvis = 0
    phasecentre = None
    keys = None
    if type(viss[0]) == tuple:
        npol = viss[0][1].npol
        nvis = viss[0][1].nvis
        phasecentre = viss[0][1].phasecentre
        keys = viss[0][1].keys
    else:
        npol = viss[0].npol
        nvis = viss[0].nvis
        phasecentre = viss[0].phasecentre
        keys = viss[0].keys

    desc = [('uvw', '>f8', (3,)),
            ('time', '>f8'),
            ('frequency', '>f8'),
            ('channel_bandwidth', '>f8'),
            ('integration_time', '>f8'),
            ('antenna1', '>i8'),
            ('antenna2', '>i8'),
            ('vis', '>c16', (npol,)),
            ('weight', '>f8', (npol,)),
            ('imaging_weight', '>f8', (npol,))]
    data = np.zeros(shape=[nvis], dtype=desc)

    for idx, vis in enumerate(viss):
        if type(vis) == tuple:
            vis = vis[1]
        if idx == 0:
            data = copy.copy(vis.data)
        else:
            data['vis'] += vis.data['vis']

    vis = visibility_for_para(data=data, nvis=nvis)
    vis.npol = npol
    vis.keys = keys
    vis.phasecentre = phasecentre

    return vis

def visibility_by_frequenct_to_one(viss: List[visibility_for_para], idxs: List[tuple]):
    npol = viss[0].npol
    nvis = viss[0].nvis * len(viss)
    desc = [('uvw', '>f8', (3,)),
            ('time', '>f8'),
            ('frequency', '>f8'),
            ('channel_bandwidth', '>f8'),
            ('integration_time', '>f8'),
            ('antenna1', '>i8'),
            ('antenna2', '>i8'),
            ('vis', '>c16', (npol,)),
            ('weight', '>f8', (npol,)),
            ('imaging_weight', '>f8', (npol,))]
    data = np.zeros(shape=[nvis], dtype=desc)
    nant = len(np.unique(viss[0].antenna1)) + 1
    nchan = len(viss)
    for vis, idx in zip(viss, idxs):
        if type(vis) == tuple:
            vis = vis[1]
        for i in range(vis.nvis):
            index = idx[2] + compute_baseline_index(vis.keys[i]['ant1'], vis.keys[i]['ant2'],
                                                                    nant) \
                                             * nchan + vis.keys[i]['time'] * nchan \
                                                                 * (nant * (nant - 1)) // 2
            data['uvw'][index] = vis.uvw[i]
            data['time'][index] = vis.time[i]
            data['frequency'][index] = vis.frequency[i]
            data['channel_bandwidth'][index] = vis.channel_bandwidth[i]
            data['integration_time'][index] = vis.integration_time[i]
            data['antenna1'][index] = vis.antenna1[i]
            data['antenna2'][index] = vis.antenna2[i]
            data['vis'][index] = vis.vis[i]
            data['weight'][index] = vis.weight[i]
            data['imaging_weight'][index] = vis.imaging_weight[i]
    visibility = Visibility(data=data, phasecentre=viss[0].phasecentre, configuration=viss[0].configuration,
                            polarisation_frame=viss[0].polarisation_frame)

    return visibility

def divide_visibility_para(vis: Visibility, model: Visibility, time_map, channel_map):
    '''
        对两个visibility的vis做矩阵除法
    :param vis:
    :param model:
    :return:
    '''
    isscalar = vis.polarisation_frame.npol == 1
    if isscalar:
        utimes = np.unique(vis.time)
        uant = np.unique(vis.antenna1)
        uchan = np.unique(vis.frequency)
        ntimes = len(utimes)
        nant = len(uant) + 1
        nchan = len(uchan)
        npol = vis.polarisation_frame.npol
        nvis = vis.nvis
        x = np.zeros([ntimes, nant, nant, nchan, 1], dtype='complex')
        xwt = np.zeros([ntimes, nant, nant, nchan, 1])
        for row in range(nvis):
            row = time_map[vis.time[row]]
            chan = channel_map[vis.frequency[row]]
            ant2 = vis.antenna2[row]
            ant1 = vis.antenna1[row]
            xwt[row, ant2, ant1, chan] = model.vis[row][0] ** 2 * vis.weight[row]
            if xwt[row, ant2, ant1, chan] > 0.0:
                x[row, ant2, ant1, chan] = vis.vis[row] / model.vis[row]

    else:
        utimes = np.unique(vis.time)
        uant = np.unique(vis.antenna1)
        uchan = np.unique(vis.frequency)
        ntimes = len(utimes)
        nant = len(uant) + 1
        nchan = len(uchan)
        npol = vis.polarisation_frame.npol
        nvis = vis.nvis
        nrec = 2
        assert nrec * nrec == npol
        x = np.zeros([ntimes, nant, nant, nchan, nrec, nrec], dtype='complex')
        xwt = np.zeros([ntimes, nant, nant, nchan, nrec, nrec])
        for row in range(nvis):
            ovis = np.matrix(vis.vis[row].reshape([2, 2]))
            mvis = np.matrix(model.vis[row].reshape([2, 2]))
            wt = np.matrix(vis.weight[row].reshape([2, 2]))
            time = time_map[vis.time[row]]
            chan = channel_map[vis.frequency[row]]
            ant2 = vis.antenna2[row]
            ant1 = vis.antenna1[row]
            # print(vis.vis[row])
            # print(time, ant2, ant1, chan)
            x[time, ant2, ant1, chan] = np.matmul(np.linalg.inv(mvis), ovis)
            xwt[time, ant2, ant1, chan] = np.dot(mvis, np.multiply(wt, mvis.H)).real

        x[abs(x)<1e-10] = 0.0
        x = x.reshape([ntimes, nant, nant, nchan, nrec * nrec])
        xwt = xwt.reshape([ntimes, nant, nant, nchan, nrec * nrec])

    return x, xwt

def subtract_visibility(vis: visibility_for_para, modelvis: visibility_for_para):
    vis.data['vis'] = vis.data['vis'] - modelvis.data['vis']
    return vis

def coalesce_visibility_para(vis: visibility_for_para, **kwargs):
    # 暂不考虑time和frequency的压缩
    pass





