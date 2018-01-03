from arl_para.data.data_models import visibility_for_para
from arl.visibility.base import create_blockvisibility
from arl.visibility.base import create_visibility
from arl.data.data_models import Configuration
import numpy as np
from astropy.coordinates import SkyCoord
from arl.data.polarisation import PolarisationFrame
from arl.util.testing_support import create_named_configuration
from arl.skycomponent.operations import create_skycomponent
from astropy import units as u
import copy
from arl.data.data_models import Visibility
from arl_para.data.data_models import visibility_for_para, visibility_share
from arl_para.test.Utils import compute_baseline_index
from typing import Union, List

def visibility_to_visibility_para(vis: Visibility, mode="nto1", keys=None):
    '''
        由visibility生成visibility_for_para, 开发测试用，实际生产环境将会直接通过Configuration生成visibility_para
    :param visibility
    :return: 并行化的visibility_for_para 键值对(id, vis) 和 visibility的共享数据
    '''
    if mode == "nto1":
        vis = copy.deepcopy(vis)
        viss = []
        # 将不同的frequency 和 time 投影到唯一的从0开始的整数上便于作为key, 即方便spark的shuffle又方便卷积时和image进行对应
        ufrequency = np.unique(vis.data['frequency'])
        m = {}
        for idx, chan in enumerate(ufrequency):
            m[chan] = idx

        utime = np.unique(vis.data['time'])
        t = {}
        for idx, time in enumerate(utime):
            t[time] = idx


        for i in range(vis.nvis):
            v = visibility_for_para(vis.data['vis'][i], vis.data['uvw'][i]
                                      ,vis.data['time'][i],vis.data['frequency'][i],vis.data["channel_bandwidth"][i]
                               ,vis.data['integration_time'][i],vis.data['antenna1'][i],vis.data['antenna2'][i]
                               ,vis.data['weight'][i], vis.data["imaging_weight"][i], visibility=vis)
            v.phasecentre = vis.phasecentre
            v.polarisation_frame = vis.polarisation_frame
            v.configuration = vis.configuration
            channel = m[v.data['frequency'][0]]
            time = t[v.data['time'][0]]
            ant1 = v.data['antenna1'][0]
            ant2 = v.data['antenna2'][0]

            v.keys = {"beam": 0, "major_loop": 0, "channel": channel, "time": time, "ant1": ant1, "ant2": ant2,
                       "facet": 0, "polarisation": 0}

            viss.append(((channel, time, ant1, ant2), v))
        vis_share = visibility_share(vis, utime.shape[0], ufrequency.shape[0], np.max(vis.antenna2 + 1))
        return viss, vis_share

    elif mode == "1to1":
        vis = copy.deepcopy(vis)
        ufrequency = np.unique(vis.data['frequency'])
        m = {}
        if keys == None:
            for idx, chan in enumerate(ufrequency):
                m[chan] = idx
        else:
            temp = (keys['chan'], ufrequency)
            for idx, chan in zip(*temp):
                m[chan] = idx

        utime = np.unique(vis.data['time'])
        t = {}
        for idx, time in enumerate(utime):
            t[time] = idx

        v = visibility_for_para(vis.data['vis'], vis.data['uvw']
                                , vis.data['time'], vis.data['frequency'], vis.data["channel_bandwidth"]
                                , vis.data['integration_time'], vis.data['antenna1'], vis.data['antenna2']
                                , vis.data['weight'], vis.data["imaging_weight"], visibility=vis, nvis=vis.nvis)
        v.phasecentre = vis.phasecentre
        v.polarisation_frame = vis.polarisation_frame
        v.configuration = vis.configuration
        for i in range(v.nvis):
            channel = m[v.data['frequency'][i]]
            time = t[v.data['time'][i]]
            ant1 = v.data['antenna1'][i]
            ant2 = v.data['antenna2'][i]
            v.keys.append({"beam": 0, "major_loop": 0, "channel": channel, "time": time, "ant1": ant1, "ant2": ant2,
                       "facet": 0, "polarisation": 0})

        vis_share = visibility_share(vis, utime.shape[0], ufrequency.shape[0], np.max(vis.antenna2 + 1))
        return v, vis_share



def visibility_para_to_visibility(viss: List[visibility_for_para], vis_share: visibility_share, mode="nto1") -> Visibility:
    '''
        拆分后的visibility还原为原visibility
    :param viss: 待还原的visbility_for_para list
    :param visibility: 共享的visibility数据
    :return:
    '''
    if mode == "nto1":
        npol = vis_share.npol
        nvis = vis_share.nvis
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
        for vis in viss:
            if type(vis) == tuple:
                vis = vis[1]
            index = vis.keys["channel"] + compute_baseline_index(vis.keys['ant1'], vis.keys['ant2'], vis_share.nant) \
                                            * vis_share.nchan + vis.keys['time'] * vis_share.nchan \
                                            * (vis_share.nant * (vis_share.nant - 1)) // 2
            data['uvw'][index] = vis.uvw[0]
            data['time'][index] = vis.time[0]
            data['frequency'][index] = vis.frequency[0]
            data['channel_bandwidth'][index] = vis.channel_bandwidth[0]
            data['integration_time'][index] = vis.integration_time[0]
            data['antenna1'][index] = vis.antenna1[0]
            data['antenna2'][index] = vis.antenna2[0]
            data['vis'][index] = vis.vis[0]
            data['weight'][index] = vis.weight[0]
            data['imaging_weight'][index] = vis.imaging_weight[0]

        visibility = Visibility(data=data, phasecentre=vis_share.phasecentre, configuration=vis_share.configuration,
                                polarisation_frame=vis_share.polarisation_frame)

    elif mode == "1to1":
        npol = vis_share.npol
        nvis = vis_share.nvis
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
        for vis in viss:
            if type(vis) == tuple:
                vis = vis[1]
            for i in range(vis.nvis):
                index = vis.keys[i]["channel"] + compute_baseline_index(vis.keys[i]['ant1'], vis.keys[i]['ant2'], vis_share.nant) \
                                                * vis_share.nchan + vis.keys[i]['time'] * vis_share.nchan \
                                                * (vis_share.nant * (vis_share.nant - 1)) // 2
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
        visibility = Visibility(data=data, phasecentre=vis_share.phasecentre, configuration=vis_share.configuration,
                                polarisation_frame=vis_share.polarisation_frame)

    return visibility

def create_visibility_para(config: Configuration, times: np.array, frequency: np.array,
                      channel_bandwidth, phasecentre: SkyCoord,
                      weight: float, polarisation_frame=PolarisationFrame('stokesI'),
                      integration_time=1.0, mode="nto1", keys=None, **kwargs):
    visibility = create_visibility(config=config, times=times ,frequency=frequency,
                      channel_bandwidth=channel_bandwidth, phasecentre=phasecentre,
                      weight=weight,  polarisation_frame=polarisation_frame,
                      integration_time=integration_time, **kwargs)
    viss, vis_share = visibility_to_visibility_para(visibility, mode=mode, keys=keys)
    return viss, vis_share














