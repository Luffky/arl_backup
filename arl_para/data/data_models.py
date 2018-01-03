import numpy as np
from arl.data.data_models import *
from astropy import constants
from astropy import units as u
from astropy.coordinates import SkyCoord
from arl.data.parameters import *

class visibility_for_para:
    def __init__(self, vis=None, uvw=None, time=None, frequency=None, bandwidth=None, integration_time=None, antenna1=None, antenna2=None,
                 weight=None, imaging_weight=None, visibility=None, nvis=1, data=None):
        npol = 4
        if visibility != None:
            npol = visibility.polarisation_frame.npol

        if type(data) == type(None):
            desc = [('uvw', '>f8', (3,)),
                    ('time', '>f8'),
                    ('frequency', '>f8'),
                    ('channel_bandwidth', '>f8'),
                    ('integration_time', '>f8'),
                    ('antenna1', '>i8'),
                    ('antenna2', '>i8'),
                    ('vis', '>c16', (npol,)),
                    ('weight', '>f8', (npol,)),
                    ('imaging_weight', '>f8', (npol,)),
                    ]

            data = np.zeros(shape=nvis, dtype=desc)
            data['uvw'] = uvw
            data['time'] = time
            data['frequency'] = frequency
            data['channel_bandwidth'] = bandwidth
            data['integration_time'] = integration_time
            data['antenna1'] = antenna1
            data['antenna2'] = antenna2
            data['vis'] = vis
            data['weight'] = weight
            data['imaging_weight'] = imaging_weight
            self.data = data
        else :
            self.data = data

        self.nvis = nvis # 通过nvis可以任意调整并行粒度
        self.npol = npol

        # 便于处理
        self.phasecentre = None
        self.polarisation_frame = None
        self.configuration = None

        # 若存在时间或空间上的压缩，此处存储压缩前的对应于本身的vis类
        self.blockvis = None

        # 在visibility中增加key的保存
        self.keys = []  # {beam, major_loop, frequency, time, ant1, ant2, facet, polarisation}

    @property
    def block_uvw(self):
        '''
            拿到blockvisibilit中的uvw值, 做一个简单的变换即可
        :return:
        '''
        return self.data['uvw'] * constants.c.value / self.data['frequency'][0]

    @property
    def uvw(self):
        return self.data['uvw']

    @property
    def w(self):
        return self.data['uvw'][:, 2]

    @property
    def vis(self):
        return self.data['vis']

    @property
    def time(self):
        return self.data['time']

    @property
    def frequency(self):
        return self.data['frequency']

    @property
    def channel_bandwidth(self):
        return self.data['channel_bandwidth']
    @property
    def integration_time(self):
        return self.data['integration_time']

    @property
    def antenna1(self):
        return self.data['antenna1']

    @property
    def antenna2(self):
        return self.data['antenna2']
    @property
    def weight(self):
        return self.data['weight']

    @property
    def imaging_weight(self):
        return self.data['imaging_weight']

    def __str__(self):
        print(self.uvw)
        print(self.vis)
        return ""

class visibility_share:
    '''
        被并行化后的visibility的共享数据
    '''
    def __init__(self, visibility: Visibility, ntime, nchan, nant):
        if visibility != None:
            self.phasecentre = visibility.phasecentre
            self.polarisation_frame = visibility.polarisation_frame
            self.configuration = visibility.configuration
            self.npol = visibility.polarisation_frame.npol
            self.nvis = visibility.nvis
        else:
            self.phasecentre = None
            self.polarisation_frame = None
            self.configuration = None
            self.npol = None
            self.nvis = None
        self.ntime = ntime
        self.nchan = nchan
        self.nant = nant


    def __str__(self):
        ret = "phasecentre: %s, polarisation_frame_type: %s, configuration: %s" % (str(self.phasecentre),
        str(self.polarisation_frame), str(self.configuration))


class image_for_para:
    '''
        image按照nchanl和npol切开后的数据结构便于并行化
        每一个image由frequency time facet polarisation唯一决定
        另外不同的切片会有不同的wcs, wcs.wcs.crpix值后产生变化，其他的数据不变
        每个被切片的image产生的第一个切片的wcs和原image的wcs是相同的
    '''
    def __init__(self, data, wcs, keys):
        self.keys = keys # {beam, major_loop, frequency, time, facet, polarisation}
        self.data = data
        self.wcs = wcs

    @property
    def beam(self): return self.keys['beam']

    @property
    def major_loop(self): return self.keys['major_loop']

    @property
    def channel(self): return self.keys['channel']

    @property
    def time(self): return self.keys['time']

    @property
    def facet(self): return self.keys['facet']

    @property
    def polarisation(self): return self.keys['polarisation']

    @property
    def ny(self): return self.data.shape[0]

    @property
    def nx(self): return self.data.shape[1]

    @property
    def shape(self):
        return self.data.shape

    @property
    def phasecentre(self):
        return SkyCoord(self.wcs.wcs.crval[0] * u.deg, self.wcs.wcs.crval[1] * u.deg)



class image_share:
    '''
        存储原本的wcs
    '''
    def __init__(self, polarisation_frame: PolarisationFrame, wcs, nchan, npol, ny, nx):
        # 每个chanel对应的frequecy实际的值
        self.polarisation_frame = polarisation_frame
        self.wcs = wcs
        self.nchan = nchan
        self.npol = npol
        self.ny = ny
        self.nx = nx

    @property
    def shape(self):
        return (self.nchan, self.npol, self.ny, self.nx)


class GainTable:
    """ Gain table with data: time, antenna, gain[:, chan, rec, rec], weight columns

    The weight is usually that output from gain solvers.
    """

    def __init__(self, data=None, gain: numpy.array = None, time: numpy.array = None, weight: numpy.array = None,
                 residual: numpy.array = None, frequency: numpy.array = None,
                 receptor_frame: ReceptorFrame = ReceptorFrame("linear")):
        """ Create a gaintable from arrays

        The definition of gain is:

            Vobs = g_i g_j^* Vmodel

        :param data:
        :param gain: [:, nchan, nrec, nrec]
        :param time:
        :param weight:
        :param residual:
        :param frequency:
        :param receptor_frame:
        :return: Gaintable
        """
        if data is None and gain is not None:
            nrec = receptor_frame.nrec
            nrows = gain.shape[0]
            nants = gain.shape[1]
            nchan = gain.shape[2]
            assert len(frequency) == nchan, "Discrepancy in frequency channels"
            desc = [('gain', '>c16', (nants, nchan, nrec, nrec)),
                    ('weight', '>f8', (nants, nchan, nrec, nrec)),
                    ('residual', '>f8', (nchan, nrec, nrec)),
                    ('time', '>f8')]
            self.data = numpy.zeros(shape=[nrows], dtype=desc)
            self.data['gain'] = gain
            self.data['weight'] = weight
            self.data['time'] = time
            self.data['residual'] = residual
        self.frequency = frequency
        self.receptor_frame = receptor_frame

    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.size * sys.getsizeof(self.data)
        return size / 1024.0 / 1024.0 / 1024.0

    @property
    def time(self):
        return self.data['time']

    @property
    def gain(self):
        return self.data['gain']

    @property
    def weight(self):
        return self.data['weight']

    @property
    def residual(self):
        return self.data['residual']

    @property
    def nants(self):
        return self.data['gain'].shape[1]

    @property
    def nchan(self):
        return self.data['gain'].shape[2]

    @property
    def nrec(self):
        return self.receptor_frame.nrec
