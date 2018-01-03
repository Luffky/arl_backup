from astropy.coordinates import SkyCoord
import astropy.units as u
from arl.data.polarisation import *
from arl.skycomponent.operations import create_skycomponent
import copy
# 常数值
class MetaData():
    def __init__(self, nan=3, npix=256, nchan=5, nfacet=2, ntime=5, npol=4, precision=-8, cellsize=0.001, pf='linear'):
        self.FACETS = nfacet
        self.PIECE = self.FACETS * self.FACETS
        self.NPOL = npol
        self.NCHAN = nchan
        self.NX = npix
        self.NY = npix
        self.BEAM = 0
        self.NTIMES = ntime
        self.NAN = nan
        self.NBASE = self.NAN * (self.NAN - 1) // 2

        self.DX = self.NX / 4
        self.DY = self.NY / 4

        # 判断是否相等的精度
        self.PRECISION = precision

        # 天文相关常量
        self.CELLSIZE = cellsize
        self.CTYPE=["RA---SIN", "DEC--SIN"]
        self.PHASECENTRE = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')
        self.POLARISATION_FRAME = PolarisationFrame(pf)

    def create_time(self, limit = 3):
        times = numpy.linspace(-limit, +limit, self.NTIMES) * (numpy.pi / 12.0)
        return times

    def create_frequency(self, start=1e8, bandwidth=0.1e8):
        frequency = numpy.arange(start, start + self.NCHAN * bandwidth, bandwidth)
        return frequency

    def create_channel_bandwidth(self, bandwidth=0.1e8):
        channel_bandwidth = numpy.full(self.NCHAN, bandwidth)
        return channel_bandwidth

    def create_phasecentre(self):
        phasecentre = copy.deepcopy(self.PHASECENTRE)
        return phasecentre

    def create_skycomponent(self):
        compabsdirection = copy.deepcopy(self.compabsdirection)
        f = numpy.array([100.0, 110.0, 120.0, 130.0])  # npol = 4
        flux = numpy.array([f + idx * 100.0 for idx in range(self.NCHAN)])  # nchan,npol
        frequency = self.create_frequency()
        comp = create_skycomponent(flux=flux, frequency=frequency, direction=compabsdirection,
                                   polarisation_frame=PolarisationFrame('linear'))
        return comp

    def create_polarisation_frame(self):
        pf = copy.deepcopy(self.POLARISATION_FRAME)
        return pf
