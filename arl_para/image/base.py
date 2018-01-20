from arl_para.data.data_models import *
from astropy.wcs import WCS
from arl.data.polarisation import PolarisationFrame
from arl.image.operations import *
from arl_para.test.Constants import *
from arl_para.Others.others import *
from arl_para.test.Constants import *
from arl.data.data_models import Image
from typing import Union, List

def create_image_from_array_para(im , y1:int, y2:int, x1:int, x2:int, beam, major_loop, frequency, time, facet, polarisation, wcs: WCS = None) -> image_for_para:
    '''
    以im二维数组作为data的内容，并按照一定坐标切分，创造新的image_para类
    :param im:  被切分的image， 类型[y, x]
    :param y1:  切分的坐标：[y1 ~ y2, x1 ~ x2]
    :param y2:
    :param x1:
    :param x2:
    :param facet:  赋予每个facet的id
   :param wcs: 每个facet的wcs
    :return: 切分后的image
    '''
    keys = {"beam": beam, "major_loop": major_loop, "channel": frequency, "time": time, "facet": facet, "polarisation": polarisation}
    fim = image_for_para(copy.deepcopy(im[y1:y2, x1:x2]), None, keys)
    if wcs is None:
        fim.wcs = None
    else:
        fim.wcs = wcs.deepcopy()
    assert type(fim) == image_for_para, "Type is %s" % type(fim)
    return fim


def image_make_facets(im: image_for_para, facets: int):
    '''
        将image切分为facets * facets个小image的生成器
    :param im: 被切片的image
    :param facets: 切片片数，type:int
    :param polarisation_frame: 极化框架，type:PolarisationFrame, 和被切片前的一样，由image_share提供
    :return: 被前片后的image_for_para list
    '''
    # 每个切片的长和宽
    dx = int(im.nx // facets)
    dy = int(im.ny // facets)
    # 每个切片的id
    id = 0
    for y in range(0, im.ny, dy):
        for x in range(0, im.nx, dx):
            # 调整切片后每个image的wcs
            wcs = im.wcs.deepcopy()
            wcs.wcs.crpix[0] -= x
            wcs.wcs.crpix[1] -= y
            # 作为生成器返回切片后的image
            yield create_image_from_array_para(im.data, y, y + dy, x, x + dx, im.beam, im.major_loop, im.channel,
                                               im.time, id, im.polarisation, wcs)
            id += 1

def create_image_para(ny, nx, frequency, phasecentre, cellsize=0.001,
                                polarisation_frame=PolarisationFrame('linear'), FACET=2):
    '''
        创建空的并行已切片的image
    :param ny:
    :param nx:
    :param frequency:
    :param phasecentre:
    :param cellsize:
    :param polarisation_frame:
    :return:
    '''
    wcs4 = WCS(naxis=4)
    wcs4.wcs.crpix = [ny // 2, nx // 2 + 1.0, 1.0, 1.0]
    wcs4.wcs.cdelt = [-180.0 * cellsize / np.pi, +180.0 * cellsize / np.pi, 1.0, frequency[1] - frequency[0]]
    wcs4.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
    wcs4.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    wcs4.wcs.radesys = 'ICRS'
    wcs4.wcs.equinox = 2000.00

    imgs = []
    nchan = frequency.shape[0]
    npol = polarisation_frame.npol
    img_share = image_share(polarisation_frame, wcs4, nchan, npol, ny, nx)

    for i in range(nchan):
        for j in range(npol):
            data = np.zeros([ny, nx])
            keys = {"beam": 0, "major_loop": 0, "channel": i, "time": 0, "facet": 0, "polarisation": j}
            image_para = image_for_para(data, wcs4_to_wcs2(wcs4), keys)
            for im in image_make_facets(image_para, FACET):
                imgs.append(((i, 0, im.facet, j), im))
    return imgs, img_share

def create_image_para_2(ny, nx, chan, pol, facet, phasecentre, cellsize=0.001, polarisation_frame=PolarisationFrame('linear'), FACET=2):
    wcs4 = WCS(naxis=4)
    wcs4.wcs.crpix = [ny * FACET // 2, nx * FACET // 2 + 1.0, 1.0, 1.0]
    wcs4.wcs.cdelt = [-180.0 * cellsize / np.pi, +180.0 * cellsize / np.pi, 1.0, 1.0]
    wcs4.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, 1.0]
    wcs4.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    wcs4.wcs.radesys = 'ICRS'
    wcs4.wcs.equinox = 2000.00
    data = np.zeros([ny, nx])
    keys = {"beam": 0, "major_loop": 0, "channel": chan, "time": 0, "facet": facet, "polarisation": pol}

    y = facet // FACET
    x = facet % FACET
    wcs4.wcs.crpix[0] -= x * nx
    wcs4.wcs.crpix[1] -= y * ny
    image_para = image_for_para(data, wcs4_to_wcs2(wcs4), keys)
    return image_para




def create_image(ny, nx, frequency, phasecentre, cellsize=0.001,  polarisation_frame=PolarisationFrame('linear')):
    '''
        创建和上个函数等效的非并行的image
    :param ny:
    :param nx:
    :param frequency:
    :param phasecentre:
    :param cellsize:
    :param polarisation_frame:
    :return:
    '''
    wcs4 = WCS(naxis=4)
    wcs4.wcs.crpix = [ny // 2, nx // 2 + 1.0, 1.0, 1.0]
    wcs4.wcs.cdelt = [-180.0 * cellsize / np.pi, +180.0 * cellsize / np.pi, 1.0, frequency[1] - frequency[0]]
    wcs4.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
    wcs4.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    wcs4.wcs.radesys = 'ICRS'
    wcs4.wcs.equinox = 2000.00
    nchan = frequency.shape[0]
    npol = polarisation_frame.npol
    image = Image()
    image.wcs = wcs4
    image.data = np.zeros([nchan, npol, ny, nx])
    image.polarisation_frame = polarisation_frame
    return image

def image_para_to_image(ims: List[image_for_para], image_share: image_share) -> Image:
    '''
        将并行image_for_para还原为原本的Image类，验证算法正确性用
    :param ims:  只有一个facet的并行Image list 类型：list[image_para]
    :param image_share: 并行Image的共享消息
    :return: 还原后的image
    '''
    image = Image()
    datatype = None
    if type(ims[0]) == tuple:
        datatype = ims[0][1].data.dtype
    else:
        datatype = ims[0].data.dtype

    data = np.zeros([image_share.nchan, image_share.npol, image_share.ny, image_share.nx], dtype=datatype)
    dy = 0
    dx = 0
    if type(ims[0]) == tuple:
        dy = ims[0][1].ny
        dx = ims[0][1].nx
    else:
        dy = ims[0].ny
        dx = ims[0].nx

    assert  image_share.ny // dy == image_share.nx // dx
    facet = image_share.ny // dy

    for im in ims:
        if type(im) == tuple:
            im = im[1]
        nchan = im.channel
        npol = im.polarisation
        y = im.facet // facet
        x = im.facet % facet
        data[nchan, npol, y*dy:(y+1)*dy, x*dx:(x+1)*dx] = im.data
    image.data = data
    image.wcs = image_share.wcs
    image.polarisation_frame = image_share.polarisation_frame
    return image

