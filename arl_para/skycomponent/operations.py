from arl_para.data.data_models import *
from arl.data.data_models import Skycomponent
from typing import Union, List
import collections
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

def insert_skycomponent_para(im: image_for_para, sc: Union[Skycomponent, List[Skycomponent]], insert_method='',
                        bandwidth=1.0, support=8) -> image_for_para:
    '''
    :param im: 被插入的image
    :param sc: 插入的skycomponent，可以有多个skycomponent
    :param insert_method: 插入方法，四种分别为: Lanczos Sinc PSWF 和 缺省方法
    :param bandwidth:
    :param support:
    :return: 新的image
    '''
    if type(im) == tuple:
        im = im[1]

    assert type(im) == image_for_para

    support = int(support / bandwidth)

    ny, nx = im.shape

    if not isinstance(sc, collections.Iterable):
        sc = [sc]

    for comp in sc:
        assert comp.shape == 'Point', "Cannot handle shape %s" % comp.shape
        pixloc = skycoord_to_pixel(comp.direction, im.wcs, 1, 'wcs')
        if insert_method == "Lanczos":
            insert_array_para(im.data, pixloc[0], pixloc[1], comp.flux[im.channel, im.polarisation], bandwidth, support,
                         insert_function=insert_function_L)
        elif insert_method == "Sinc":
            insert_array_para(im.data, pixloc[0], pixloc[1], comp.flux[im.channel, im.polarisation], bandwidth, support,
                         insert_function=insert_function_sinc)
        elif insert_method == "PSWF":
            insert_array_para(im.data, pixloc[0], pixloc[1], comp.flux[im.channel, im.polarisation], bandwidth, support,
                         insert_function=insert_function_pswf)
        else:
            y, x = numpy.round(pixloc[1]).astype('int'), numpy.round(pixloc[0]).astype('int')
            if x >= 0 and x < nx and y >= 0 and y < ny:
                im.data[y, x] += comp.flux[im.channel, im.polarisation]
    return im

# 以下三个函数不受并行化影响
def insert_function_sinc(x):
    s = numpy.zeros_like(x)
    s[x != 0.0] = numpy.sin(numpy.pi*x[x != 0.0])/(numpy.pi*x[x != 0.0])
    return s

def insert_function_L(x, a = 5):
    L = insert_function_sinc(x) * insert_function_sinc(x/a)
    return L

def insert_function_pswf(x, a=5):
    from arl.fourier_transforms.convolutional_gridding import grdsf
    return grdsf(abs(x)/a)[0]

def insert_array_para(im: image_for_para, x, y, flux, bandwidth=1.0, support=7, insert_function=insert_function_sinc):
    '''
        根据insert_function的不同的插入的大小和值不同
    :param im: image
    :param x: pixloc，插入的位置相关
    :param y:
    :param flux: 插入的值
    :param bandwidth:
    :param support:
    :param insert_function: 插入的方式
    :return:
    '''
    ny, nx = im.shape
    intx = int(numpy.round(x))
    inty = int(numpy.round(y))
    fracx = x - intx
    fracy = y - inty
    gridx = numpy.arange(-support, support)
    gridy = numpy.arange(-support, support)

    insert = numpy.outer(insert_function(bandwidth * (gridy - fracy)),
                         insert_function(bandwidth * (gridx - fracx))) # [support * 2, support * 2]

    insertsum = numpy.sum(insert)
    assert insertsum > 0, "Sum of interpolation coefficients %g" % insertsum
    insert = insert / insertsum # [support * 2, support * 2]

    for iy in gridy:
        for ix in gridx:
            if (iy + inty >= 0 and iy + inty < ny and ix + intx >=0 and ix + intx < nx):
                im[iy + inty, ix + intx] += flux * insert[iy + support, ix + support]

    return im