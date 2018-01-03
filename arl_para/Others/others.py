from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import copy
import numpy as np

def wcs4_to_wcs2(wcs4: WCS) -> WCS:
    '''
        将wcs4转换为wcs2, 主要是将四维的Image 拆分成二维的image_for_para时使用
    :param wcs4: axis为4的wcs
    :return: axis为2的wcs
    '''
    newwcs = WCS(naxis=2)
    newwcs.wcs.crpix = wcs4.wcs.crpix[0:2]
    newwcs.wcs.cdelt = wcs4.wcs.cdelt[0:2]
    newwcs.wcs.crval = wcs4.wcs.crval[0:2]
    newwcs.wcs.ctype = [wcs4.wcs.ctype[0], wcs4.wcs.ctype[1]]
    newwcs.wcs.radesys = wcs4.wcs.radesys
    newwcs.wcs.equinox = wcs4.wcs.equinox
    return newwcs

def create_new_wcs_new_shape(im, shape):
    '''
        对新的image的phasecentre进行重新计算
    :param im:
    :param shape:
    :return:
    '''
    newwcs = copy.deepcopy(im.wcs)
    ny = 0
    nx = 0
    if len(im.shape) == 4:
        ny, nx = im.shape[2:]
    elif len(im.shape) == 2:
        ny, nx = im.shape
    image_phasecentre = pixel_to_skycoord(nx // 2, ny // 2, im.wcs, origin=1)
    newshape = np.array(shape)
    newwcs.wcs.crval[0] = image_phasecentre.ra.deg
    newwcs.wcs.crval[1] = image_phasecentre.dec.deg
    # print(newwcs.wcs.crval)
    return newwcs, newshape