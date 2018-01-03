from reproject import reproject_interp
from arl_para.data.data_models import *
from arl_para.image.base import create_image_from_array_para
from astropy.wcs import WCS

def reproject_image_para(im: image_for_para, newwcs: WCS, shape=None) -> (Image, Image):
    '''
        按照一个新的WCS将image_for_para重新投影到shape的大小
    :param im: 原image
    :param newwcs: 新的wcs
    :param shape: 投影的大小
    :return: 一个新的image
    '''
    assert type(im) == image_for_para
    rep, foot = reproject_interp((im.data, im.wcs), newwcs, shape, order='bicubic',
                                 independent_celestial_slices=True)
    return create_image_from_array_para(rep, 0, rep.shape[0], 0, rep.shape[1], im.beam, im.major_loop, im.channel, im.time, im.facet, im.polarisation, newwcs)\
        , create_image_from_array_para(foot, 0, foot.shape[0], 0, foot.shape[1], im.beam, im.major_loop, im.channel, im.time, im.facet, im.polarisation,  newwcs)