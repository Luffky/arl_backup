from arl_para.data.data_models import *
import copy
from arl_para.test.Constants import *
from arl_para.imaging.params import *
from arl_para.fourier_transforms.fft_support import *
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from arl_para.visibility.operations import *

def predict_2d_base_para(vis: visibility_for_para, model: image_for_para, **kwargs):
    '''
        用image 预测 visibility 生成新的visibility
    :param vis: 待预测的visibility
    :param model:
    :param kwargs:
    :return: 预测后的visiblity
    '''
    ny, nx = model.shape
    FACETS = get_parameter(kwargs, "FACETS", 2)
    uvw_mode, shape, padding, vuvwmap = get_uvw_map_para(vis, model, padding=FACETS)
    kernel_name, gcf, vkernellist = get_kernel_list_para(vis, model, FACETS, **kwargs)
    # 此处gcf大小和padding后的model大小相同, 然后做fft，fft和padding函数均不用变动
    uvgrid = fft((pad_mid(model.data, int(round(padding * nx))) * gcf).astype(dtype=complex))
    vis.data['vis'] = convolutional_degrid_para(vkernellist, vis.data['vis'].shape, uvgrid,
                                            vuvwmap, model.polarisation)

    svis = shift_vis_to_image_para(vis, model, tangent=True, inverse=True)
    return svis

def predict_facets_para(vis: visibility_for_para, model: image_for_para, predict_function=predict_2d_base_para, **kwargs):
    '''
        predict的主入口
    :param vis:
    :param model:
    :param predict_function:
    :param kwargs:
    :return:
    '''
    if type(vis) == tuple:
        vis = vis[1]
    return predict_with_image_iterator_para(vis, model, predict_function=predict_function,
                                       **kwargs)

def predict_with_image_iterator_para(vis: visibility_for_para, model: image_for_para,
                                predict_function=predict_2d_base_para, **kwargs):
    '''
    4个pol分在同一组
    :param vis:
    :param model:
    :param predict_function:
    :param kwargs:
    :return:
    '''
    result = copy.deepcopy(vis)
    result.data['vis'][...] = 0.0
    result = predict_function(result, model, **kwargs)
    vis.data['vis'][...] += result.data['vis'][...]
    return vis

def convolutional_degrid_para(kernel_list, vshape, uvgrid, vuvwmap, pol):
    kernel_indices, kernels = kernel_list
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    ny, nx = uvgrid.shape
    vis = numpy.zeros(vshape, dtype='complex')

    y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:, 1])
    y -= gh // 2
    x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:, 0])
    x -= gw // 2 # (1,1)

    if len(kernels) > 1:
        coords = (kernel_indices, x, y, xf, yf)
        ckernels = numpy.conjugate(kernels)
        vis[..., pol] = [
            numpy.sum(uvgrid[yy:yy + gh, xx:xx + gw] * ckernels[kind][yyf, xxf, :, :])
            for kind, xx, yy, xxf, yyf in zip(*coords)
        ]
    else:
        coords = (x, y, xf, yf)
        ckernel0 = numpy.conjugate(kernels[0])
        vis[..., pol] = [
            numpy.sum(uvgrid[yy:yy + gh, xx:xx + gw] * ckernel0[yyf, xxf, :, :])
            for xx, yy, xxf, yyf in zip(*coords)
        ]

    return vis

def frac_coord(npixel, kernel_oversampling, p):
    """ Compute whole and fractional parts of coordinates, rounded to
    """
    assert numpy.array(p >= -0.5).all() and numpy.array(
        p < 0.5).all(), "Cellsize is too large: uv overflows grid uv= %s" % str(p)
    x = npixel // 2 + p * npixel
    flx = numpy.floor(x + 0.5 / kernel_oversampling)
    fracx = numpy.around((x - flx) * kernel_oversampling)
    return flx.astype(int), fracx.astype(int)

def shift_vis_to_image_para(vis: visibility_for_para, im: image_for_para, tangent: bool = True, inverse: bool = False):
    '''
        旋转visibility的phasecentre到image的phasecentre，改变vis的uvw等值
    :param vis:
    :param im:
    :param tangent:
    :param inverse:
    :return:
    '''
    ny, nx = im.shape
    image_phasecentre = pixel_to_skycoord(nx // 2, ny // 2, im.wcs, origin=1)

    if vis.phasecentre.separation(image_phasecentre).rad > 1e-15:
        vis = phaserotate_visibility_para(vis, image_phasecentre, tangent=tangent, inverse=inverse)
        vis.phasecentre = im.phasecentre


    return vis




