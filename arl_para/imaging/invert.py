from arl_para.data.data_models import *
import copy
from astropy.wcs import WCS
from arl_para.imaging.convolution import *
from arl_para.fourier_transforms.fft_support import *

def invert_2d_base_para(vis: visibility_for_para, im: image_for_para, dopsf: bool = False, normalize: bool = True, **kwargs) \
        -> (image_for_para, numpy.ndarray):
    svis = copy.deepcopy(vis)
    if dopsf:
        svis.data['vis'] = numpy.ones_like(svis.data['vis'])
    svis = shift_vis_to_image_para(svis, im, tangent=True, inverse=False)
    ny, nx = im.data.shape

    FACETS = get_parameter(kwargs, "FACETS", 2)
    uvw_mode, shape, padding, vuvwmap = get_uvw_map_para(svis, im, padding=FACETS)
    kernel_name, gcf, vkernellist = get_kernel_list_para(svis, im, FACETS, **kwargs)

    # Optionally pad to control aliasing
    imgridpad = numpy.zeros([int(round(padding * ny)), int(round(padding * nx))], dtype='complex')
    imgridpad, sumwt = convolutional_grid_para(vkernellist, imgridpad, svis.data['vis'],
                                          svis.data['imaging_weight'],
                                          vuvwmap, im.polarisation)

    # Fourier transform the padded grid to image, multiply by the gridding correction
    # function, and extract the unpadded inner part.

    # Normalise weights for consistency with transform
    sumwt /= float(padding * int(round(padding * nx)) * ny)

    imaginary = get_parameter(kwargs, "imaginary", False)
    if imaginary:
        result = extract_mid(ifft(imgridpad) * gcf, npixel=nx)
        resultreal = create_image_from_array_2(result.real, im.wcs, vis.polarisation_frame)
        resultimag = create_image_from_array_2(result.imag, im.wcs, vis.polarisation_frame)
        if normalize:
            resultreal = normalize_sumwt_para(resultreal, sumwt)
            resultimag = normalize_sumwt_para(resultimag, sumwt)
        return resultreal, sumwt, resultimag
    else:
        result = extract_mid(numpy.real(ifft(imgridpad)) * gcf, npixel=nx)
        resultimage = create_image_from_array_2(result, im.wcs)
        if normalize:
            resultimage = normalize_sumwt_para(resultimage, sumwt)
        return resultimage, sumwt

def create_image_from_array_2(data: numpy.array, wcs: WCS = None,
                            polarisation_frame=PolarisationFrame('linear')):
    fim = image_for_para(None, None, None)
    fim.polarisation_frame = polarisation_frame
    fim.data = data
    if wcs is None:
        fim.wcs = None
    else:
        fim.wcs = wcs.deepcopy()
    return fim

def convolutional_grid_para(kernel_list, uvgrid, vis, visweights, vuvwmap, pol):
    kernel_indices, kernels = kernel_list
    kernel_oversampling, _, gh, gw = kernels[0].shape
    assert gh % 2 == 0, "Convolution kernel must have even number of pixels"
    assert gw % 2 == 0, "Convolution kernel must have even number of pixels"
    ny, nx = uvgrid.shape

    sumwt = numpy.zeros([1])

    y, yf = frac_coord(ny, kernel_oversampling, vuvwmap[:, 1])
    y -= gh // 2
    x, xf = frac_coord(nx, kernel_oversampling, vuvwmap[:, 0])
    x -= gw // 2

    # Now we can loop over all rows
    wts = visweights[...]
    viswt = vis[...] * visweights[...]
    npol = vis.shape[-1]

    if len(kernels) > 1:
        coords = (kernel_indices, x, y, xf, yf)
        for v, vwt, kind, xx, yy, xxf, yyf in zip(viswt[..., pol], wts[..., pol], *coords):
            uvgrid[yy:yy+gh, xx:xx+gw] += kernels[kind][yyf, xxf, :, :] * v
            sumwt[0] += vwt
    else:
        kernel0 = kernels[0]
        coords = x, y, xf, yf
        for v, vwt, xx, yy, xxf, yyf in zip(viswt[..., pol], wts[..., pol], *coords):
            uvgrid[yy:yy+gh, xx:xx+gw] += kernel0[yyf, xxf, :, :] * v
            sumwt[0] += vwt


    return uvgrid, sumwt

def invert_facets_para(vis: visibility_for_para, im: image_for_para, dopsf=False, normalize=True, invert_function=invert_2d_base_para, **kwargs) \
        -> (Image, numpy.ndarray):
    log.info("invert_facets: Inverting by image facets")
    return invert_with_image_iterator_para(vis, im, normalize=normalize, dopsf=dopsf,
                                      invert_function=invert_function, **kwargs)

def invert_with_image_iterator_para(vis, im, dopsf=False,
                               normalize=True, invert_function=invert_2d_base_para,
                               **kwargs) -> (image_for_para, numpy.ndarray):
    totalwt = numpy.zeros([1])
    result, sumwt = invert_function(vis, im, dopsf, normalize=False, **kwargs)
    totalwt = sumwt
    im.data[...] = result.data[...]
    if normalize:
        im = normalize_sumwt_para(im, totalwt)
    return im, totalwt

def normalize_sumwt_para(im: image_for_para, sumwt) -> image_for_para:
    if sumwt[0] > 0.0:
        im.data[:, :] /= sumwt[0]
    else:
        im.data[:, :] = 0.0
    return im


