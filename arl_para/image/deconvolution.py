from arl_para.data.data_models import *
import copy
from arl_para.image.base import *
from typing import List
from arl.image.cleaners import *
import numpy as np
def sumfacet(dirty: List[image_for_para], psf: List[image_for_para], wcs, **kwargs):
    psf_support = get_parameter(kwargs, 'psf_support', None)
    if isinstance(psf_support, int):
        if (psf_support < psf[0].shape[0] // 2) and ((psf_support < psf[0].shape[1] // 2)):
            for i in range(len(psf)):
                centre = [psf[i].shape[0] // 2, psf[i].shape[1] // 2]
                psf[i].data = psf[i].data[(centre[0] - psf_support):(centre[0] + psf_support),
                           (centre[1] - psf_support):(centre[1] + psf_support)]

    nmoments = get_parameter(kwargs, "nmoments", 3)
    assert nmoments > 0, "Number of frequency moments must be greater than zero"
    npol = get_parameter(kwargs, "npol", 4)
    dirty_taylor = calculate_image_frequency_moments_para(dirty, wcs, nmoments=nmoments, npol=npol)
    psf_taylor = calculate_image_frequency_moments_para(psf, wcs, nmoments=2 * nmoments, npol=npol)

    return dirty_taylor, psf_taylor

def calculate_image_frequency_moments_para(im: List[image_for_para], wcs, reference_frequency=None, nmoments=3, npol=4) -> List[image_for_para]:
    ny, nx = im[0].shape
    nchan = len(im) // 4 # 此处为了便于和串行程序进行比较，故除以4
    # nchan = len(im)
    channels = numpy.arange(nchan)
    freq = wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]

    if reference_frequency is None:
        reference_frequency = numpy.average(freq)


    imgs = []
    for moment in range(nmoments):
        moment_data = numpy.zeros([ny, nx])
        for chan in range(nchan):
            weight = numpy.power((freq[chan] - reference_frequency) / reference_frequency, moment)
            # moment_data += im[chan].data * weight
            moment_data += im[chan * 4].data * weight # 此处为了便于和串行程序作比较故乘以4
        keys = copy.deepcopy(im[0].keys)
        keys["channel"] = moment
        imgs.append(image_for_para(moment_data, im[0].wcs.deepcopy(), keys))


    return imgs

def identify_component(dirty_taylor: List[image_for_para], psf_taylor: List[image_for_para], wcs, mscale, mx, my, mval,
                       scale_counts, scale_flux, m_model, smresidual, i, **kwargs):
    nx = get_parameter(kwargs, "nx", 256)
    ny = get_parameter(kwargs, "ny", 256)
    nchan = get_parameter(kwargs, "nchan", 5)

    gain = get_parameter(kwargs, 'gain', 0.7)
    assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
    thresh = get_parameter(kwargs, 'threshold', 0.0)
    assert thresh >= 0.0
    niter = get_parameter(kwargs, 'niter', 100)
    assert niter > 0
    scales = get_parameter(kwargs, 'scales', [0, 3, 10, 30])
    fracthresh = get_parameter(kwargs, 'fractional_threshold', 0.1)
    assert 0.0 < fracthresh < 1.0

    moments = len(dirty_taylor)

    dirty_taylor2 = numpy.empty([len(dirty_taylor), dirty_taylor[0].shape[0], dirty_taylor[0].shape[1]])
    psf_taylor2 = numpy.empty([len(psf_taylor), psf_taylor[0].shape[0], psf_taylor[0].shape[1]])
    for idx, img in enumerate(dirty_taylor):
        dirty_taylor2[idx] = img.data

    for idx, img in enumerate(psf_taylor):
        psf_taylor2[idx] = img.data

    if psf_taylor[0].data[:, :].max():
        dirty_taylor2 = numpy.empty([len(dirty_taylor), dirty_taylor[0].shape[0], dirty_taylor[0].shape[1]])
        psf_taylor2 = numpy.empty([len(psf_taylor), psf_taylor[0].shape[0], psf_taylor[0].shape[1]])
        for idx, img in enumerate(dirty_taylor):
            dirty_taylor2[idx] = img.data

        for idx, img in enumerate(psf_taylor):
            psf_taylor2[idx] = img.data

        nscales = len(scales)

        pmax = psf_taylor2.max()
        assert pmax > 0.0

        lpsf = psf_taylor2 / pmax
        ldirty = dirty_taylor2 / pmax

        nmoments, ny, nx = dirty_taylor2.shape
        assert psf_taylor2.shape[0] == 2 * nmoments

        scaleshape = [nscales, ldirty.shape[1], ldirty.shape[2]]
        scalestack = create_scalestack(scaleshape, scales, norm=True)

        ssmmpsf = calculate_scale_scale_moment_moment_psf(lpsf, scalestack)


        absolutethresh = max(thresh, fracthresh * numpy.fabs(smresidual[0, 0, :, :]).max())

        scale_counts[mscale] += 1
        scale_flux[mscale] += mval[0]

        peak = numpy.max(numpy.fabs(mval))
        if peak < absolutethresh:
            return True, m_model, pmax * smresidual[0, :, :, :]

        lhs, rhs = overlapIndices(ldirty[0, ...], psf_taylor2[0, ...], mx, my)

        m_model = update_moment_model(m_model, scalestack, lhs, rhs, gain, mscale, mval)
        smresidual = update_scale_moment_residual(smresidual, ssmmpsf, lhs, rhs, gain, mscale, mval)

        if i < niter - 1:
            return False, scale_counts, scale_flux, m_model, smresidual
        else:
            return True, m_model, pmax * smresidual[0, :, :, :]
    else:
        return True, numpy.zeros([moments, ny, nx]), numpy.zeros([moments, ny, nx])


def calculate_comp_residual(comp_array, residual_array, dirty_taylor, wcs, **kwargs):
    nx = get_parameter(kwargs, "nx", 256)
    ny = get_parameter(kwargs, "ny", 256)
    nchan = get_parameter(kwargs, "nchan", 5)

    return_moments = get_parameter(kwargs, "return_moments", False)
    if not return_moments:
        comp_image = image_for_para(comp_array, dirty_taylor[0].wcs, dirty_taylor[0].keys)
        residual_image = image_for_para(residual_array, dirty_taylor[0].wcs, dirty_taylor[0].keys)

        comp_image = calculate_image_from_frequency_moments_para(comp_image, wcs, nchan=nchan, ny=ny, nx=nx)
        residual_image = calculate_image_from_frequency_moments_para(residual_image, wcs, nchan=nchan, ny=ny, nx=nx)
        return comp_image, residual_image

    else:
        comp_images = []
        residual_images = []
        for i in range(comp_array.shape[0]):
            temp_img = image_for_para(comp_array[i], dirty_taylor[i].wcs, dirty_taylor[i].keys)
            comp_images.append(temp_img)
        for i in residual_array.shape[0]:
            temp_img = image_for_para(residual_array[i], dirty_taylor[i].wcs, dirty_taylor[i].keys)
            residual_images.append(temp_img)

        return comp_images, residual_images

def calculate_image_from_frequency_moments_para(moment_image, wcs, reference_frequency=None, nchan=5, ny=256, nx=256) -> List[image_for_para]:
    nmoments, mny, mnx = moment_image.shape

    assert ny == mny
    assert nx == mnx


    channels = numpy.arange(nchan)
    freq = wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]

    if reference_frequency is None:
        reference_frequency = numpy.average(freq)

    newims=[]


    for moment in range(nmoments):
        if moment == 0:
            for chan in range(nchan):
                keys = copy.deepcopy(moment_image.keys)
                keys["channel"] = chan
                newim = image_for_para(np.zeros([ny, nx]), moment_image.wcs, keys)
                newims.append(newim)
        for chan in range(nchan):
            weight = numpy.power((freq[chan] - reference_frequency) / reference_frequency, moment)
            newims[chan].data[...] += moment_image.data[moment, ...] * weight

    return newims

def subimacom(dirty_taylor: List[image_for_para], psf_taylor: List[image_for_para], wcs, scale_counts, scale_flux, m_model, smresidual, i, **kwargs):
    window = get_parameter(kwargs, 'window', None)
    nx = get_parameter(kwargs, "nx", 256)
    ny = get_parameter(kwargs, "ny", 256)
    nchan = get_parameter(kwargs, "nchan", 5)

    if window == 'quarter':
        qx = nx // 4
        qy = ny // 4
        window = numpy.zeros([nchan, ny, nx])
        window[..., (qy + 1):3 * qy, (qx + 1):3 * qx] = 1.0
    else:
        window = None

    findpeak = get_parameter(kwargs, "findpeak", 'ARL')
    gain = get_parameter(kwargs, 'gain', 0.7)
    assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
    thresh = get_parameter(kwargs, 'threshold', 0.0)
    assert thresh >= 0.0
    niter = get_parameter(kwargs, 'niter', 100)
    assert niter > 0
    scales = get_parameter(kwargs, 'scales', [0, 3, 10, 30])
    fracthresh = get_parameter(kwargs, 'fractional_threshold', 0.1)
    assert 0.0 < fracthresh < 1.0


    dirty_taylor2 = numpy.empty([len(dirty_taylor), dirty_taylor[0].shape[0], dirty_taylor[0].shape[1]])
    psf_taylor2 = numpy.empty([len(psf_taylor), psf_taylor[0].shape[0], psf_taylor[0].shape[1]])
    for idx, img in enumerate(dirty_taylor):
        dirty_taylor2[idx] = img.data

    for idx, img in enumerate(psf_taylor):
        psf_taylor2[idx] = img.data

    if psf_taylor[0].data[:, :].max():
        if i < niter:
            if m_model is None:
                m_model = numpy.zeros(dirty_taylor2.shape)

            nscales = len(scales)

            pmax = psf_taylor2.max()
            assert pmax > 0.0

            lpsf = psf_taylor2 / pmax
            ldirty = dirty_taylor2 / pmax

            nmoments, ny, nx = dirty_taylor2.shape
            assert psf_taylor2.shape[0] == 2 * nmoments

            scaleshape = [nscales, ldirty.shape[1], ldirty.shape[2]]
            scalestack = create_scalestack(scaleshape, scales, norm=True)

            if smresidual is None:
                smresidual = calculate_scale_moment_residual(ldirty, scalestack)

            if scale_counts is None:
                scale_counts = numpy.zeros(nscales, dtype='int')

            if scale_flux is None:
                scale_flux = numpy.zeros(nscales)

            ssmmpsf = calculate_scale_scale_moment_moment_psf(lpsf, scalestack)
            hsmmpsf, ihsmmpsf = calculate_scale_inverse_moment_moment_hessian(ssmmpsf)


            if window is None:
                windowstack = None
            else:
                windowstack = numpy.zeros_like(scalestack)
                windowstack[convolve_scalestack(scalestack, window) > 0.9] = 1.0

            mscale, mx, my, mval = find_global_optimum(hsmmpsf, ihsmmpsf, smresidual, windowstack, findpeak)

            return mscale, mx, my, mval, scale_counts, scale_flux, m_model, smresidual,

        else:
            return None, None, None, None, scale_counts, scale_flux, m_model, smresidual

    else:
        return None, None, None, None, None, None, None, None
