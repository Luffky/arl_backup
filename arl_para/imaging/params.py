from arl_para.data.data_models import *

# 生成卷积核的过程几乎未修改，仅适用于2d卷积，wprojection卷积未深入探究
def get_kernel_list_para(vis: visibility_for_para, im: image_for_para, facets, **kwargs):
    """Get the list of kernels, one per visibility

    """

    shape = im.shape
    npixel = shape[1]
    cellsize = numpy.pi * im.wcs.wcs.cdelt[1] / 180.0

    kernelname = get_parameter(kwargs, "kernel", "2d")
    oversampling = get_parameter(kwargs, "oversampling", 8)
    padding = get_parameter(kwargs, "padding", facets)

    gcf, _ = anti_aliasing_calculate((padding * npixel, padding * npixel), oversampling)

    wabsmax = numpy.max(numpy.abs(vis.w))
    if kernelname == 'wprojection' and wabsmax > 0.0:
        # 暂时不用wprojection
        pass
        # # wprojection needs a lot of commentary!
        # log.debug("get_kernel_list: Using wprojection kernel")
        #
        # # The field of view must be as padded! R_F is for reporting only so that
        # # need not be padded.
        # fov = cellsize * npixel * padding
        # r_f = (cellsize * npixel / 2) ** 2 / abs(cellsize)
        # log.debug("get_kernel_list: Fresnel number = %f" % (r_f))
        # delA = get_parameter(kwargs, 'wloss', 0.02)
        #
        # advice = advise_wide_field(vis, delA)
        # wstep = get_parameter(kwargs, "wstep", advice['w_sampling_primary_beam'])
        #
        # log.debug("get_kernel_list: Using w projection with wstep = %f" % (wstep))
        #
        # # Now calculate the maximum support for the w kernel
        # kernelwidth = get_parameter(kwargs, "kernelwidth",
        #                             (2 * int(round(numpy.sin(0.5 * fov) * npixel * wabsmax * cellsize))))
        # kernelwidth = max(kernelwidth, 8)
        # assert kernelwidth % 2 == 0
        # log.debug("get_kernel_list: Maximum w kernel full width = %d pixels" % (kernelwidth))
        # padded_shape = [im.shape[0], im.shape[1], im.shape[2] * padding, im.shape[3] * padding]
        #
        # remove_shift = get_parameter(kwargs, "remove_shift", True)
        # padded_image = pad_image(im, padded_shape)
        # kernel_list = w_kernel_list(vis, padded_image, oversampling=oversampling, wstep=wstep,
        #                             kernelwidth=kernelwidth, remove_shift=remove_shift)
    else:
        kernelname = '2d'
        kernel_list = standard_kernel_list(vis, (padding * npixel, padding * npixel),
                                           oversampling=oversampling)

    return kernelname, gcf, kernel_list

def get_uvw_map_para(vis: visibility_for_para, im: image_for_para, padding: int) :
    ny, nx = im.shape
    shape = (1, int(round(padding * ny)), int(round(padding * nx)))
    uvwscale = numpy.zeros([3])
    uvwscale[0:2] = im.wcs.wcs.cdelt[0:2] * numpy.pi / 180.0
    assert uvwscale[0] != 0.0, "Error in uv scaling"
    vuvwmap = uvwscale * vis.data['uvw']
    uvw_mode = "2d"

    return uvw_mode, shape, padding, vuvwmap

def anti_aliasing_calculate(shape, oversampling=1, support=3):
    """
    Compute the prolate spheroidal anti-aliasing function

    The kernel is to be used in gridding visibility data onto a grid on for degridding from a grid.
    The gridding correction function (gcf) is used to correct the image for decorrelation due to
    gridding.

    Return the 2D grid correction function (gcf), and the convolving kernel (kernel

    See VLA Scientific Memoranda 129, 131, 132
    :param shape: (height, width) pair
    :param oversampling: Number of sub-samples per grid pixel
    :param support: Support of kernel (in pixels) width is 2*support+2
    """

    # 2D Prolate spheroidal angular function is separable
    ny, nx = shape
    nu = numpy.abs(2.0 * coordinates(nx))
    gcf1d, _ = grdsf(nu)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]

    s1d = 2 * support + 2
    nu = numpy.arange(-support, +support, 1.0 / oversampling)
    kernel1d = grdsf(nu / support)[1]
    l1d = len(kernel1d)
    # Rearrange to get the convolution function isolated by (yf, xf). For this convolution function
    # the result is heavily redundant but it does fit well into the general framework
    kernel4d = numpy.zeros((oversampling, oversampling, s1d, s1d))
    for yf in range(oversampling):
        my = range(yf, l1d, oversampling)[::-1]
        for xf in range(oversampling):
            mx = range(xf, l1d, oversampling)[::-1]
            kernel4d[yf, xf, 2:, 2:] = numpy.outer(kernel1d[my], kernel1d[mx])
    return gcf, (kernel4d / numpy.sum(kernel4d[0,0,:,:])).astype('complex')

def coordinates(npixel: int) -> object:
    """ 1D array which spans [-.5,.5[ with 0 at position npixel/2

    """
    return (numpy.arange(npixel) - npixel // 2) / npixel

def grdsf(nu):
    """Calculate PSWF using an old SDE routine re-written in Python

    Find Spheroidal function with M = 6, alpha = 1 using the rational
    approximations discussed by Fred Schwab in 'Indirect Imaging'.
    This routine was checked against Fred's SPHFN routine, and agreed
    to about the 7th significant digit.
    The gridding function is (1-NU**2)*GRDSF(NU) where NU is the distance
    to the edge. The grid correction function is just 1/GRDSF(NU) where NU
    is now the distance to the edge of the image.
    """
    p = numpy.array([[8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
                     [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
    q = numpy.array([[1.0000000e0, 8.212018e-1, 2.078043e-1],
                     [1.0000000e0, 9.599102e-1, 2.918724e-1]])

    _, np = p.shape
    _, nq = q.shape

    nu = numpy.abs(nu)

    nuend = numpy.zeros_like(nu)
    part = numpy.zeros(len(nu), dtype='int')
    part[(nu >= 0.0) & (nu < 0.75)] = 0
    part[(nu > 0.75) & (nu < 1.0)] = 1
    nuend[(nu >= 0.0) & (nu <= 0.75)] = 0.75
    nuend[(nu > 0.75) & (nu < 1.0)] = 1.0

    delnusq = nu ** 2 - nuend ** 2

    top = p[part, 0]
    for k in range(1, np):
        top += p[part, k] * numpy.power(delnusq, k)

    bot = q[part, 0]
    for k in range(1, nq):
        bot += q[part, k] * numpy.power(delnusq, k)

    grdsf = numpy.zeros_like(nu)
    ok = (bot > 0.0)
    grdsf[ok] = top[ok] / bot[ok]
    ok = numpy.abs(nu > 1.0)
    grdsf[ok] = 0.0

    # Return the gridding function and the grid correction function
    return grdsf, (1 - nu ** 2) * grdsf

def standard_kernel_list(vis, shape, oversampling=8, support=3):
    """Return a generator to calculate the standard visibility kernel

    :param vis: visibility
    :param shape: tuple with 2D shape of grid
    :param oversampling: Oversampling factor
    :param support: Support of kernel
    :return: Function to look up gridding kernel
    """
    return numpy.zeros_like(vis.w, dtype='int'), [anti_aliasing_calculate(shape, oversampling, support)[1]]