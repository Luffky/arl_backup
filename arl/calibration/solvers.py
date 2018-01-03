""" Functions to solve for antenna/station gain

This uses an iterative substitution algorithm due to Larry D'Addario c 1980'ish. Used
in the original VLA Dec-10 Antsol.


For example::

    gtsol = solve_gaintable(vis, originalvis, phase_only=True, niter=niter, crosspol=False, tol=1e-6)
    vis = apply_gaintable(vis, gtsol, inverse=True)
 

"""

import logging

import numpy

from arl.calibration.operations import create_gaintable_from_blockvisibility
from arl.data.data_models import GainTable, BlockVisibility, assert_vis_gt_compatible
from arl_para.test.Constants import *
from arl.visibility.iterators import vis_timeslice_iter
from arl.visibility.operations import divide_visibility
from arl.visibility.base import create_visibility_from_rows
from arl.visibility.coalesce import *

log = logging.getLogger(__name__)


def solve_gaintable(vis: BlockVisibility, modelvis: BlockVisibility=None, phase_only=True, niter=30, tol=1e-8,
                    crosspol=False, **kwargs) -> GainTable:
    """Solve a gain table by fitting an observed visibility to a model visibility
    
    If modelvis is None, a point source model is assumed.

    :param vis: BlockVisibility containing the observed data
    :param modelvis: BlockVisibility containing the visibility predicted by a model
    :param phase_only: Solve only for the phases (default=True)
    :param niter: Number of iterations (default 30)
    :param tol: Iteration stops when the fractional change in the gain solution is below this tolerance
    :param crosspol: Do solutions including cross polarisations i.e. XY, YX or RL, LR
    :return: GainTable containing solution

    """
    assert type(vis) is BlockVisibility, "vis is not a BlockVisibility: %r" % vis
    assert type(modelvis) is BlockVisibility or type(modelvis) is None, "modelvis is not None or a " \
                                                                            "BlockVisibility: %r" % vis
    
    if phase_only:
        log.info('solve_gaintable: Solving for phase only')
    else:
        log.info('solve_gaintable: Solving for complex gain')

    # 根据blockvisibility的元数据初始化一个gaintable
    gt = create_gaintable_from_blockvisibility(vis)
    for chunk, rows in enumerate(vis_timeslice_iter(vis, **kwargs)): # 对visibility按照time切片
        # 切片好的visibility shape: [1,nant,nant,nchan,npol]
        subvis = create_visibility_from_rows(vis, rows)
        #若存在model
        if modelvis is not None:
            # model_visibility也要以相同的方式按time切片
            model_subvis = create_visibility_from_rows(modelvis, rows)
            # 两个vis相除计算得到新的block_vis，其中的元数据未改变，只有vis和weight发生了改变
            pointvis = divide_visibility(subvis, model_subvis)
            # 此处因为第0个axis是time轴，并且vis已经按照time分片，所以按照第0axis做了average以后值不发生变化
            x = numpy.average(pointvis.vis, axis=0)
            xwt = numpy.average(pointvis.weight, axis=0)

        else:
            x = numpy.average(subvis.vis, axis=0)
            xwt = numpy.average(subvis.weight, axis=0)

        # 将数据填入gaintable中
        # solve 和 timeslot的分界线
        gt = solve_from_X(gt, x, xwt, chunk, crosspol, niter, phase_only,
                        tol, npol=vis.polarisation_frame.npol)


    assert type(gt) is GainTable, "gt is not a GainTable: %r" % gt
    
    assert_vis_gt_compatible(vis, gt)

    
    return gt


def solve_from_X(gt: GainTable, x: numpy.ndarray, xwt: numpy.ndarray, chunk, crosspol, niter, phase_only, tol, npol) \
        -> GainTable:
    """ Solve for gains from the point source equivalents

    :param x: point source visibility
    :param xwt: point source weight
    :param chunk: which chunk of the gaintable?
    :param crosspol:
    :param niter:
    :param phase_only:
    :param tol:
    :param npol:
    :return:
    """
    gainshape = gt.data['gain'][chunk, ...].shape
    if npol > 1:
        if crosspol:
            gt.data['gain'][chunk, ...], gt.data['weight'][chunk, ...], gt.data['residual'][chunk, ...] = \
                solve_antenna_gains_itsubs_matrix(gainshape, x, xwt, phase_only=phase_only, niter=niter,
                                                  tol=tol)
        else:
            gt.data['gain'][chunk, ...], gt.data['weight'][chunk, ...], gt.data['residual'][chunk, ...] = \
                solve_antenna_gains_itsubs_vector(gainshape, x, xwt, phase_only=phase_only, niter=niter,
                                                  tol=tol)
    
    else:
        gt.data['gain'][chunk, ...], gt.data['weight'][chunk, ...], gt.data['residual'][chunk, ...] = \
            solve_antenna_gains_itsubs_scalar(gainshape, x, xwt, phase_only=phase_only, niter=niter,
                                              tol=tol)
    return gt


def solve_antenna_gains_itsubs_scalar(gainshape, x, xwt, niter=30, tol=1e-8, phase_only=True, refant=0):
    """Solve for the antenna gains

    x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    This uses an iterative substitution algorithm due to Larry D'Addario c 1980'ish. Used
    in the original VLA Dec-10 Antsol.

    :param gainshape: Shape of output gains
    :param x: Equivalent point source visibility[nants, nants, ...]
    :param xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :return: gain [nants, ...], weight [nants, ...]
    """
    
    nants = x.shape[0]
    for ant1 in range(nants):
        x[ant1, ant1, ...] = 0.0
        xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            x[ant1, ant2, ...] = numpy.conjugate(x[ant2, ant1, ...])
            xwt[ant1, ant2, ...] = xwt[ant2, ant1, ...]
    
    gain = numpy.ones(shape=gainshape, dtype=x.dtype)
    gwt = numpy.zeros(shape=gainshape, dtype=xwt.dtype)
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_scalar(gain, x, xwt)
        mask = numpy.abs(gain) > 0.0
        if phase_only:
            gain[mask] = gain[mask] / numpy.abs(gain[mask])
        angles = numpy.angle(gain)
        gain *= numpy.exp(-1j * angles)[refant, ...]
        gain = 0.5 * (gain + gainLast)
        change = numpy.max(numpy.abs(gain - gainLast))
        if change < tol:
            return gain, gwt, solution_residual_scalar(gain, x, xwt)
    
    return gain, gwt, solution_residual_scalar(gain, x, xwt)


def gain_substitution_scalar(gain, x, xwt):
    nants, nchan, nrec, _ = gain.shape
    newgain = numpy.ones_like(gain, dtype='complex')
    gwt = numpy.zeros_like(gain, dtype='float')
    
    # We are going to work with Jones 2x2 matrix formalism so everything has to be
    # converted to that format
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    for ant1 in range(nants):
        for chan in range(nchan):
            # Loop over e.g. 'RR', 'LL, or 'xx', 'YY' ignoring cross terms
            top = numpy.sum(x[:, ant1, chan, 0, 0] \
                            * gain[:, chan, 0, 0] * xwt[:, ant1, chan, 0, 0], axis=0)
            bot = numpy.sum((gain[:, chan, 0, 0] * numpy.conjugate(gain[:, chan, 0, 0])
                             * xwt[:, ant1, chan, 0, 0]).real, axis=0)
            
            if bot > 0.0:
                newgain[ant1, chan, 0, 0] = top / bot
                gwt[ant1, chan, 0, 0] = bot
            else:
                newgain[ant1, chan, 0, 0] = 0.0
                gwt[ant1, chan, 0, 0] = 0.0
    return newgain, gwt


def solve_antenna_gains_itsubs_vector(gainshape, x, xwt, niter=30, tol=1e-8, phase_only=True, refant=0):
    """Solve for the antenna gains using full matrix expressions

    x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    See Appendix D, section D.1 in:
    
    J. P. Hamaker, “Understanding radio polarimetry - IV. The full-coherency analogue of
    scalar self-calibration: Self-alignment, dynamic range and polarimetric fidelity,” Astronomy
    and Astrophysics Supplement Series, vol. 143, no. 3, pp. 515–534, May 2000.

    :param gainshape: Shape of output gains
    :param x: Equivalent point source visibility[nants, nants, ...]
    :param xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :return: gain [nants, ...], weight [nants, ...]
    """
    
    nants, _, nchan, npol = x.shape
    assert npol == 4
    newshape = (nants, nants, nchan, 2, 2)
    x = x.reshape(newshape)
    xwt = xwt.reshape(newshape)
    # print(x)
    # print(100*'-')
    # print(xwt)
    for ant1 in range(nants):
        x[ant1, ant1, ...] = 0.0
        xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            x[ant1, ant2, ...] = numpy.conjugate(x[ant2, ant1, ...])
            xwt[ant1, ant2, ...] = xwt[ant2, ant1, ...]
    # print(100 * '-')
    # print(x)
    # print(100 * '-')
    # print(xwt)
    gain = numpy.ones(shape=gainshape, dtype=x.dtype)
    gain[..., 0, 1] = 0.0
    gain[..., 1, 0] = 0.0
    gwt = numpy.zeros(shape=gainshape, dtype=xwt.dtype)
    # 做迭代
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_vector(gain, x, xwt)
        for rec in [0, 1]:
            gain[..., rec, 1 - rec] = 0.0
            if phase_only:
                gain[..., rec, rec] = gain[..., rec, rec] / numpy.abs(gain[..., rec, rec])
            gain[..., rec, rec] *= numpy.conjugate(gain[refant, ..., rec, rec]) / numpy.abs(gain[refant, ..., rec, rec])
        change = numpy.max(numpy.abs(gain - gainLast))
        gain = 0.5 * (gain + gainLast)
        # 若改变的值过小，则跳出迭代直接返回结果和残差,因为拆分的缘故，将会导致误差
        # if change < tol:
        #     return gain, gwt, solution_residual_vector(gain, x, xwt)
    # 消除因为浮点计算而产生的误差
    PRECISION=-5
    low = pow(1, PRECISION)
    gain[abs(gain.imag)<low] = gain[abs(gain.imag)<low].real
    return gain, gwt, solution_residual_vector(gain, x, xwt)


def gain_substitution_vector(gain, x, xwt):
    nants, nchan, nrec, _ = gain.shape
    newgain = numpy.ones_like(gain, dtype='complex')
    if nrec > 0:
        newgain[..., 0, 1] = 0.0
        newgain[..., 1, 0] = 0.0
    
    gwt = numpy.zeros_like(gain, dtype='float')
    
    # We are going to work with Jones 2x2 matrix formalism so everything has to be
    # converted to that format
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    if nrec > 0:
        gain[..., 0, 1] = 0.0
        gain[..., 1, 0] = 0.0
    
    for ant1 in range(nants):
        for chan in range(nchan):
            # Loop over e.g. 'RR', 'LL, or 'xx', 'YY' ignoring cross terms
            for rec in range(nrec):
                top = numpy.sum(x[:, ant1, chan, rec, rec] \
                                * gain[:, chan, rec, rec] * xwt[:, ant1, chan, rec, rec], axis=0)
                bot = numpy.sum((gain[:, chan, rec, rec] * numpy.conjugate(gain[:, chan, rec, rec])
                                 * xwt[:, ant1, chan, rec, rec]).real, axis=0)
                
                if bot > 0.0:
                    newgain[ant1, chan, rec, rec] = top / bot
                    gwt[ant1, chan, rec, rec] = bot
                else:
                    newgain[ant1, chan, rec, rec] = 0.0
                    gwt[ant1, chan, rec, rec] = 0.0
    
    return newgain, gwt


def solve_antenna_gains_itsubs_matrix(gainshape, x, xwt, niter=30, tol=1e-8, phase_only=True, refant=0):
    """Solve for the antenna gains using full matrix expressions

    x(antenna2, antenna1) = gain(antenna1) conj(gain(antenna2))

    See Appendix D, section D.1 in:

    J. P. Hamaker, “Understanding radio polarimetry - IV. The full-coherency analogue of
    scalar self-calibration: Self-alignment, dynamic range and polarimetric fidelity,” Astronomy
    and Astrophysics Supplement Series, vol. 143, no. 3, pp. 515–534, May 2000.

    :param gainshape: Shape of gaintable
    :param x: Equivalent point source visibility[nants, nants, ...]
    :param xwt: Equivalent point source weight [nants, nants, ...]
    :param niter: Number of iterations
    :param tol: tolerance on solution change
    :param phase_only: Do solution for only the phase? (default True)
    :param refant: Reference antenna for phase (default=0.0)
    :return: gain [nants, ...], weight [nants, ...]
    """
    
    nants, _, nchan, npol = x.shape
    assert npol == 4
    newshape = (nants, nants, nchan, 2, 2)
    x = x.reshape(newshape)
    xwt = xwt.reshape(newshape)
    
    for ant1 in range(nants):
        x[ant1, ant1, ...] = 0.0
        xwt[ant1, ant1, ...] = 0.0
        for ant2 in range(ant1 + 1, nants):
            x[ant1, ant2, ...] = numpy.conjugate(x[ant2, ant1, ...])
            xwt[ant1, ant2, ...] = xwt[ant2, ant1, ...]
    
    gain = numpy.ones(shape=gainshape, dtype=x.dtype)
    gain[..., 0, 1] = 0.0
    gain[..., 1, 0] = 0.0
    gwt = numpy.zeros(shape=gainshape, dtype=xwt.dtype)
    
    for iter in range(niter):
        gainLast = gain
        gain, gwt = gain_substitution_matrix(gain, x, xwt)
        if phase_only:
            gain = gain / numpy.abs(gain)
        change = numpy.max(numpy.abs(gain - gainLast))
        gain = 0.5 * (gain + gainLast)
        if change < tol:
            return gain, gwt, solution_residual_matrix(gain, x, xwt)
    
    return gain, gwt, solution_residual_matrix(gain, x, xwt)


def gain_substitution_matrix(gain, x, xwt):
    nants, nchan, nrec, _ = gain.shape
    newgain = numpy.ones_like(gain, dtype='complex')
    gwt = numpy.zeros_like(gain, dtype='float')
    
    # We are going to work with Jones 2x2 matrix formalism so everything has to be
    # converted to that format
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    # Write these loops out explicitly. Derivation of these vector equations is tedious but they are
    # structurally identical to the scalar case with the following changes
    # Vis -> 2x2 coherency vector, g-> 2x2 Jones matrix, *-> matmul, conjugate->Hermitean transpose (.H)
    for ant1 in range(nants):
        for chan in range(nchan):
            top = 0.0
            bot = 0.0
            for ant2 in range(nants):
                if ant1 != ant2:
                    xmat = x[ant2, ant1, chan]
                    xwtmat = xwt[ant2, ant1, chan]
                    g2 = gain[ant2, chan]
                    top += xmat * xwtmat * g2
                    bot += numpy.conjugate(g2) * xwtmat * g2
            newgain[ant1, chan][bot > 0.0] = top[bot > 0.0] / bot[bot > 0.0]
            newgain[ant1, chan][bot <= 0.0] = 0.0
            gwt[ant1, chan] = bot.real
    return newgain, gwt


def solution_residual_scalar(gain, x, xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities
    
    :param gain: gain [nant, ...]
    :param x: Point source equivalent visibility [nant, ...]
    :param xwt: Point source equivalent weight [nant, ...]
    :return: residual[...]
    """
    
    nants, nchan, nrec, _ = gain.shape
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    
    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])
    
    for ant1 in range(nants):
        for ant2 in range(nants):
            for chan in range(nchan):
                error = x[ant2, ant1, chan, 0, 0] - \
                        gain[ant1, chan, 0, 0] * numpy.conjugate(gain[ant2, chan, 0, 0])
                residual += (error * xwt[ant2, ant1, chan, 0, 0] * numpy.conjugate(error)).real
                sumwt += xwt[ant2, ant1, chan, 0, 0]
    
    residual[sumwt > 0.0] = numpy.sqrt(residual[sumwt > 0.0] / sumwt[sumwt > 0.0])
    residual[sumwt <= 0.0] = 0.0
        
    return residual


def solution_residual_vector(gain, x, xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities
    
    Vector case i.e. off-diagonals of gains are zero

    :param gain: gain [nant, ...]
    :param x: Point source equivalent visibility [nant, ...]
    :param xwt: Point source equivalent weight [nant, ...]
    :return: residual[...]
    """
    
    nants, nchan, nrec, _ = gain.shape
    x = x.reshape(nants, nants, nchan, nrec, nrec)
    x[..., 1, 0] = 0.0
    x[..., 0, 1] = 0.0
    
    xwt = xwt.reshape(nants, nants, nchan, nrec, nrec)
    xwt[..., 1, 0] = 0.0
    xwt[..., 0, 1] = 0.0
    
    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])
    for ant1 in range(nants):
        for ant2 in range(nants):
            for chan in range(nchan):
                for rec in range(nrec):
                    # print(chan, ant1, ant2, rec)
                    error = x[ant2, ant1, chan, rec, rec] - \
                            gain[ant1, chan, rec, rec] * numpy.conjugate(gain[ant2, chan, rec, rec])
                    residual += (error * xwt[ant2, ant1, chan, rec, rec] * numpy.conjugate(error)).real
                    sumwt += xwt[ant2, ant1, chan, rec, rec]

    residual[sumwt > 0.0] = numpy.sqrt(residual[sumwt > 0.0] / sumwt[sumwt > 0.0])
    residual[sumwt <= 0.0] = 0.0

    return residual


def solution_residual_matrix(gain, x, xwt):
    """Calculate residual across all baselines of gain for point source equivalent visibilities

    :param gain: gain [nant, ...]
    :param x: Point source equivalent visibility [nant, ...]
    :param xwt: Point source equivalent weight [nant, ...]
    :return: residual[...]
    """
    
    nants, _, nchan, nrec, _ = x.shape
    
    residual = numpy.zeros([nchan, nrec, nrec])
    sumwt = numpy.zeros([nchan, nrec, nrec])
    
    # This is written out in long winded form but should be optimised for
    # production code!
    for ant1 in range(nants):
        for ant2 in range(nants):
            for chan in range(nchan):
                for rec1 in range(nrec):
                    for rec2 in range(nrec):
                        error = x[ant2, ant1, chan, rec2, rec1] - \
                                gain[ant1, chan, rec2, rec1] * numpy.conjugate(gain[ant2, chan, rec2, rec1])
                        residual[chan, rec2, rec1] += (error * xwt[ant2, ant1, chan, rec2, rec1] * numpy.conjugate(
                            error)).real
                        sumwt[chan, rec2, rec1] += xwt[ant2, ant1, chan, rec2, rec1]
    
    residual[sumwt > 0.0] = numpy.sqrt(residual[sumwt > 0.0] / sumwt[sumwt > 0.0])
    residual[sumwt <= 0.0] = 0.0
    return residual
