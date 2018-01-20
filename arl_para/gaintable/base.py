import numpy
from arl.data.polarisation import *
from arl_para.data.data_models import *
import copy
from arl_para.test.Constants import *
from arl_para.test.Utils import compute_baseline_index

def create_gaintable_from_visibility_para(vis: Visibility, time_width: float = None,
                                          frequency_width: float = None, **kwargs) -> GainTable:

    utimes = np.unique(vis.time)
    uant = np.unique(vis.antenna1)
    ufrequency = np.unique(vis.frequency)
    ntimes = len(utimes)
    nants = len(uant) + 1
    nfrequency = len(ufrequency)

    POLARISATION_FRAME = get_parameter(kwargs, "polarisation_frame", PolarisationFrame('linear'))
    receptor_frame = ReceptorFrame(POLARISATION_FRAME.type)
    nrec = receptor_frame.nrec

    gainshape = [1, nants, nfrequency, nrec, nrec]
    gain = numpy.ones(gainshape, dtype='complex')
    if nrec > 1:
        gain[..., 0, 1] = 0.0
        gain[..., 1, 0] = 0.0

    gain_weight = numpy.ones(gainshape)
    gain_time = utimes
    gain_frequency = ufrequency
    gain_residual = numpy.zeros([1, nfrequency, nrec, nrec])

    gt = GainTable(gain=gain, time=gain_time, weight=gain_weight, residual=gain_residual, frequency=gain_frequency,
                   receptor_frame=receptor_frame)


    return gt




def gaintable_n_to_1(gains):
    gain = gains[0][1]
    gainshape = [len(gains), gain.nants, gain.nchan, gain.nrec, gain.nrec]
    gain_table = numpy.zeros(gainshape, dtype='complex')
    gain_weight = numpy.ones(gainshape)
    gain_time = numpy.zeros(len(gains))
    gain_frequency = gain.frequency
    gain_residual = numpy.zeros([len(gains), gain.nchan, gain.nrec, gain.nrec])
    for idx, g in gains:
       gain_table[idx] = g.gain
       gain_weight[idx] = g.weight
       gain_time[idx] = g.time[0]
       gain_residual[idx] = g.residual
    gt = GainTable(gain=gain_table, time=gain_time, weight=gain_weight, residual=gain_residual, frequency=gain_frequency,
                   receptor_frame=gain.receptor_frame)

    return gt

def vis_timeslice_iter(vis: visibility_for_para, **kwargs) -> numpy.ndarray:
    timemin = numpy.min(vis.time)
    timemax = numpy.max(vis.time)

    timeslice = get_parameter(kwargs, "timeslice", None)
    if timeslice is None or timeslice == 'auto':
        vis_slices = get_parameter(kwargs, "vis_slices", None)
        if vis_slices is None:
            vis_slices = len(numpy.unique(vis.time))
        boxes = numpy.linspace(timemin, timemax, vis_slices)
        timeslice = (timemax - timemin) / vis_slices
    else:
        vis_slices = 1 + 2 * numpy.round((timemax - timemin) / timeslice).astype('int')
        boxes = numpy.linspace(timemin, timemax, vis_slices)

    for box in boxes:
        rows = numpy.abs(vis.time - box) <= 0.5 * timeslice
        yield rows

def apply_gaintable_para(vis: visibility_for_para, gt:GainTable, chan, inverse=False, iscopy=True, **kwargs):
    for chunk, rows in enumerate(vis_timeslice_iter(vis)):
        vistime = numpy.average(vis.time[rows])
        integration_time = numpy.average(vis.integration_time[rows])
        gaintable_rows = abs(gt.time - vistime) < integration_time / 2.0

        # Lookup the gain for this set of visibilities
        gain = gt.data['gain'][gaintable_rows]


        # The shape of the mueller matrix is
        ntimes, nant, nchan, nrec, _ = gain.shape
        visnant = len(numpy.unique(vis.antenna1)) + 1

        original = vis.vis[rows]
        applied = copy.deepcopy(original)
        for time in range(ntimes):
            for a1 in range(visnant - 1):
                for a2 in range(a1 + 1, visnant):
                    if iscopy:
                        mueller = numpy.kron(gain[time, a1, chan//4 , :, :], numpy.conjugate(gain[time, a2, chan//4, :, :]))
                    else:
                        mueller = numpy.kron(gain[time, a1, chan, :, :],
                                             numpy.conjugate(gain[time, a2, chan, :, :]))
                    if inverse:
                        mueller = numpy.linalg.inv(mueller)
                    idx = time * visnant * (visnant - 1) // 2 + compute_baseline_index(a1, a2, visnant)
                    applied[idx] = numpy.matmul(mueller, original[idx])

        vis.data['vis'][rows] = applied







