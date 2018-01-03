import math
from arl_para.test.Constants import *
import numpy as np
from arl.data.data_models import Visibility, Image, GainTable

def float_equal(a, b, PRECISION):
    '''
        判断两个float数是否相等，
    :param a:
    :param b:
    :return:
    '''
    # print(math.fabs(a - b))
    if math.fabs(a - b) < math.pow(10, PRECISION):
        return True
    # nan 和 10^-20判断做相等
    elif (a < math.pow(1, -20) and math.isnan(b)) or (b < math.pow(1, -20) and math.isnan(a)) or (math.isnan(a) and math.isnan(b)):
        return True
    else:
        return False

def uvw_equal(a, b, PRECISION):
    '''
        判断两个uvw是否相等
    :param a:
    :param b:
    :return:
    '''
    for x,y in zip(a, b):
        if math.fabs(x - y) >= math.pow(10, PRECISION):
            return False

    return True

def complex_equal(a: np.complex_, b: np.complex_, PRECISION):
    '''
        判断两个complex数是否相等，虚部和实部之差均小于1e-7即认为相等
    :param a:
    :param b:
    :return:
    '''
    if (math.fabs(a.imag - b.imag) < math.pow(10, PRECISION)) and (math.fabs(a.real - b.real) < math.pow(10, PRECISION)):
        return True
    else:
        return False

def compute_baseline_index(ant1, ant2, numOfAnt):
    return int(numOfAnt * ant1 - (ant1 * ant1 + 3 * ant1) / 2 + ant2 - 1)

def visibility_right(a: Visibility, b: Visibility, precision=-5):
    '''
        验证两个Visibility是否完全相等
    :param a:
    :param b:
    :return:
    '''
    assert a.nvis == b.nvis, "two Visibilities' shape are different: %s and %s" %(a.nvis, b.nvis)
    assert a.polarisation_frame.type == b.polarisation_frame.type, "two Visibilities' polarisation_frame are different"
    for i in range(a.nvis):
        assert uvw_equal(a.data['uvw'][i], b.data['uvw'][i], precision), "uvw are different %s and %s localtion: %s" % (a.data['uvw'][i], b.data['uvw'][i], i)
        assert a.data['time'][i] == b.data['time'][i], "time are different"
        assert a.data['frequency'][i] == b.data['frequency'][i], "frequency are different %s and %s"% (a.data['frequency'][i], b.data['frequency'][i])
        assert a.data['channel_bandwidth'][i] == b.data['channel_bandwidth'][i], "channel_bandwidth are different"
        assert a.data['integration_time'][i] == b.data['integration_time'][i], "integration_time are different"
        assert a.data['antenna1'][i] == b.data['antenna1'][i], "antenna1 are different"
        assert a.data['antenna2'][i] == b.data['antenna2'][i], "antenna2 are different"
        if type(a.data['vis'][i][0]) == type(b.data['vis'][i][0]) and type(a.data['vis'][i][0]) == np.complex_:
            for j in range(a.polarisation_frame.npol):
                assert complex_equal(a.data['vis'][i][j], b.data['vis'][i][j], precision), "vis are different %s and %s" % (a.data['vis'][i][j], b.data['vis'][i][j])
        assert (a.data['weight'][i] == b.data['weight'][i]).all(), "weight are different"
        assert (a.data['imaging_weight'][i] == b.data['imaging_weight'][i]).all(), "imaging_weight are different"

    print("the two visibilities are the same")

def image_right(a:Image, b:Image, precision=-5):
    '''
    验证两个image是否完全相等
    :param a:
    :param b:
    :return:
    '''
    assert a.shape == b.shape, "two Images' shape are different %s and %s" % (a.shape, b.shape)
    ncha, npol, ny, nx = a.shape
    for i in range(ncha):
        for j in range(npol):
            for k in range(ny):
                for l in range(nx):
                    assert type(a.data[i,j,k,l]) == type(b.data[i,j,k,l]), "two images' data type are different %s and %s"%(type(a.data[i,j,k,l]), type(b.data[i,j,k,l]))

                    if type(a.data[i,j,k,l]) == np.float_ or type(a.data[i,j,k,l]) == np.int_:
                        assert float_equal(a.data[i,j,k,l], b.data[i,j,k,l], precision), "two Images' data are different: %s and %s, location: %s, %s, %s, %s" %(a.data[i,j,k,l],b.data[i,j,k,l], i, j, k, l)

                    elif type(a.data[i,j,k,l]) == np.complex_:
                        assert complex_equal(a.data[i,j,k,l], b.data[i,j,k,l], precision), "two Images' data are different: %s and %s" %(a.data[i,j,k,l], b.data[i,j,k,l])

    assert a.polarisation_frame.type == b.polarisation_frame.type, "two Images' polarisation_frame are different %s and %s" % (a.polarisation_frame.type, b.polarisation_frame.type)
    print("the two images are the same")

def gaintable_right(a: GainTable, b: GainTable, precision=-5):
    assert (a.frequency == b.frequency).all(), "frequency are different %s and %s" % (
    a.frequency, b.frequency)
    assert (a.time == b.time).all(), "time are different %s and %s" % (a.time, b.time)
    for i in range(a.gain.shape[0]):
        for j in range(a.nants):
            for k in range(a.nchan):
                for l in range(a.nrec):
                    for m in range(a.nrec):
                        assert complex_equal(a.gain[i, j, k, l, m],
                                             b.gain[i, j, k, l, m], precision), "gain are different %s and %s" % (
                        a.gain[i, j, k, l, m], b.gain[i, j, k, l, m])
                        assert float_equal(a.weight[i, j, k, l, m],
                                           b.weight[i, j, k, l, m], precision), "weight are different %s and %s" % (
                        a.weight[i, j, k, l, m], b.weight[i, j, k, l, m])
                        if j == 0:
                            assert float_equal(a.residual[i, k, l, m],
                                               b.residual[i, k, l, m], precision), "residual are different %s and %s" % (
                            a.residual, b.residual)


