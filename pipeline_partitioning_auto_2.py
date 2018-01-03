from pyspark import SparkContext, SparkConf, RDD
from pyspark import util
import numpy as np
from collections import defaultdict
import os
from arl.util.testing_support import *
from arl.image.deconvolution import deconvolve_cube
from arl.imaging.base import predict_skycomponent_visibility
from arl_para.image.base import *
from arl_para.visibility.base import *
from arl_para.test.Utils import *
from arl.skycomponent.operations import *
from arl.visibility.base import *
from arl_para.skycomponent.operations import *
from arl.calibration.solvers import *
from arl_para.solve.solve import *
from arl_para.imaging.invert import *
from arl.imaging.facets import *
from arl_para.test.Constants import *
import sys, getopt


os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"

def parse_console_arg():
    opts, args = getopt.getopt(sys.argv[1:], "a:p:f:c:t:")
    ret = {"nan":3, "nchan":5, "npix":256, "nfacet":2, "ntime":5}
    for op, value in opts:
        if op == "-a":
            ret["nan"] = int(value)
        elif op == "-p":
            ret['npix'] = int(value)
        elif op == "-f":
            ret['nfacet'] = int(value)
        elif op == "-t":
            ret['ntime'] = int(value)
        elif op == "-c":
            ret['nchan'] = int(value)
    return ret

scale = parse_console_arg()
print(scale)
metadata = MetaData(nan=scale["nan"], nchan=scale["nchan"], npix=scale["npix"], nfacet=scale["nfacet"], ntime=scale["ntime"])

def SDPPartitioner_pharp_alluxio(key):
	'''
		Partitioner_function
	'''
	return int(str(key).split(',')[2])

def SDPPartitioner(key):
	'''
		Partitioner_function
	'''
	return int(str(key).split(',')[4])

def MapPartitioner(partitions):
	def _inter(key):
		partition = partitions
		return partition[key]
	return _inter


def extract_lsm_handle():
	partitions = defaultdict(int)
	partition = 0
	initset = []
	beam = 0
	major_loop = 0
	partitions[(beam, major_loop)] = partition
	partition += 1
	initset.append(((beam, major_loop), ()))
	partitioner = MapPartitioner(partitions)
	return sc.parallelize(initset).partitionBy(len(partitions), partitioner).mapPartitions(extract_lsm_kernel, True)

def local_sky_model_handle():
	partitions = defaultdict(int)
	partition = 0
	initset = []
	partitions[()] = partition
	partition += 1
	initset.append(((), ()))
	partitioner = MapPartitioner(partitions)
	return sc.parallelize(initset).partitionBy(len(partitions), partitioner).mapPartitions(local_sky_model_kernel, True)

def telescope_management_handle():
	partitions = defaultdict(int)
	partition = 0
	initset = []
	partitions[()] = partition
	partition += 1
	initset.append(((), ()))
	partitioner = MapPartitioner(partitions)
	return sc.parallelize(initset).partitionBy(len(partitions), partitioner).mapPartitions(telescope_management_kernel, True)

def visibility_buffer_handle():
	initset = []
	beam = 0
	for frequency in range(0, metadata.NCHAN * 4):
		time = 0
		baseline = 0
		polarisation = 0
		initset.append((beam, frequency, time, baseline, polarisation))
	return sc.parallelize(initset).map(lambda x: visibility_buffer_kernel(x))

def telescope_data_handle(telescope_management):
	partitions = defaultdict(int)
	partition = 0
	dep_telescope_management = defaultdict(list)
	beam = 0
	frequency = 0
	time = 0
	baseline = 0
	partitions[(beam, frequency, time, baseline)] = partition
	partition += 1
	dep_telescope_management[()] = [(beam, frequency, time, baseline)]
	input_telescope_management = telescope_management.flatMap(lambda ix_data: map(lambda x: (x, ix_data[1]),dep_telescope_management[ix_data[0]]))
	partitioner = MapPartitioner(partitions)
	return input_telescope_management.partitionBy(len(partitions), partitioner).mapPartitions(telescope_data_kernel, True)

def reppre_ifft_handle(broadcast_lsm):
	initset = []
	dep_extract_lsm = defaultdict(list)
	beam = 0
	major_loop = 0
	for frequency in range(0, metadata.NCHAN):
		time = 0
		for facet in range(0, metadata.PIECE):
			for polarisation in range(0, metadata.NPOL):
				initset.append((beam, major_loop, frequency, time, facet, polarisation))

	return sc.parallelize(initset).map(lambda ix: reppre_ifft_kernel((ix, broadcast_lsm)))

def degrid_handle(reppre_ifft, broads_input_telescope_data):
	return reppre_ifft.flatMap(lambda ix: degrid_kernel((ix, broads_input_telescope_data)))

def pharotpre_dft_sumvis_handle(degrid, broadcast_lsm):
	return degrid.partitionBy(metadata.NCHAN * 4, SDPPartitioner_pharp_alluxio).mapPartitions(lambda ix: pharotpre_dft_sumvis_kernel((ix, broadcast_lsm)))

def timeslots_handle(broads_input0, broads_input1):
	initset = []
	beam = 0
	for time in range(0, metadata.NTIMES):
		frequency = 0
		baseline = 0
		polarisation = 0
		major_loop = 0
		initset.append((beam, major_loop, frequency, time, baseline, polarisation))

	return sc.parallelize(initset, metadata.NTIMES).map(lambda ix: timeslots_kernel((ix, broads_input0, broads_input1)))

def solve_handle(timeslots):
	dep_timeslots = defaultdict(list)
	beam = 0
	major_loop = 0
	baseline = 0
	frequency = 0
	for time in range(0, metadata.NTIMES):
		polarisation = 0
		dep_timeslots[(beam, major_loop, frequency, time, baseline, polarisation)] = (beam, major_loop, frequency, time, baseline, polarisation)
	return timeslots.map(solve_kernel)

def cor_subvis_flag_handle(broads_input0, broads_input1, broads_input2):
	initset = []
	beam = 0
	for frequency in range(0, 4 * metadata.NCHAN):
		time = 0
		baseline = 0
		polarisation = 0
		major_loop = 0
		initset.append((beam, major_loop, frequency, time, baseline, polarisation))
	return sc.parallelize(initset, metadata.NCHAN * 4).map(lambda ix: cor_subvis_flag_kernel((ix, broads_input0, broads_input1, broads_input2)))

def grikerupd_pharot_grid_fft_rep_handle(broads_input_telescope_data, broads_input):
	initset = []
	beam = 0
	frequency = 0
	for facet in range(0, metadata.PIECE):
		for polarisation in range(0, metadata.NPOL):
			time = 0
			major_loop = 0
			initset.append((beam, major_loop, frequency, time, facet, polarisation))
	return sc.parallelize(initset).map(lambda ix: grikerupd_pharot_grid_fft_rep_kernel((ix, broads_input_telescope_data, broads_input)))

def sum_facets_handle(grikerupd_pharot_grid_fft_rep):
	initset = []
	beam = 0
	frequency = 0
	for facet in range(0, metadata.PIECE):
		for polarisation in range(0, metadata.NPOL):
			time = 0
			major_loop = 0
			initset.append((beam, major_loop, frequency, time, facet, polarisation))
	return grikerupd_pharot_grid_fft_rep.map(sum_facets_kernel)

def identify_component_handle(sum_facets):
	partitions = defaultdict(int)
	partition = 0
	dep_sum_facets = defaultdict(list)
	beam = 0
	major_loop = 0
	frequency = 0
	for facet in range(0, metadata.PIECE):
		partitions[(beam, major_loop, frequency, facet)] = partition
		partition += 1
		for i_polarisation in range(0, metadata.NPOL):
			dep_sum_facets[(beam, major_loop, frequency, 0, facet, i_polarisation)] = [(beam, major_loop, frequency, facet)]
	return sum_facets.partitionBy(metadata.PIECE, SDPPartitioner).mapPartitions(identify_component_kernel_partitions)

def source_find_handle(identify_component):
    partitions = defaultdict(int)
    partition = 0
    dep_identify_component = defaultdict(list)
    beam = 0
    major_loop = 0
    partitions[(beam, major_loop)] = partition
    partition += 1
    for i_facet in range(0, metadata.PIECE):
        dep_identify_component[(beam, major_loop, 0, i_facet)] = [(beam, major_loop)]
    input_identify_component = identify_component.flatMap(lambda ix_data: map(lambda x: (x, ix_data[1]), dep_identify_component[ix_data[0]]))
    partitioner = MapPartitioner(partitions)
    return input_identify_component.partitionBy(len(partitions), partitioner).mapPartitions(source_find_kernel, True)

def subimacom_handle(sum_facets, identify_component):
	partitions = defaultdict(int)
	partition = 0
	dep_identify_component = defaultdict(list)
	dep_sum_facets = defaultdict(list)
	beam = 0
	major_loop = 0
	frequency = 0
	for facet in range(0, metadata.PIECE):
		partitions[(beam, major_loop, frequency, facet)] = partition
		partition += 1
		for polarisation in range(0, metadata.NPOL):
			dep_sum_facets[(beam, major_loop, frequency, 0, facet, polarisation)] = (beam, major_loop, frequency, facet)
		dep_identify_component[(beam, major_loop, frequency, facet)] = (beam, major_loop, frequency, facet)
	input_identify_component = identify_component.flatMap(lambda ix_data: map(lambda x: (x, ix_data[1]), dep_identify_component[ix_data[0]]))
	input_sum_facets = identify_component.flatMap(lambda ix_data: map(lambda  x: (x, ix_data[1]), dep_sum_facets[ix_data[0]]))
	partitioner = MapPartitioner(partitions)
	return input_identify_component.partitionBy(len(partitions), partitioner).cogroup(input_sum_facets.partitionBy(len(partitions), partitioner)).mapPartitions(subimacom_kernel)

def update_lsm_handle(local_sky_model, source_find):
	partitions = defaultdict(int)
	partition = 0
	dep_local_sky_model = defaultdict(list)
	dep_source_find = defaultdict(list)
	beam = 0
	major_loop = 0
	partitions[(beam, major_loop)] = partition
	partition += 1
	dep_local_sky_model[()] = [(beam, major_loop)]
	dep_source_find[(beam, major_loop)] = [(beam, major_loop)]
	input_local_sky_model = local_sky_model.flatMap(lambda ix_data: map(lambda x: (x, ix_data[1]), dep_local_sky_model[ix_data[0]]))
	input_source_find = source_find.flatMap(lambda ix_data: map(lambda x: (x, ix_data[1]), dep_source_find[ix_data[0]]))
	partitioner = MapPartitioner(partitions)
	# print 100*'-'
	# print input_source_find.cache().collect()
	# print input_local_sky_model.cache().collect()
	# print input_local_sky_model.partitionBy(len(partitions), partitioner).cogroup(input_source_find.partitionBy(len(partitions), partitioner)).collect()
	# print 100*'-'
	return input_local_sky_model.partitionBy(len(partitions), partitioner).cogroup(input_source_find.partitionBy(len(partitions), partitioner)).mapPartitions(update_lsm_kernel, True)

# kernel函数
def extract_lsm_kernel(ixs):
    '''
    	生成skycomponent(s)
    :param ixs: key
    :return: iter[(key, skycoponent)]
    '''
    result = []
    for ix in ixs: #每次循环生成一个skycomponent
        comp = metadata.create_skycomponent()
        result.append((ix, comp))
    label = "Ectract_LSM (0.0M MB, 0.00 Tflop) " + str(ix)
    print(label + str(result))
    return iter(result)

def local_sky_model_kernel(ixs):
	Hash = 0
	input_size = 0
	ix = next(ixs)[0]
	label = "Local Sky Model (0.0 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (Hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 0)), int)
	result[0] = Hash
	return iter([(ix, result)])

def telescope_management_kernel(ixs):
	'''
    	生成总的conf类，留待telescope_data_kernel进一步划分
    :param ixs:
    :return: iter[(key, conf)]
    '''
	ix = next(ixs)[0]
	conf = create_named_configuration('LOWBD2-CORE')
	result = (ix, conf)
	label = "Telescope Management (0.0 MB, 0.00 Tflop) "
	print(label + str(result))
	return iter([result])

def visibility_buffer_kernel(ixs):
    '''
        按frequency生成list[visibility_buffer]
    :param ixs:
    :return:
    '''
    beam, chan, time, baseline, polarisation = ixs
    ixs = (beam, 0, chan, time, baseline, polarisation)

    # 此处模拟从sdp的上一步传入的visibility, 该visibility中的vis应该是有值的，而不是0
    lowcore = create_named_configuration('LOWBD2-CORE')
    times = metadata.create_time()  # time = 5
    frequency = metadata.create_frequency() # nchan = 5
    channel_bandwidth = metadata.create_channel_bandwidth() # nchan = 5
    phasecentre = metadata.create_phasecentre()
    vis_para, _ = create_visibility_para(config=lowcore, times=times, frequency=frequency[chan//4:chan//4 + 1],
                                     channel_bandwidth=channel_bandwidth[chan//4:chan//4 + 1],
                                     phasecentre=phasecentre, weight=1.0,
                                     polarisation_frame=metadata.create_polarisation_frame(),
                                     integration_time=1.0, mode="1to1", keys={"chan": [chan//4]}, NAN=metadata.NAN)
    # 模拟望远镜实际接收到的visibility
    blockvis_observed = create_blockvisibility(lowcore, times=times, frequency=frequency,
                                               channel_bandwidth=channel_bandwidth,
                                               phasecentre=phasecentre, weight=1,
                                               polarisation_frame=metadata.create_polarisation_frame(),
                                               integration_time=1.0, NAN=metadata.NAN)
    # 用整数填充vis， 模拟实际接收到的block_visibility
    vis_observed = coalesce_visibility(blockvis_observed)
    vis_observed.data['vis'].flat = range(vis_observed.nvis * 4)
    # 用自然数值填充vis，模拟实际接收到的block_visibility，并且各个vis的值各不相同易于区分
    vis_para.data['vis'] = copy.deepcopy(vis_observed.data['vis'][chan//4::metadata.NCHAN][:])

    result = (ixs, vis_para)
    label = "Visibility Buffer (546937.1 MB, 0.00 Tflop) "
    print(label + str(result))
    return result

def telescope_data_kernel(ixs):
    '''
		分割visibility类为visibility_para
	:param ixs:
    :return: iter[(key, visibility_para)]
    '''
    result = []
    for data in ixs:
        ix, conf = data
        times = metadata.create_time()  # time = 5
        frequency = metadata.create_frequency()  # nchan = 5
        channel_bandwidth = metadata.create_channel_bandwidth()  # nchan = 5
        phasecentre = metadata.create_phasecentre()
        result.append((ix, (conf, times, frequency, channel_bandwidth, phasecentre)))
        label = "Telescope Data (0.0 MB, 0.00 Tflop) "
    return iter(result)

def reppre_ifft_kernel(ixs):
	'''

	:param ixs: (reppre(key), skycomponent(value)
	:return: (key, image_for_para)
	'''
	reppre, data_extract_lsm = ixs
	ix = reppre
	###生成空的image数据============###
	frequency = metadata.create_frequency() # nchan = 5
	channel_bandwidth = metadata.create_channel_bandwidth()  # nchan = 5
	phasecentre = metadata.create_phasecentre()
	beam, major_loop, channel, time, facet, polarisation = ix
	image_para = create_image_para_2(metadata.NY//metadata.FACETS, metadata.NX//metadata.FACETS, channel, polarisation, facet, phasecentre,
									 cellsize=0.001, polarisation_frame=metadata.create_polarisation_frame(), FACET=metadata.FACETS)

	for dix, comp in data_extract_lsm.value:
		insert_skycomponent_para(image_para, comp, insert_method="Sinc")
        # 暂时注释掉，便于检查之后的步骤是否正确
		# newwcs, newshape = create_new_wcs_new_shape(image_para, image_para.shape)
		# image_para = reproject_image_para(image_para, new_wcs, newshape)[0]
		# image_para.data = fft(image_para.data)
	result = (ix, image_para)
	label = "Reprojection Predict + IFFT (14645.6 MB, 2.56 Tflop) "
	print(label + str(result))
	return result

def degrid_kernel(ixs):
	data_reppre_ifft, data_telescope_data = ixs
	iix, image = data_reppre_ifft
	ix = iix
	result = []
	beam, major_loop, chan, time, facet, polarisation = ix
	cix, (conf, times, frequency, channel_bandwidth, phasecentre) = data_telescope_data.value[0]
	# 创建新的空的visibility
	vis_para, _ = create_visibility_para(config=conf, times=times, frequency=frequency[chan:chan + 1],
										 channel_bandwidth=channel_bandwidth[chan:chan + 1],
										 phasecentre=phasecentre, weight=1.0,
										 polarisation_frame=metadata.create_polarisation_frame(),
										 integration_time=1.0, mode="1to1", keys={"chan": [chan]}, NAN=metadata.NAN)
	result = predict_facets_para(vis_para, image, FACETS=metadata.FACETS)
	label = "Degridding Kernel Update + Degrid (674.8 MB, 0.59 Tflop) "
	# 复制四份
	mylist = np.empty(4, list)
	temp1 = chan * 4
	mylist[0] = ((beam, major_loop, temp1, time, facet, polarisation), copy.deepcopy(result))
	temp2 = chan * 4 + 1
	mylist[1] = ((beam, major_loop, temp2, time, facet, polarisation), copy.deepcopy(result))
	temp3 = chan * 4 + 2
	mylist[2] = ((beam, major_loop, temp3, time, facet, polarisation), copy.deepcopy(result))
	temp4 = chan * 4 + 3
	mylist[3] = ((beam, major_loop, temp4, time, facet, polarisation), copy.deepcopy(result))
	print(label + str(mylist[0]))
	return (mylist)

def pharotpre_dft_sumvis_kernel(ixs):
    data, data_extract_lsm = ixs
    data_degrid = []
    for item in data:
        data_degrid.append(item)
    sum_vis = sum_visibility_in_one_facet_pol(data_degrid)
    # 不确定这一步是否需要还要再一步phaserotate，因为predict_facet已经在最后调用过一次phaserotate
    newphasecentre = SkyCoord(ra=+10.0 * u.deg, dec=-30.0 * u.deg, frame='icrs', equinox='J2000')
    phaserotate_vis = phaserotate_visibility_para(sum_vis, newphasecentre=newphasecentre)
    ix = (data_degrid[0][0][0], data_degrid[0][0][1], data_degrid[0][0][2], data_degrid[0][0][3], 0, 0)
    result = (ix, predict_skycomponent_visibility_para(phaserotate_vis, np.array(data_extract_lsm.value)[:, 1]))
    label = "Phase Rotation Predict + DFT + Sum visibilities (546937.1 MB, 512.53 Tflop) "
    print(label + str(result))
    return iter([result])

def timeslots_kernel(ixs):
    ix, data_pharotpre_dft_sumvis, data_visibility_buffer = ixs
    viss = []
    modelviss = []
    idxs = []
    idxs2 = []
    for (idx, vis), (idx2, model_vis) in zip(data_visibility_buffer.value, data_pharotpre_dft_sumvis.value):
        viss.append(vis)
        idxs.append(idx)
        modelviss.append(model_vis)
        idxs2.append(idx2)
    gt, x, xwt = solve_gaintable_para(viss, idxs, ix[3], modelviss, idxs2, polarisation_frame=metadata.create_polarisation_frame())
    label = "Timeslots (1518.3 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
    result = (ix, (gt, x, xwt))
    print(label + str(ix))
    return result

def solve_kernel(ixs):
    dix, (gt, x, xwt) = ixs
    ix = dix
    result = (ix, solve_from_X_para(gt, x, xwt, metadata.NPOL, precision=metadata.PRECISION))
    label = "Solve (8262.8 MB, 16.63 Tflop) " + str(ix).replace(" ", "")
    print(label + str(result))
    return result

def cor_subvis_flag_kernel(ixs):
    ix, data_pharotpre_dft_sumvis, data_visibility_buffer, data_solve = ixs
    gs = []
    for idx, gt in data_solve.value:
        gs.append((idx[3], gt))
    gaintable = gaintable_n_to_1(gs)
    v = None
    model_v = None
    for (idx, vis), (idx2, model_vis) in zip(data_visibility_buffer.value, data_pharotpre_dft_sumvis.value) :
        if idx[2] == ix[2]:
            v = vis
        if idx2[2] == ix[2]:
            model_v = model_vis
    apply_gaintable_para(v, gaintable, ix[2])
    result = (ix, subtract_visibility(v, model_v))
    label = "Correct + Subtract Visibility + Flag (153534.1 MB, 4.08 Tflop) " + str(ix).replace(" ", "")
    print(label + str(result))
    return result

def grikerupd_pharot_grid_fft_rep_kernel(ixs):
    ix, data_telescope_data, data_cor_subvis_flag = ixs
    input_size = 0
    beam, major_loop, chan, time, facet, pol = ix
    cix, (conf, times, frequency, channel_bandwidth, phasecentre) = data_telescope_data.value[0]
    imgs = []
    for idx, vis in data_cor_subvis_flag.value:
        chan = idx[2]
        image_para = create_image_para_2(metadata.NY // metadata.FACETS, metadata.NX // metadata.FACETS, chan, pol, facet, phasecentre, cellsize=0.001,
                                         polarisation_frame=metadata.create_polarisation_frame(), FACET=metadata.FACETS)
        image_para, wt = invert_facets_para(vis, image_para, FACETS=metadata.FACETS)
        imgs.append(image_para)
    result = (ix, imgs)
    label = "Gridding Kernel Update + Phase Rotation + Grid + FFT + Reprojection (14644.9 MB, 20.06 Tflop) " + str(ix).replace(" ", "")
    print(label + str(result))
    return result

def sum_facets_kernel(ixs):
    input_size = 0
    ix, data = ixs
    result = None
    for img in data:
        if result == None:
            result = img
        else:
            result.data += img.data
    reuslt = (ix, result)
    label = "Sum Facets (14644.9 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
    print(label + str(result))
    return result

def identify_component_kernel_partitions(ixs):
	input_size = 0
	ix = (0, 0, 0, 0, 0, 0)
	for dix, data in ixs:
		ix = dix
	label = "Identify Component (0.2 MB, 1830.61 Tflop) " + str(ix).replace(" ", "")
	print(label + " from " + str(input_size / 1000000) + " MB input)")
	return iter([((ix[0], ix[1], ix[2], ix[4]), result)])

def source_find_kernel(ixs):
	Hash = 0
	input_size = 0
	ix = (0, 0)
	for dix, data in ixs:
		Hash ^= data[0]
		input_size += data.shape[0]
		ix = dix
	label = "Source Find (5.8 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 19200)), int)
	result[0] = Hash
	return iter([(ix, result)])

def subimacom_kernel(ixs):
	Hash = 0
	input_size = 0
	ix = (0, 0, 0, 0)
	# for temp in ixs:
	# 	ix, (data_identify_component, data_sum_facets) = temp
	# 	for data in data_identify_component:
	# 		Hash ^= data[0]
	# 		input_size += 1
	# 	for data in data_sum_facets:
	# 		Hash ^= data[0]
	# 		input_size += 1

	label = "Subtract Image Component (73224.4 MB, 67.14 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 244081369)), int)
	result[0] = Hash
	return iter([(ix, result)])

def update_lsm_kernel(ixs):
	Hash = 0
	input_size = 0
	ix = (0, 0)
	for temp in ixs:

		ix, (data_local_sky_mode, data_source_find) = temp
		for data in data_local_sky_mode:
			Hash ^= data[0]
			input_size += 1

		for data in data_source_find:
			Hash ^= data[0]
			input_size += 1

	label = "Update LSM (0.0 MB, 0.00 Tflop) " + str(ix).replace(" ", "")
	Hash ^= hash(label)
	print(label + " (hash " + hex(Hash) + " from " + str(input_size / 1000000) + " MB input)")
	result = np.zeros(max(1, int(scale_data * 0)), int)
	result[0] = Hash
	return iter([(ix, result)])

scale_data = 0
scale_compute = 0

def serialize_program():
    result = []
    lowcore = create_named_configuration('LOWBD2-CORE')
    times = metadata.create_time()  # time = 5
    frequency = metadata.create_frequency() # nchan = 5
    channel_bandwidth = metadata.create_channel_bandwidth()  # nchan = 5
	#---predict_module---#
    phasecentre = metadata.create_phasecentre()
    comp = metadata.create_skycomponent()

    image = create_image(metadata.NY, metadata.NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
                         polarisation_frame=metadata.create_polarisation_frame(), )
    insert_skycomponent(image, comp, insert_method="Sinc")
    blockvis = create_blockvisibility(lowcore, times=times, frequency=frequency,
                                      channel_bandwidth=channel_bandwidth,
                                      phasecentre=phasecentre, weight=1,
                                      polarisation_frame=metadata.create_polarisation_frame(),
                                      integration_time=1.0, NAN=metadata.NAN)
    visibility = coalesce_visibility(blockvis)
    visibility = predict_facets(visibility, image, facets=metadata.FACETS)
    # 不确定这一步是否需要还要再一步phaserotate，因为predict_facet已经在最后调用过一次phaserotate
    newphasecentre = SkyCoord(ra=+10.0 * u.deg, dec=-30.0 * u.deg, frame='icrs', equinox='J2000')
    model_vis = phaserotate_visibility(visibility, newphasecentre)
    predict_skycomponent_visibility(model_vis, comp)
    result.append(model_vis)
    model_vis = decoalesce_visibility(model_vis)
	#---solve_module---#
    # 模拟望远镜实际接收到的visibility
    blockvis_observed = create_blockvisibility(lowcore, times=times, frequency=frequency,
                                               channel_bandwidth=channel_bandwidth,
                                               phasecentre=phasecentre, weight=1,
                                               polarisation_frame=metadata.create_polarisation_frame(),
                                           integration_time=1.0, NAN=metadata.NAN)
    # 用整数填充vis， 模拟实际接收到的block_visibility
    vis_observed = coalesce_visibility(blockvis_observed)
    vis_observed.data['vis'].flat = range(vis_observed.nvis * 4)
    blockvis_observed = decoalesce_visibility(vis_observed)

    gaintable = solve_gaintable(blockvis_observed, model_vis)
    apply_gaintable(blockvis_observed, gaintable)
    blockvis_observed.data['vis'] = blockvis_observed.data['vis'] - model_vis.data['vis']
    visibility = coalesce_visibility(blockvis_observed)  # 空的image，接受visibility的invert
    result.append(visibility)
	#---backwards_module---#
    image = create_image(metadata.NY, metadata.NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
                         polarisation_frame=metadata.create_polarisation_frame())
    image, wt = invert_facets(visibility, image, facets=metadata.FACETS)

    dirty_image = Image()
    dirty_image.wcs = image.wcs
    dirty_image.polarisation_frame = image.polarisation_frame
    dirty_image.data = np.sum(image.data, axis=0).reshape((1, metadata.NPOL, metadata.NY, metadata.NX)) * 4
    result.append(dirty_image)

    #---deconvolution_module---#
    psf_image = create_image(metadata.NY, metadata.NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
                         polarisation_frame=metadata.create_polarisation_frame())
    psf_image, psf_wt = invert_facets(visibility, psf_image, dopsf=True, facets=metadata.FACETS)

    comp_image, residual_image = deconvolve_cube(image, psf_image, algorithm="mfsmsclean")
    result.append((comp_image, residual_image))


    return result

def create_vis_share():
    vis_share = visibility_share(None, metadata.NTIMES, metadata.NCHAN, metadata.NAN)
    vis_share.configuration = create_named_configuration('LOWBD2-CORE')
    vis_share.polarisation_frame = PolarisationFrame('linear')
    vis_share.nvis = metadata.NTIMES * metadata.NCHAN * metadata.NBASE
    vis_share.npol = PolarisationFrame('linear').npol
    return vis_share


if __name__ == '__main__':
    # conf = SparkConf().set("spark.eventLog.enabled", "true").set("spark.eventLog.dir","/home/hadoop/arl_temp/uilog")
    conf = SparkConf().setMaster("local[4]").setAppName("io")
    sc = SparkContext(conf=conf)
    sc.addFile("./data/configurations/LOWBD2-CORE.csv")
    result = serialize_program()
    # === Extract Lsm ===
    extract_lsm = extract_lsm_handle()
    broadcast_lsm = sc.broadcast(extract_lsm.collect())
    # === Local Sky Model ===
    local_sky_model = local_sky_model_handle()
    # === Telescope Management ===
    telescope_management = telescope_management_handle()
    # # === Visibility Buffer ===
    visibility_buffer = visibility_buffer_handle()
    visibility_buffer.cache()
    broads_input1 = sc.broadcast(visibility_buffer.collect())
    # === reppre_ifft ===
    reppre_ifft = reppre_ifft_handle(broadcast_lsm)
    reppre_ifft.cache()
    # === Telescope Data ===
    telescope_data = telescope_data_handle(telescope_management)
    broads_input_telescope_data = sc.broadcast(telescope_data.collect())
    # # === degrid ===
    degrid = degrid_handle(reppre_ifft, broads_input_telescope_data)
    degrid.cache()
    # === pharotpre_dft_sumvis ===
    pharotpre_dft_sumvis = pharotpre_dft_sumvis_handle(degrid, broadcast_lsm)
    pharotpre_dft_sumvis.cache()
    broads_input0 = sc.broadcast(pharotpre_dft_sumvis.collect())
    # 验证predict module的正确性
    # phase_vis = pharotpre_dft_sumvis.collect()
    # vis_share = create_vis_share()
    # vis_share.phasecentre = phase_vis[0][1].phasecentre
    # back_visibility = visibility_para_to_visibility(phase_vis, vis_share, mode="1to1")
    # visibility_right(result[0], back_visibility)

    #  === Timeslots ===
    timeslots = timeslots_handle(broads_input0, broads_input1)
    timeslots.cache()
    # === solve ===
    solve = solve_handle(timeslots)
    solve.cache()
    broads_input2 = sc.broadcast(solve.collect())
    # === correct + Subtract Visibility + Flag ===
    cor_subvis_flag = cor_subvis_flag_handle(broads_input0, broads_input1, broads_input2)
    cor_subvis_flag.cache()
    broads_input = sc.broadcast(cor_subvis_flag.collect())
    # 验证solve module的正确性
    # subtract_vis = cor_subvis_flag.collect()
    # vis_share = create_vis_share()
    # vis_share.phasecentre = subtract_vis[0][1].phasecentre
    # back_visibility = visibility_para_to_visibility(subtract_vis, vis_share, mode="1to1")
    # visibility_right(result[1], back_visibility)



    # === Gridding Kernel Update + Phase Rotation + Grid + FFT + Rreprojection ===
    grikerupd_pharot_grid_fft_rep = grikerupd_pharot_grid_fft_rep_handle(broads_input_telescope_data, broads_input)
    grikerupd_pharot_grid_fft_rep.cache()
    # ===Sum Facets ===
    sum_facets = sum_facets_handle(grikerupd_pharot_grid_fft_rep)
    sum_facets.cache()
    # 验证sum module的正确性
    sum_image = sum_facets.collect()
    img_share = image_share(metadata.POLARISATION_FRAME, result[2].wcs, 1, metadata.NPOL, metadata.NY, metadata.NX)
    back_image = image_para_to_image(sum_image, img_share)
    image_right(result[2], back_image, precision=metadata.PRECISION)



	# === Identify Component ===
	# identify_component = identify_component_handle(sum_facets)
	# # === Source Find ===
	# source_find = source_find_handle(identify_component)
	# # === Substract Image Component ===
	# subimacom = subimacom_handle(sum_facets, identify_component)

	# # === Update LSM ===
	# update_lsm = update_lsm_handle(local_sky_model, source_find)

	# === Terminate ===
	# print("Finishing...")
	# #print("Subtract Image Component: %d" % subimacom.count())
	# print("Update LSM: %d" % update_lsm.count())



