from arl.imaging.base import predict_skycomponent_visibility
from arl.visibility.coalesce import decoalesce_visibility, coalesce_visibility
from arl.imaging.facets import predict_facets
from arl.calibration.solvers import solve_gaintable
import unittest
from arl.imaging.facets import *
from arl.util.testing_support import *
from arl.skycomponent.operations import *
from arl_para.visibility.base import *
from arl_para.test.Utils import *
from arl_para.image.base import *
from arl_para.skycomponent.operations import *
from arl_para.image.operations import *
from arl_para.imaging.convolution import *
from collections import defaultdict
from arl.visibility.base import *
from arl_para.solve.solve import *
from arl_para.fourier_transforms.fft_support import fft as arl_para_fft
from arl.fourier_transforms.fft_support import fft as arl_fft
from arl_para.imaging.invert import *

import numpy as np

lowcore = create_named_configuration('LOWBD2-CORE')
times = numpy.linspace(-3, +3, 5) * (numpy.pi / 12.0)  # time = 5
frequency = numpy.array([1e8, 1.1e8, 1.2e8, 1.3e8, 1.4e8])  # nchan = 5
channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])  # nchan = 5

f = numpy.array([100.0, 110.0, 120.0, 130.0])  # npol = 4
flux = numpy.array([f, f + 100.0, f + 200.0, f + 300.0, f + 400.0])  # nchan,npol = 2, 4

phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')
comp = create_skycomponent(flux=flux, frequency=frequency, direction=compabsdirection,
                           polarisation_frame=PolarisationFrame('linear'))  # flux [2,4]



class TestFunction(unittest.TestCase):
    # def test_visibility_base(self):
    #     blockvis = create_blockvisibility(lowcore, times=times, frequency=frequency,
    #                                              channel_bandwidth=channel_bandwidth,
    #                                              phasecentre=phasecentre, weight=1,
    #                                              polarisation_frame=PolarisationFrame('linear'),
    #                                              integration_time=1.0)
    #     visibility = coalesce_visibility(blockvis)
    #
    #     viss, vis_share = create_visibility_para(config=lowcore, times=times, frequency=frequency, channel_bandwidth=channel_bandwidth,
    #                                              phasecentre=phasecentre,weight=1.0, polarisation_frame=PolarisationFrame('linear'),
    #                                              integration_time=1.0)
    #
    #     back_visibility = visibility_para_to_visibility(viss, vis_share)
    #
    #     visibility_right(visibility, back_visibility)

    # def test_image_base(self):
    #     image = create_image(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                          polarisation_frame=PolarisationFrame('linear'),)
    #
    #     imgs, img_share = create_image_para(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                                         polarisation_frame=PolarisationFrame('linear'))
    #
    #     back_image = image_para_to_image(imgs, img_share)
    #
    #     image_right(image, back_image)

    # def test_insert_skycomponent(self):
    #     for method in ["Sinc", "Lanczos", "PSWF", "normal"]:
    #         imgs, img_share = create_image_para(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                                         polarisation_frame=PolarisationFrame('linear'))
    #         for img in imgs:
    #             insert_skycomponent_para(img, comp, insert_method=method)
    #
    #         back_image = image_para_to_image(imgs, img_share)
    #
    #
    #         image = create_image(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                                                       polarisation_frame=PolarisationFrame('linear'),)
    #         insert_skycomponent(image, comp, insert_method=method)
    #
    #         image_right(image, back_image)
    #         print("%s method test passed" % method)

    # def test_fft(self):
    #     # TODO 个人认为此处的fft不应该单单是直接对image的data进行fft，而是还有其他操作，使得切片后的数据不会发生改变
    #     # 可能是使用更高层的arl.image.operations.fft_image方法
    #     image = create_image(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                          polarisation_frame=PolarisationFrame('linear'), )
    #     insert_skycomponent(image, comp, insert_method="Sinc")
    #     image.data = arl_fft(image.data)
    #
    #     imgs, img_share = create_image_para(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                                         polarisation_frame=PolarisationFrame('linear'))
    #     for img in imgs:
    #         insert_skycomponent_para(img, comp, insert_method="Sinc")
    #         img[1].data = arl_para_fft(img[1].data)
    #
    #     back_image = image_para_to_image(imgs, img_share)
    #     image_right(image, back_image)


    # def test_reproject_image(self):
    #     # TODO 同上
    #     image = create_image(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                          polarisation_frame=PolarisationFrame('linear'), )
    #     insert_skycomponent(image, comp, insert_method="Sinc")
    #     newwcs, newshape = create_new_wcs_new_shape(image, image.shape)
    #     image = reproject_image(image, newwcs, newshape)[0]
    #
    #     imgs, img_share = create_image_para(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                                         polarisation_frame=PolarisationFrame('linear'))
    #     for img in imgs:
    #         insert_skycomponent_para(img, comp, insert_method="Sinc")
    #         newwcs, newshape = create_new_wcs_new_shape(img[1], img[1].shape)
    #         temp = reproject_image_para(img[1], newwcs, newshape)
    #         img[1].data = temp[0].data
    #         img[1].wcs = temp[0].wcs
    #
    #     back_image = image_para_to_image(imgs, img_share)
    #
    #     image_right(image, back_image)

    # def test_predict_facet_predict_visibility(self):
    #     image = create_image(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                          polarisation_frame=PolarisationFrame('linear'), )
    #     insert_skycomponent(image, comp, insert_method="Sinc")
    #
    #     blockvis = create_blockvisibility(lowcore, times=times, frequency=frequency,
    #                                       channel_bandwidth=channel_bandwidth,
    #                                       phasecentre=phasecentre, weight=1,
    #                                       polarisation_frame=PolarisationFrame('linear'),
    #                                       integration_time=1.0)
    #     visibility = coalesce_visibility(blockvis)
    #     visibility = predict_facets(visibility, image, facets=FACETS)
    #     newphasecentre = SkyCoord(ra=+10.0 * u.deg, dec=-30.0 * u.deg, frame='icrs', equinox='J2000')
    #     visibility = phaserotate_visibility(visibility, newphasecentre)
    #     predict_skycomponent_visibility(visibility, comp)
    #
    #
    #     # imgs, img_share = create_image_para(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #     #                                     polarisation_frame=PolarisationFrame('linear'))
    #     imgs = []
    #     for chan in range(NCHAN):
    #         for facet in range(FACETS):
    #             for pol in range(NPOL):
    #                 image_para = create_image_para_2(NY // FACETS, NX // FACETS, chan, pol, facet,
    #                                                  phasecentre,
    #                                                  cellsize=0.001, polarisation_frame=PolarisationFrame('linear'),
    #                                                  FACET=FACETS)
    #                 imgs.append(((chan, 0, facet, pol), image_para))
    #
    #
    #
    #     channel_list = defaultdict(list)
    #
    #     for img in imgs:
    #         chan = img[0][0]
    #         facet = img[0][2]
    #         pol = img[0][3]
    #         insert_skycomponent_para(img, comp, insert_method="Sinc")
    #         vis_para, _ = create_visibility_para(config=lowcore, times=times, frequency=frequency[chan:chan+1], channel_bandwidth=channel_bandwidth[chan:chan+1],
    #                                              phasecentre=phasecentre,weight=1.0, polarisation_frame=PolarisationFrame('linear'),
    #                                              integration_time=1.0, mode="1to1", keys={"chan": [chan]})
    #         vis_para = predict_facets_para(vis_para, img[1])
    #         channel_list[chan].append(vis_para)
    #
    #     vis_share = visibility_share(visibility, times.shape[0], frequency.shape[0], NAN)
    #
    #
    #     vis_list = []
    #     for chan in channel_list:
    #         sum_vis = sum_visibility_in_one_facet_pol(channel_list[chan])
    #         newphasecentre = SkyCoord(ra=+10.0 * u.deg, dec=-30.0 * u.deg, frame='icrs', equinox='J2000')
    #         phaserotate_vis = phaserotate_visibility_para(sum_vis, newphasecentre=newphasecentre)
    #         predict_skycomponent_visibility_para(phaserotate_vis, comp)
    #         vis_list.append(phaserotate_vis)
    #
    #     vis_share.phasecentre = vis_list[0].phasecentre
    #     back_visibility = visibility_para_to_visibility(vis_list, vis_share, mode="1to1")
    #
    #     visibility_right(visibility, back_visibility)

    # def test_solve_gaintable(self):
    #     image = create_image(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                          polarisation_frame=PolarisationFrame('linear'), )
    #     insert_skycomponent(image, comp, insert_method="Sinc")
    #
    #     blockvis = create_blockvisibility(lowcore, times=times, frequency=frequency,
    #                                       channel_bandwidth=channel_bandwidth,
    #                                       phasecentre=phasecentre, weight=1,
    #                                       polarisation_frame=PolarisationFrame('linear'),
    #                                       integration_time=1.0)
    #     telescope_data = coalesce_visibility(blockvis)
    #     model_vis = predict_facets(telescope_data, image, facets=FACETS)
    #     newphasecentre = SkyCoord(ra=+10.0 * u.deg, dec=-30.0 * u.deg, frame='icrs', equinox='J2000')
    #     model_vis = phaserotate_visibility(model_vis, newphasecentre)
    #     predict_skycomponent_visibility(model_vis, comp)
    #     model_vis = decoalesce_visibility(model_vis)
    #
    #     #模拟望远镜实际接收到的visibility
    #     blockvis_observed = create_blockvisibility(lowcore, times=times, frequency=frequency,
    #                                                channel_bandwidth=channel_bandwidth,
    #                                                phasecentre=phasecentre, weight=1,
    #                                                polarisation_frame=PolarisationFrame('linear'),
    #                                                integration_time=1.0)
    #     # 用整数填充vis， 模拟实际接收到的block_visibility
    #     vis_observed = coalesce_visibility(blockvis_observed)
    #     vis_observed.data['vis'].flat = range(vis_observed.nvis * 4)
    #     blockvis_observed = decoalesce_visibility(vis_observed)
    #
    #     gaintable = solve_gaintable(blockvis_observed, model_vis)
    #     apply_gaintable(blockvis_observed, gaintable)
    #     blockvis_observed.data['vis'] = blockvis_observed.data['vis'] - model_vis.data['vis']
    #     visibility = coalesce_visibility(blockvis_observed)
    #
    #     vis_share = visibility_share(visibility, times.shape[0], frequency.shape[0], NAN)
    #
    #
    #     imgs = []
    #     for chan in range(NCHAN):
    #         for facet in range(FACETS):
    #             for pol in range(NPOL):
    #                 image_para = create_image_para_2(NY // FACETS, NX // FACETS, chan, pol, facet,
    #                                                  phasecentre,
    #                                                  cellsize=0.001, polarisation_frame=PolarisationFrame('linear'),
    #                                                  FACET=FACETS)
    #                 imgs.append(((chan, 0, facet, pol), image_para))
    #
    #
    #
    #     model_list = defaultdict(list)
    #     vis_list = defaultdict(list)
    #     # 此处模拟visibility_buffer
    #     for chan in range(NCHAN):
    #         vis_para, _ = create_visibility_para(config=lowcore, times=times,
    #                                              frequency=frequency[chan:chan+1],
    #                                              channel_bandwidth=channel_bandwidth[chan:chan+ 1],
    #                                              phasecentre=phasecentre, weight=1.0,
    #                                              polarisation_frame=PolarisationFrame('linear'),
    #                                              integration_time=1.0, mode="1to1", keys={"chan": [chan]})
    #         vis_para.data['vis'] = copy.deepcopy(vis_observed.data['vis'][chan::5][:])
    #         vis_list[chan].append(vis_para)
    #
    #
    #
    #     for img in imgs:
    #         chan = img[0][0]
    #         facet = img[0][2]
    #         pol = img[0][3]
    #         insert_skycomponent_para(img, comp, insert_method="Sinc")
    #         vis_para, _ = create_visibility_para(config=lowcore, times=times, frequency=frequency[chan:chan+1], channel_bandwidth=channel_bandwidth[chan:chan+1],
    #                                              phasecentre=phasecentre,weight=1.0, polarisation_frame=PolarisationFrame('linear'),
    #                                              integration_time=1.0, mode="1to1", keys={"chan": [chan]})
    #
    #         vis_para = predict_facets_para(vis_para, img[1])
    #         model_list[chan].append(vis_para)
    #
    #
    #     modelvis_list = defaultdict(list)
    #     for chan in model_list:
    #         sum_vis = sum_visibility_in_one_facet_pol(model_list[chan])
    #         newphasecentre = SkyCoord(ra=+10.0 * u.deg, dec=-30.0 * u.deg, frame='icrs', equinox='J2000')
    #         phaserotate_vis = phaserotate_visibility_para(sum_vis, newphasecentre=newphasecentre)
    #         predict_skycomponent_visibility_para(phaserotate_vis, comp)
    #         modelvis_list[chan].append(phaserotate_vis)
    #
    #     gs = []
    #     for time in range(5):
    #         viss = []
    #         modelviss = []
    #         idxs = []
    #         idxs2 = []
    #         for chan in vis_list:
    #             viss.append(vis_list[chan][0])
    #             idxs.append(tuple([0,0,chan,0,0,0]))
    #             modelviss.append(modelvis_list[chan][0])
    #             idxs2.append(tuple([0,0,chan,0,0,0]))
    #         gt, x, xwt = solve_gaintable_para(viss, idxs, time, modelviss, idxs2)
    #         g = solve_from_X_para(gt, x, xwt, 4)
    #         gs.append(g)
    #     back_gaintable = gaintable_n_to_1(gs)
    #
    #     gaintable_right(gaintable, back_gaintable)
    #
    #     viss = []
    #     for chan in vis_list:
    #         vis = vis_list[chan][0]
    #         apply_gaintable_para(vis, back_gaintable, chan, iscopy=False)
    #         subtract_visibility(vis, modelvis_list[chan][0])
    #         viss.append(vis)
    #
    #     vis_share.phasecentre = viss[0].phasecentre
    #     back_visibility = visibility_para_to_visibility(viss, vis_share, mode="1to1")
    #
    #     visibility_right(visibility, back_visibility)

    # def test_invert_facet(self):
    #     image = create_image(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                          polarisation_frame=PolarisationFrame('linear'), )
    #     # reprojevt
    #     # fft
    #     insert_skycomponent(image, comp, insert_method="Sinc")
    #
    #     blockvis = create_blockvisibility(lowcore, times=times, frequency=frequency,
    #                                       channel_bandwidth=channel_bandwidth,
    #                                       phasecentre=phasecentre, weight=1,
    #                                       polarisation_frame=PolarisationFrame('linear'),
    #                                       integration_time=1.0)
    #     telescope_data = coalesce_visibility(blockvis)
    #     model_vis = predict_facets(telescope_data, image, facets=FACETS)
    #     newphasecentre = SkyCoord(ra=+10.0 * u.deg, dec=-30.0 * u.deg, frame='icrs', equinox='J2000')
    #     model_vis = phaserotate_visibility(model_vis, newphasecentre)
    #     predict_skycomponent_visibility(model_vis, comp)
    #     model_vis = decoalesce_visibility(model_vis)
    #
    #     # 模拟望远镜实际接收到的visibility
    #     blockvis_observed = create_blockvisibility(lowcore, times=times, frequency=frequency,
    #                                                channel_bandwidth=channel_bandwidth,
    #                                                phasecentre=phasecentre, weight=1,
    #                                                polarisation_frame=PolarisationFrame('linear'),
    #                                                integration_time=1.0)
    #     # 用整数填充vis， 模拟实际接收到的block_visibility
    #     vis_observed = coalesce_visibility(blockvis_observed)
    #     vis_observed.data['vis'].flat = range(vis_observed.nvis * 4)
    #     blockvis_observed = decoalesce_visibility(vis_observed)
    #
    #     gaintable = solve_gaintable(blockvis_observed, model_vis)
    #     apply_gaintable(blockvis_observed, gaintable)
    #     blockvis_observed.data['vis'] = blockvis_observed.data['vis'] - model_vis.data['vis']
    #     visibility = coalesce_visibility(blockvis_observed)
    #     # 空的image，接受visibility的invert
    #     image = create_image(NY, NX, frequency=frequency, phasecentre=phasecentre, cellsize=0.001,
    #                          polarisation_frame=PolarisationFrame('linear'))
    #     image, wt = invert_facets(visibility, image)
    #     #ifft
    #     #reprojet
    #
    #
    #     imgs = []
    #     for chan in range(NCHAN):
    #         for facet in range(FACETS):
    #             for pol in range(NPOL):
    #                 image_para = create_image_para_2(NY // FACETS, NX // FACETS, chan, pol, facet,
    #                                                  phasecentre,
    #                                                  cellsize=0.001, polarisation_frame=PolarisationFrame('linear'),
    #                                                  FACET=FACETS)
    #                 imgs.append(((chan, 0, facet, pol), image_para))
    #
    #     model_list = defaultdict(list)
    #     vis_list = defaultdict(list)
    #     # 此处模拟visibility_buffer
    #     for chan in range(NCHAN):
    #         vis_para, _ = create_visibility_para(config=lowcore, times=times,
    #                                              frequency=frequency[chan:chan + 1],
    #                                              channel_bandwidth=channel_bandwidth[chan:chan + 1],
    #                                              phasecentre=phasecentre, weight=1.0,
    #                                              polarisation_frame=PolarisationFrame('linear'),
    #                                              integration_time=1.0, mode="1to1", keys={"chan": [chan]})
    #         vis_para.data['vis'] = copy.deepcopy(vis_observed.data['vis'][chan::5][:])
    #         vis_list[chan].append(vis_para)
    #
    #     for img in imgs:
    #         chan = img[0][0]
    #         facet = img[0][2]
    #         pol = img[0][3]
    #         insert_skycomponent_para(img, comp, insert_method="Sinc")
    #         vis_para, _ = create_visibility_para(config=lowcore, times=times, frequency=frequency[chan:chan + 1],
    #                                              channel_bandwidth=channel_bandwidth[chan:chan + 1],
    #                                              phasecentre=phasecentre, weight=1.0,
    #                                              polarisation_frame=PolarisationFrame('linear'),
    #                                              integration_time=1.0, mode="1to1", keys={"chan": [chan]})
    #
    #         vis_para = predict_facets_para(vis_para, img[1])
    #         model_list[chan].append(vis_para)
    #
    #     modelvis_list = defaultdict(list)
    #     for chan in model_list:
    #         sum_vis = sum_visibility_in_one_facet_pol(model_list[chan])
    #         newphasecentre = SkyCoord(ra=+10.0 * u.deg, dec=-30.0 * u.deg, frame='icrs', equinox='J2000')
    #         phaserotate_vis = phaserotate_visibility_para(sum_vis, newphasecentre=newphasecentre)
    #         predict_skycomponent_visibility_para(phaserotate_vis, comp)
    #         modelvis_list[chan].append(phaserotate_vis)
    #
    #     gs = []
    #     for time in range(5):
    #         viss = []
    #         modelviss = []
    #         idxs = []
    #         idxs2 = []
    #         for chan in vis_list:
    #             viss.append(vis_list[chan][0])
    #             idxs.append(tuple([0, 0, chan, 0, 0, 0]))
    #             modelviss.append(modelvis_list[chan][0])
    #             idxs2.append(tuple([0, 0, chan, 0, 0, 0]))
    #         gt, x, xwt = solve_gaintable_para(viss, idxs, time, modelviss, idxs2)
    #         g = solve_from_X_para(gt, x, xwt, 4)
    #         gs.append(g)
    #     back_gaintable = gaintable_n_to_1(gs)
    #
    #     gaintable_right(gaintable, back_gaintable)
    #
    #     viss = {}
    #     for chan in vis_list:
    #         vis = vis_list[chan][0]
    #         apply_gaintable_para(vis, back_gaintable, chan, iscopy=False)
    #         subtract_visibility(vis, modelvis_list[chan][0])
    #         viss[chan] = vis
    #
    #     imgs = list()
    #     for facet in range(PIECE):
    #         for pol in range(NPOL):
    #             for chan in range(NCHAN):
    #                 image_para = create_image_para_2(NY // FACETS, NX // FACETS, chan, pol, facet,
    #                                                  phasecentre,
    #                                                  cellsize=0.001, polarisation_frame=PolarisationFrame('linear'),
    #                                                  FACET=FACETS)
    #                 image_para, wt = invert_facets_para(viss[chan], image_para)
    #                 imgs.append(((chan, facet, pol), image_para))
    #
    #     img_share = image_share(POLARISATION_FRAME, image.wcs, NCHAN, NPOL, NY, NX)
    #     back_image = image_para_to_image(imgs, img_share)
    #
    #     image_right(image, back_image)

    def test_identify_component(self):
        pass























































if __name__ == '__main__':
    unittest.main()
