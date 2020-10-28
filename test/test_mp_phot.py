__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
from random import gauss, seed, uniform

# External packages:
import pandas as pd
import pytest

# From external (EVD) package(s):

# TARGET TEST MODULE:
import mpc.mp_phot

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, "test", '$test_data_do_color')


def test_do_color():
    # Test with a single FITS file in each filter:
    # Assumes start(), and assess() have been run, that proper color_control.ini file is present,
    #     and that make_dfs() has been run so that df*.csv files are ready.
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    this_directory, mp_string, an_string = mpc.mp_phot.get_context()
    # log_file = open(mpc.mp_phot.LOG_FILENAME, mode='a')
    # mpc.mp_phot.write_color_control_ini_template(this_directory, log_file)
    # mpc.mp_phot.make_dfs()
    mpc.mp_phot.do_color(filters=('R', 'I'), target_color=('SR', 'SI'),
                         color_index_passbands=('SR', 'SI'))
    # assert True


def test_make_color_control_dict():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    context = mpc.mp_phot.get_context()
    this_directory, mp_string, an_string = context
    ccd = mpc.mp_phot.make_color_control_dict(this_directory)
    assert isinstance(ccd, dict)
    assert len(ccd) == 13
    assert ccd['omit comps'] == ('888', '22', '432')
    assert ccd['omit obs'] == ('12', '33', '522')
    assert ccd['omit images'] == ('MP_426-0012-R.fts', 'MP_426-0014-R.fts')
    assert ccd['min sr mag'] == 10
    assert ccd['max sr mag'] == 16
    assert ccd['max catalog dsr mmag'] == 20
    assert ccd['min sloan ri color'] == -0.4
    assert ccd['max sloan ri color'] == +0.8
    assert len(ccd['transforms']) == 2
    assert ccd['transforms'][('R', 'SR', 'SR', 'SI')] == tuple(['use', -0.15])
    assert ccd['transforms'][('I', 'SI', 'SR', 'SI')] == tuple(['fit'])
    assert len(ccd['extinctions']) == 2
    assert ccd['extinctions']['R'] == tuple(['fit'])
    assert ccd['extinctions']['I'] == tuple(['use', 0.11])
    assert ccd['fit vignette'] == True
    assert ccd['fit xy'] == False
    assert ccd['fit jd'] == False








# def test_make_comp_diagnostics():
#     seed(1234)
#     # Make test df_model:
#     n_comps, n_images = 8, 5
#     comps = ['comp' + str(i) for i in range(n_comps)]
#     catmag_r = [uniform(11, 15) for i in range(n_comps)]
#     catmag_i = [r + 0.2 + gauss(0, 0.01) for r in catmag_r]
#     offset = [0, 0.03, 0.07, -0.10] + ((n_comps - 4) * [0])
#     images = ['image' + str(i) for i in range(n_images)]
#     random_effect = [gauss(0, 0.05) for i in range(n_images)]  # image effect
#     airmass = [1.2 + 0.03 * i for i in range(n_images)]
#     transform = mp_phot.TRANSFORM_CLEAR_SR_SR_SI
#     state = mp_phot.get_session_state()
#     extinction = state['extinction']['Clear']
#     zero_point = -20.0
#
#     dict_list = []
#     for i_comp, comp in enumerate(comps):
#         for i_image, image in enumerate(images):
#             inst_mag = zero_point \
#                        + catmag_r[i_comp] \
#                        + transform * (catmag_r[i_comp] - catmag_i[i_comp]) \
#                        + extinction * airmass[i_image] \
#                        + offset[i_comp] \
#                        + random_effect[i_image] \
#                        + gauss(0, 0.01)
#             this_dict = {'SourceID': comp, 'FITSfile': image, 'Airmass': airmass[i_image],
#                          'r': catmag_r[i_comp], 'i': catmag_i[i_comp], 'InstMag': inst_mag}
#             dict_list.append(this_dict)
#     df_model = pd.DataFrame(data=dict_list)
#     df_model['Type'] = 'Comp'
#     df_model['Filter'] = 'Clear'
#     df_model['Serial'] = range(len(df_model))
#     df_model.index = list(df_model['Serial'])
#
#     # Call test fn to produce df_comp_diagnostics
#     df_comp_diagnostics = mp_phot.make_comp_diagnostics(df_model)
#     assert len(df_comp_diagnostics) == n_comps

