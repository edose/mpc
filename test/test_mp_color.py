__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
from random import gauss, seed, uniform

# External packages:
import numpy as np
import pandas as pd
import pytest

# Other EVD modules:
import mpc.mp_phot
import mpc.ini

# TARGET TEST MODULE:
import mpc.mp_color

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, "test", '$test_data_mp_color')


def XXX_test_do_color_2filters():
    # Test with a single FITS file in each filter:
    # Assumes start(), and assess() have been run, that proper color_control.ini file is present,
    #     and that make_dfs() has been run so that df*.csv files are ready.
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    this_directory, mp_string, an_string = mpc.mp_phot.get_context()
    # log_file = open(mpc.mp_phot.LOG_FILENAME, mode='a')
    # mpc.mp_phot._write_color_control_ini_template(this_directory, log_file)
    # mpc.mp_phot.make_dfs()
    mpc.mp_color.do_one_color(definition=('R', 'I', 'SR', 'SI'), catalog_color_index=('SR', 'SI'))
    assert 1 == 1


def test_make_color_control_dict():
    mpc.mp_color.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    defaults_dict = mpc.ini.make_defaults_dict()
    color_log_filename = defaults_dict['color log filename']
    context = mpc.mp_phot.get_context(color_log_filename)
    this_directory, mp_string, an_string = context
    ccd = mpc.mp_color.make_color_control_dict(this_directory, defaults_dict)

    assert isinstance(ccd, dict)
    assert len(ccd) == 19

    assert ccd['mp location filenames'] == ('MP_426-0011-R.fts', 'MP_426-0012-I.fts')
    assert ccd['x pixels'] == (1034.4, 1036.0)
    assert ccd['y pixels'] == (454.3, 455.4)

    assert ccd['max mp obs mag uncertainty'] == 0.05
    assert ccd['max comp obs mag uncertainty'] == 0.025
    assert ccd['min sr mag'] == 10
    assert ccd['max sr mag'] == 16
    assert ccd['max catalog dsr mmag'] == 20
    assert ccd['min sloan ri color'] == -0.4
    assert ccd['max sloan ri color'] == +0.8
    assert ccd['omit comps'] == ('99999', '111111')
    assert ccd['omit obs'] == ('888888', '888889', '999996')
    assert ccd['omit images'] == ('MP_426-0016-R.fts', 'MP_426-0014-R.fts')
    assert ccd['min valid comps per image'] == 6

    assert len(ccd['transforms']) == 2
    assert ccd['transforms'][('R', 'SR', 'SR', 'SI')] == tuple(['use', -0.15])
    assert ccd['transforms'][('I', 'SI', 'SR', 'SI')] == tuple(['fit'])
    assert len(ccd['extinctions']) == 2
    assert ccd['extinctions']['R'] == tuple(['fit'])
    assert ccd['extinctions']['I'] == tuple(['use', 0.11])
    assert ccd['fit vignette'] == True
    assert ccd['fit xy'] == False
    assert ccd['fit jd'] == False


_____TEST_SUPPORT_FUNCTIONS_________________________________ = 0


def test__verify_input_parms():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    context = mpc.mp_phot.get_context()
    this_directory, mp_string, an_string = context
    defaults_dict = mpc.ini.make_defaults_dict()
    instrument_dict = mpc.ini.make_instrument_dict(defaults_dict)
    ccd = mpc.mp_color.make_color_control_dict(this_directory, defaults_dict)

    # Case: one color, valid, definition is already a nested tuple:
    definition = (('R', 'I', 'SR', 'SI'),)
    catalog_color_index = ('SR', 'SI')
    verified = mpc.mp_color._verify_input_parms(definition, catalog_color_index, instrument_dict, ccd)
    assert verified == True

    # Case: one color, valid, definition is a bare tuple of 4 strings:
    definition = ('R', 'I', 'SR', 'SI')
    catalog_color_index = ('SR', 'SI')
    verified = mpc.mp_color._verify_input_parms(definition, catalog_color_index, instrument_dict, ccd)
    assert verified == True

    # Case: one color, definition has invalid form:
    definition = ('R', 'I', 'SR')  # 3 elements, should be 4
    catalog_color_index = ('SR', 'SI')
    verified = mpc.mp_color._verify_input_parms(definition, catalog_color_index, instrument_dict, ccd)
    assert verified == False
    definition = ('R', 'I', 'SR', 999)  # should be strings
    verified = mpc.mp_color._verify_input_parms(definition, catalog_color_index, instrument_dict, ccd)
    assert verified == False
    definition = tuple()  # empty tuple
    verified = mpc.mp_color._verify_input_parms(definition, catalog_color_index, instrument_dict, ccd)
    assert verified == False
    definition = ['R', 'I', 'SR', 'SI']  # list, but should be tuple
    verified = mpc.mp_color._verify_input_parms(definition, catalog_color_index, instrument_dict, ccd)
    assert verified == False

    # Case: one color, definition requires filter that is unavailable:
    definition = ('XXX', 'I', 'SR', 'SI')
    catalog_color_index = ('SR', 'SI')
    verified = mpc.mp_color._verify_input_parms(definition, catalog_color_index, instrument_dict, ccd)
    assert verified == False
    definition = ('R', 'XXXY', 'SR', 'SI')
    verified = mpc.mp_color._verify_input_parms(definition, catalog_color_index, instrument_dict, ccd)
    assert verified == False

    # Case: one color, extinction missing from color control file:
    definition = ('V', 'I', 'SR', 'SI')
    catalog_color_index = ('SR', 'SI')
    verified = mpc.mp_color._verify_input_parms(definition, catalog_color_index, instrument_dict, ccd)
    assert verified == False

    # Case: one color, catalog color index has a passband not listed as available:
    definition = ('R', 'I', 'SR', 'SI')
    catalog_color_index = ('XXX', 'SI')
    verified = mpc.mp_color._verify_input_parms(definition, catalog_color_index, instrument_dict, ccd)
    assert verified == False

    # TODO: test two-color cases.


def test__verify_images_available():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    assert mpc.mp_color._verify_images_available(df_all, definition=('R', 'I', 'SR', 'SI')) == True
    assert mpc.mp_color._verify_images_available(df_all, definition=('R', 'SI', 'SR', 'SI')) == False
    assert mpc.mp_color._verify_images_available(df_all, definition=('R', 'I', 'SR', 'I')) == True
    mp_in_r = (df_all['Type'] == 'MP') & (df_all['Filter'] == 'R')
    df_partial = df_all.loc[mp_in_r, :]
    assert mpc.mp_color._verify_images_available(df_partial, definition=('R', 'I', 'SR', 'SI')) == False


def test__verify_catalog_color_available():
    import random
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify df column names available:
    assert mpc.mp_color._verify_catalog_color_available(df_all, definition=('R', 'I', 'SR', 'SI')) == True
    assert mpc.mp_color._verify_catalog_color_available(df_all, definition=('R', 'I', 'XX', 'SI')) == False
    # Verify values available:
    df_defaced = df_all.copy()
    is_comp = (df_defaced['Type'] == 'Comp')
    comp_serials = df_defaced.loc[is_comp, 'Serial']
    count_to_set_nan = (2 * len(comp_serials) // 3)
    serials_to_set_nan = random.sample(list(comp_serials), count_to_set_nan)
    df_defaced.loc[serials_to_set_nan, 'SR'] = np.nan
    assert mpc.mp_color._verify_catalog_color_available(df_defaced,
                                                        definition=('R', 'I', 'SR', 'SI')) == False
    df_defaced.loc[:, 'SR'] = 1.0
    assert mpc.mp_color._verify_catalog_color_available(df_defaced,
                                                        definition=('R', 'I', 'SR', 'SI')) == True


_____TEST_SCREENING_FUNCTIONS____________________________ = 0

# The approach for each test is:
#    (1) verify df_all is ok,
#    (2) verify fn doesn't change df_all (if no defects found),
#    (3) verify that test operates correctly on df_defaced (= df_all with added defect to screen for).


def make_df_all_for_tests():
    df_all = mpc.mp_phot.make_df_all(('R', 'I'), comps_only=False, require_mp_obs_each_image=True)
    df_all = df_all.rename(columns={'g': 'SG',   'r': 'SR',   'i': 'SI',   'z': 'SZ',
                                    'dg': 'dSG', 'dr': 'dSR', 'di': 'dSI', 'dz': 'dSZ'})
    assert len(df_all) == 183
    assert set(df_all['FITSfile']) == {'MP_426-0011-R.fts', 'MP_426-0012-I.fts'}
    return df_all


def test__remove_images_on_user_request():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify no change on inoperative screen:
    df_screened = mpc.mp_color._remove_images_on_user_request(df_all, ['NOT AN IMAGE.fts'])
    assert df_screened.equals(df_all)
    # Verify proper change on proper screen:
    df_screened = mpc.mp_color._remove_images_on_user_request(df_all, ['MP_426-0011-R.fts'])
    assert len(df_screened) == 87
    assert set(df_screened['FITSfile']) == {'MP_426-0012-I.fts'}


def test__remove_comps_on_user_request():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify no change on inoperative screen:
    df_screened = mpc.mp_color._remove_comps_on_user_request(df_all, ['NOT A COMP ID'])
    assert df_screened.equals(df_all)
    # Verify proper change on proper screen:
    comp_ids_to_remove = ['603', '647', '819', '603', 'NOT A COMP ID']
    df_screened = mpc.mp_color._remove_comps_on_user_request(df_all, comp_ids_to_remove)
    assert len(df_screened) == 178
    assert all([id not in df_screened['SourceID'] for id in comp_ids_to_remove])


def test__remove_obs_with_unusable_instmag_instmagsigma():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify no change on inoperative screen:
    df_screened = mpc.mp_color._remove_obs_with_unusuable_instmag_instmagsigma(df_all)
    assert df_screened.equals(df_all)
    # Verify proper change on proper screen:
    df_defaced = df_all.copy()
    df_defaced.loc['162', ['InstMag', 'InstMagSigma']] = None
    df_defaced.loc['179', 'InstMag'] = None
    bad_obs_ids = ['162', '179']
    df_screened = mpc.mp_color._remove_obs_with_unusuable_instmag_instmagsigma(df_defaced)
    assert len(df_screened) == 181
    assert set(df_screened['Serial']) == set(df_all['Serial']) - set(bad_obs_ids)


def test__keep_images_with_one_mp_obs():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify no change on inoperative screen:
    df_screened = mpc.mp_color._keep_images_with_one_mp_obs(df_all)
    assert df_screened.equals(df_all)
    # Verify proper change on proper screen:
    df_defaced = df_all.copy()
    obs_to_remove = (df_defaced['Type'] == 'MP') & (df_defaced['FITSfile'] == 'MP_426-0011-R.fts')
    df_defaced = df_defaced.loc[~obs_to_remove, :]
    df_screened = mpc.mp_color._keep_images_with_one_mp_obs(df_defaced)
    assert not any(df_screened['FITSfile'] == 'MP_426-0011-R.fts')
    assert sum(df_screened['FITSfile'] != 'MP_426-0011-R.fts') == \
           sum(df_all['FITSfile'] != 'MP_426-0011-R.fts')
    assert sum(df_defaced['Type'] == 'MP') == 1


def test__keep_images_with_low_mp_obs_uncertainty():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify no change on inoperative screen:
    max_mp_obs_uncert = df_all.loc[df_all['Type'] == 'MP', 'InstMagSigma'].max()
    df_screened = mpc.mp_color._keep_images_with_low_mp_obs_uncertainty(df_all, max_mp_obs_uncert * 1.05)
    assert df_screened.equals(df_all)
    # Verify proper change on proper screen:
    median_mp_obs_uncert = df_all.loc[df_all['Type'] == 'MP', 'InstMagSigma'].median()
    df_screened = mpc.mp_color._keep_images_with_low_mp_obs_uncertainty(df_all, median_mp_obs_uncert)
    assert set(df_screened['FITSfile']) == {'MP_426-0011-R.fts'}
    assert len(df_screened) == 96


def test__remove_comps_obs_with_high_obs_uncertainty():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify no change on inoperative screen:
    max_comp_obs_uncert = df_all.loc[df_all['Type'] == 'Comp', 'InstMagSigma'].max()
    df_screened = mpc.mp_color._remove_comps_obs_with_high_obs_uncertainty(df_all,
                                                                           max_comp_obs_uncert * 1.05)
    assert df_screened.equals(df_all)
    # Verify proper change on proper screen:
    df_screened = mpc.mp_color._remove_comps_obs_with_high_obs_uncertainty(df_all,
                                                                           max_comp_obs_uncert * 0.95)
    assert len(df_screened) == 181
    assert sum(df_screened['Type'] == 'MP') == 2


def test__apply_selections_to_comp_obs():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify no change on inoperative screen:
    color_control_dict_0 = {'min sr mag': 0,
                            'max sr mag': 100,
                            'max catalog dsr mmag': 1000,
                            'min sloan ri color': -10,
                            'max sloan ri color': +100
                            }  # these values will remove no comp obs.
    df_screened = mpc.mp_color._apply_selections_to_comp_obs(df_all, color_control_dict_0)
    assert df_screened.equals(df_all)
    # Verify proper change on single proper screen:
    color_control_dict_1 = color_control_dict_0.copy()
    color_control_dict_1['min sr mag'] = df_all['SR'].min() + 0.5  # MP are NaN thus ignored here.
    df_screened = mpc.mp_color._apply_selections_to_comp_obs(df_all, color_control_dict_1)
    assert len(df_screened) == 180  # 3 comp obs removed, all MP obs retained
    assert df_screened['SR'].min() > color_control_dict_1['min sr mag']
    # Verify proper change on multiple proper screens:
    color_control_dict_2 = color_control_dict_0.copy()
    color_control_dict_2['min sr mag'] = df_all['SR'].min() + 0.5  # MP are NaN thus ignored here.
    color_control_dict_2['max sr mag'] = df_all['SR'].max() + 0.3  # "
    color_control_dict_2['max catalog dsr mmag'] = df_all['dSR'].max() - 0.9  # "
    color_control_dict_2['min sloan ri color'] = (df_all['SR'] - df_all['SI']).min() + 0.1  # "
    color_control_dict_2['max sloan ri color'] = (df_all['SR'] - df_all['SI']).max() - 0.1  # "
    df_screened = mpc.mp_color._apply_selections_to_comp_obs(df_all, color_control_dict_2)
    assert len(df_screened) == 113  # 3 comp obs removed, all MP obs retained
    assert df_screened['SR'].min() > color_control_dict_2['min sr mag']
    assert df_screened['SR'].max() < color_control_dict_2['max sr mag']
    assert df_screened['dSR'].max() < color_control_dict_2['max catalog dsr mmag']
    assert (df_screened['SR'] - df_screened['SI']).min() > color_control_dict_2['min sloan ri color']
    assert (df_screened['SR'] - df_screened['SI']).max() < color_control_dict_2['max sloan ri color']


def test__remove_images_with_few_comps():
    import random
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify no change on inoperative screen:
    df_screened = mpc.mp_color._remove_images_with_few_comps(df_all)
    assert df_screened.equals(df_all)
    # Verify proper change on single proper screen:
    df_defaced = df_all.copy()
    is_first_image = (df_defaced['FITSfile'] == 'MP_426-0011-R.fts')
    is_comp = (df_defaced['Type'] == 'Comp')
    serials_first_image_comps = df_defaced.loc[is_first_image & is_comp, 'Serial']
    count_to_remove = (2 * len(serials_first_image_comps) // 3)
    serials_to_remove = random.sample(list(serials_first_image_comps), count_to_remove)
    obs_to_keep = [serial not in serials_to_remove for serial in df_defaced['Serial']]
    df_defaced = df_defaced.loc[obs_to_keep, :]
    df_screened = mpc.mp_color._remove_images_with_few_comps(df_defaced)
    assert len(df_screened) == sum(df_all['FITSfile'] != 'MP_426-0011-R.fts')
    assert not any(df_screened['FITSfile'] == 'MP_426-0011-R.fts')


def test__remove_comps_absent_from_any_image():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify few changes on screen of original dataframe:
    df_screened = mpc.mp_color._remove_comps_absent_from_any_image(df_all)
    assert len(df_screened) == 172
    comp_ids = set(df_screened.loc[df_screened['Type'] == 'Comp', 'SourceID'])
    n_images = len(set(df_screened['FITSfile']))
    assert all([sum(df_screened['SourceID'] == comp_id) == n_images for comp_id in comp_ids])


def test__remove_comp_obs_on_user_request():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify no change on inoperative screen:
    df_screened = mpc.mp_color._remove_comp_obs_on_user_request(df_all, ['NOT A VALID OBS'])
    assert df_screened.equals(df_all)
    # Verify proper change on proper screen:
    obs_ids_to_remove = ['182', '1', '21', 'NOT A VALID OBS']
    df_screened = mpc.mp_color._remove_comp_obs_on_user_request(df_all, obs_ids_to_remove)
    df_all_not_in_obs_ids_to_remove = [s not in obs_ids_to_remove for s in df_all['Serial']]
    assert df_screened.equals(df_all.copy().loc[df_all_not_in_obs_ids_to_remove, :])


def test__remove_images_with_too_few_comps():
    import random
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    df_all = make_df_all_for_tests()
    # Verify no change on inoperative screen:
    df_screened = mpc.mp_color._remove_images_with_too_few_comps(df_all, min_valid_comps_per_image=5)
    assert df_screened.equals(df_all)
    # Verify proper change on proper screen:
    df_defaced = df_all.copy()
    is_first_image = (df_defaced['FITSfile'] == 'MP_426-0011-R.fts')
    is_comp = (df_defaced['Type'] == 'Comp')
    serials_first_image_comps = df_defaced.loc[is_first_image & is_comp, 'Serial']
    count_to_remove = len(serials_first_image_comps) - 20  # that is, keep 20.
    serials_to_remove = random.sample(list(serials_first_image_comps), count_to_remove)
    obs_to_keep = list(~ df_defaced['Serial'].isin(serials_to_remove))
    df_defaced = df_defaced.loc[obs_to_keep, :]
    df_screened = mpc.mp_color._remove_images_with_too_few_comps(df_defaced, min_valid_comps_per_image=21)
    assert len(df_screened) == 87
    assert all([f != 'MP_426-0011-R.fts' for f in df_screened['FITSfile']])


_____HELPER_FUNCTIONS_for_TESTING___________________________ = 0


def resume_with_df_all(mp_string, an_string):
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, mp_string, an_string)
    context = mpc.mp_phot.get_context()
    this_directory, mp_string, an_string = context
    defaults_dict = mpc.ini.make_defaults_dict()
    instrument_dict = mpc.ini.make_instrument_dict(defaults_dict)
    color_control_dict = mpc.mp_color.make_color_control_dict(this_directory, defaults_dict)
    return this_directory, defaults_dict, instrument_dict, test_make_color_control_dict
