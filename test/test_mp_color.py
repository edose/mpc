__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
from random import gauss, seed, uniform

# External packages:
import pandas as pd
import pytest

# Other EVD modules:
import mpc.mp_phot
import mpc.ini

# TARGET TEST MODULE:
import mpc.mp_color

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, "test", '$test_data_mp_color')


def test_do_color_2filters():
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
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    context = mpc.mp_phot.get_context()
    this_directory, mp_string, an_string = context
    defaults_dict = mpc.ini.make_defaults_dict()
    ccd = mpc.mp_color.make_color_control_dict(this_directory, defaults_dict)

    assert isinstance(ccd, dict)
    assert len(ccd) == 16
    assert ccd['mp location filenames'] == ('MP_426-0011-R.fts', 'MP_426-0012-I.fts')
    assert ccd['x pixel'] == (1034.4, 1036.0)
    assert ccd['y pixel'] == (454.3, 455.4)

    assert ccd['max mp obs mag uncertainty']
    assert ccd['max comp obs mag uncertainty']
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
    resume_with_df_all('426', '20201023')
    df_all = mpc.mp_phot.make_df_all(('R', 'I'), comps_only=False, require_mp_obs_each_image=True)
    assert mpc.mp_color._verify_images_available(df_all, definition=('R', 'I', 'SR', 'SI')) == True
    assert mpc.mp_color._verify_images_available(df_all, definition=('R', 'SI', 'SR', 'SI')) == False
    assert mpc.mp_color._verify_images_available(df_all, definition=('R', 'I', 'SR', 'I')) == False
    mp_in_r = (df_all['Type'] == 'MP') & (df_all['Filter'] == 'R')
    df_partial = df_all.loc[mp_in_r, :]
    assert mpc.mp_color._verify_images_available(df_partial, definition=('R', 'I', 'SR', 'SI')) == False






_____TEST_SCREENING_FUNCTIONS____________________________ = 0


def test__remove_images_on_user_request():
    pass


_____HELPER_FUNCTIONS_for_TESTING___________________________ = 0


def resume_with_df_all(mp_string, an_string):
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, mp_string, an_string)
    context = mpc.mp_phot.get_context()
    this_directory, mp_string, an_string = context
    defaults_dict = mpc.ini.make_defaults_dict()
    instrument_dict = mpc.ini.make_instrument_dict(defaults_dict)
    color_control_dict = mpc.mp_color.make_color_control_dict(this_directory, defaults_dict)
    return this_directory, defaults_dict, instrument_dict, test_make_color_control_dict
