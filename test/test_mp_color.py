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
TEST_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, "test", '$test_data_do_color')


def test_do_color_2filters():
    # Test with a single FITS file in each filter:
    # Assumes start(), and assess() have been run, that proper color_control.ini file is present,
    #     and that make_dfs() has been run so that df*.csv files are ready.
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    this_directory, mp_string, an_string = mpc.mp_phot.get_context()
    # log_file = open(mpc.mp_phot.LOG_FILENAME, mode='a')
    # mpc.mp_phot._write_color_control_ini_template(this_directory, log_file)
    # mpc.mp_phot.make_dfs()
    mpc.mp_color.do_color_2filters(filters=('R', 'I'), target_color=('SR', 'SI'),
                                   color_index_passbands=('SR', 'SI'))
    assert 1 == 1


def test_make_color_control_dict():
    mpc.mp_phot.resume(TEST_TOP_DIRECTORY, 426, '20201023')
    context = mpc.mp_phot.get_context()
    this_directory, mp_string, an_string = context
    defaults_dict = mpc.ini.make_defaults_dict()

    ccd = mpc.mp_color.make_color_control_dict(this_directory, defaults_dict)
    assert isinstance(ccd, dict)
    assert len(ccd) == 16
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
