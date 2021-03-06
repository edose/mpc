__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os

# External packages:
import pandas as pd
import pytest

# Other EVD modules:
import mpc.mp_phot
import mpc.ini

# TARGET TEST MODULE:
import mpc.mp_planning as pl

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_make_color_roster():
    # Normal case, with both outside files:
    pl.make_color_roster(an=20201129, site_name='DSW', min_vmag=11, max_mandatory_mp_number=300)



