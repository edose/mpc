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
    pl.make_color_roster(an=20201122, site_name='DSW')
