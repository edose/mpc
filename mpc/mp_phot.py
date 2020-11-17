# Python core packages:
import os
from math import cos, pi, log10, log, floor, ceil, sqrt, sin
from collections import Counter
from datetime import datetime, timezone
# from statistics import pstdev, mean

# External packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.formula.api as smf
# import statsmodels.api as sm
from scipy.stats import norm
from astropy.nddata import CCDData  # TODO: for temporary testing only / 20200812.
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astroplan import Observer, FixedTarget
# import ephem

# From this (mpc) package:
from mpc.catalogs import Refcat2, get_bounding_ra_dec
from mpc.mp_planning import all_mpfile_names, MPfile
from mpc.ini import *

# From EVD package photrix (deprecated import--> prefer astropak:
from photrix.image import Image, FITS
from photrix.util import RaDec, jd_from_datetime_utc, datetime_utc_from_jd, \
    MixedModelFit, ra_as_hours, dec_as_hex

# From EVD package astropak (preferred):
import astropak.ini

__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# MPC_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BOOT_INI_FILENAME = 'defaults.ini'

ARROW_CHARACTER = '\U00002794'
DAYS_PER_YEAR_NOMINAL = 365.25
VALID_FITS_FILE_EXTENSIONS = ['.fits', '.fit', '.fts']
DEGREES_PER_RADIAN = 180.0 / pi

# To assess FITS files in assess():
MIN_FWHM = 1.5  # in pixels.
MAX_FWHM = 14  # "
FOCUS_LENGTH_MAX_PCT_DEVIATION = 3.0

MARGIN_RA_ZERO = 5  # in degrees, to handle RA~zero; s/be well larger than images' height or width.

# For color handling:
DEFAULT_FILTERS_FOR_MP_COLOR_INDEX = ('V', 'I')
COLOR_INDEX_PASSBANDS = ('r', 'i')
DEFAULT_MP_RI_COLOR = 0.22  # close to known Sloan mean (r-i) for MPs.
MAX_COLOR_COMP_UNCERT = 0.015  # mag

# To screen observations:
MAX_MAG_UNCERT = 0.03  # min signal/noise for COMP obs (as InstMagSigma).
# The next two probably should go in Instrument class, but ok for now.
ADU_SATURATED = 56000  # Max ADU allowable in original (Ur) images.
VIGNETTING = (1846, 0.62)  # (px from center, max fract of ADU_SATURATED allowed) both at corner.

# For this package:
MP_PHOT_TOP_DIRECTORY = 'C:/Astro/MP Photometry/'
MP_PHOT_LOG_FILENAME = 'mp_photometry.log'
MP_PHOT_CONTROL_FILENAME = 'control.txt'
TRANSFORM_CONTROL_FILENAME = 'control_transform.txt'
# COLOR_CONTROL_FILENAME = 'color_control.ini'
# COLOR_CONTROL_TEMPLATE_FILENAME = 'color_control.template'
MP_PHOT_DF_OBS_ALL_FILENAME = 'df_obs_all.csv'
MP_PHOT_DF_IMAGES_ALL_FILENAME = 'df_images_all.csv'
MP_PHOT_DF_COMPS_ALL_FILENAME = 'df_comps_all.csv'
# TRANSFORM_CLEAR_SR_SR_SI = 0.025  # estimate from MP 1074 20191109 (37 images).
TRANSFORM_CLEAR_SR_SR_SI = -0.20  # estimate from MP 191 20200617 (18 Clear images).
DEFAULT_MODEL_OPTIONS = {'fit_transform': 'fit=2', 'fit_extinction': False,
                         'fit_vignette': True, 'fit_xy': False,
                         'fit_jd': True}  # defaults if not found in control file.
TRANSFORM_IMAGE_PREFIX = 'Image_Transform_'
SESSION_IMAGE_PREFIX = 'Image_Session_'


# For selecting comps within Refcat2 object (intentionally wide; can narrow with a control file later):
MIN_R_MAG = 10
MAX_R_MAG = 16     # may increase if deeper tranches of refcat2 get used later.
MAX_G_UNCERT = 20  # millimagnitudes
MAX_R_UNCERT = 20  # "
MAX_I_UNCERT = 20  # "
MIN_SLOAN_RI_COLOR = -0.4  # probably no stars r-i < -0.4.
MAX_SLOAN_RI_COLOR = 0.8   # only to reject possible variable stars as comps.

DO_PHOT_COMP_SELECTION_DEFAULTS = {
    'min_catalog_r_mag': 10,
    'max_catalog_r_mag': 16,
    'max_catalog_dr_mag': 15,
    'min_catalog_ri_color': 0.0,
    'max_catalog_ri_color': 0.4
}

# This might get used later if do_phot() gets adapted to transform determination:
# TRANSFORM_COMP_SELECTION_DEFAULTS = {
#     'min_catalog_r_mag': 10,
#     'max_catalog_r_mag': 16,
#     'max_catalog_dr_mag': 20,
#     'min_catalog_ri_color': -0.4,
#     'max_catalog_ri_color': 0.8
# }

# COLOR_COMP_SELECTION_DEFAULTS = {
#     'min_catalog_r_mag': 10,
#     'max_catalog_r_mag': 16,
#     'max_catalog_dr_mag': 20,
#     'min_catalog_ri_color': -0.4,
#     'max_catalog_ri_color': 0.8
# }

# For ALCDEF File generation:
DSW_SITE_DATA = {'longitude': -105.6536, 'latitude': +35.3311, 'elevation': 2210,
                 'facility': 'Deep Sky West Observatory', 'mpccode': 'V28'}
ALCDEF_DATA = {'contactname': 'Eric V. Dose',
               'contactinfo': 'MP@ericdose.com',
               'observers': 'Dose, E.V.',
               'filter': 'C',
               'magband': 'SR'}


_____EARLY_WORKFLOW_ATLAS_BASED________________________________________________ = 0

"""  ***************************************************************************
     WORKFLOW STEPS (example lines):
     * Ensure at least 1 file IN Photometric (e.g., Clear or BB) filter.
     >>> start(MP_PHOT_TOP_DIRECTORY + '/Test', 1074, 20191109)
     >>> assess()
     * Edit control.txt: add 2 MP positions (x_pixel, y_pixel), at early and late times.
     >>> make_dfs()
     * Edit control.txt: (1) comp-selection limits, (2) model options [both optional].
     >>> do_mp_phot()
     ********************************************
     (at any time:)
     >>> resume(MP_PHOT_TOP_DIRECTORY + '/Test', 1074, 20191109), esp. after Console restart.
     ***************************************************************************
"""


def start(mp_top_directory=MP_PHOT_TOP_DIRECTORY, mp_number=None, an_string=None,
          log_filename=MP_PHOT_LOG_FILENAME):
    """  Preliminaries to begin MP photometry workflow.
    :param mp_top_directory: path of lowest directory common to all MP photometry FITS, e.g.,
               'C:/Astro/MP Photometry' [string]
    :param mp_number: number of target MP, e.g., 1602 for Indiana. [integer or string].
    :param an_string: Astronight string representation, e.g., '20191106' [string].
    :param log_filename: name of log file to initiate. [string]
    :return: [None]
    """
    if mp_number is None or an_string is None:
        print(' >>>>> Usage: start(top_directory, mp_number, an_string)')
        return

    # Construct directory path and make it the working directory:
    mp_int = int(mp_number)  # put this in try/catch block?
    mp_string = str(mp_int)
    mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    os.chdir(mp_directory)
    print('Working directory set to:', mp_directory)

    # Initiate log file and finish:
    log_file = open(log_filename, mode='w')  # new file; wipe old one out if it exists.
    log_file.write(mp_directory + '\n')
    log_file.write('MP: ' + mp_string + '\n')
    log_file.write('AN: ' + an_string + '\n')
    log_file.write('This log started: ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    log_file.close()
    print('Log file started.')
    print('Next: assess()')


def resume(mp_top_directory=MP_PHOT_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Restart a workflow in its correct working directory,
         but keep the previous log file--DO NOT overwrite it.
        Parameters as for start().
    :return: [None]
    """
    if mp_number is None or an_string is None:
        print(' >>>>> Usage: start(top_directory, mp_number, an_string)')
        return

    # Go to proper working directory:
    mp_int = int(mp_number)  # put this in try/catch block?
    mp_string = str(mp_int)
    this_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    os.chdir(this_directory)

    # Verify that proper log file already exists in the working directory:
    this_context = get_context()
    if get_context() is None:
        print(' >>>>> Can\'t resume in', this_directory, '(has start() been run?)')
        return
    log_this_directory, log_mp_string, log_an_string = this_context
    if log_mp_string.lower() == mp_string.lower() and log_an_string.lower() == an_string.lower():
        print('READY TO GO in', this_directory)
    else:
        print(' >>>>> Can\'t resume in', this_directory)


def assess(log_filename=MP_PHOT_LOG_FILENAME, write_mp_phot_control_stub=True):
    """  First, verify that all required files are in the working directory or otherwise accessible.
         Then, perform checks on FITS files in this directory before performing the photometry proper.
         Modeled after and extended from assess() found in variable-star photometry package 'photrix'.
                                    May be zero for MP color index determination only. [int]
    :param log_filename: name of log file to be written, e.g., mp_photometry.log or mp_color.log. [string]
    :param write_mp_phot_control_stub: True iff photometry control stub file is to be written
        (typically True for mp_phot lightcurve workflow, False for color workflow and any others). [boolean]
    :return: [None]
    """
    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
    log_file = open(log_filename, mode='a')  # set up append to log file.
    log_file.write('\n===== access()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    n_warnings = 0

    # Get FITS file names in current directory:
    fits_filenames = get_fits_filenames(this_directory)
    print(str(len(fits_filenames)) + ' FITS files found:')
    log_file.write(str(len(fits_filenames)) + ' FITS files found:' + '\n')

    # Count FITS files by filter, write totals
    #    (we've stopped classifying files by intention; now we include all valid FITS in dfs):
    filter_counter = Counter()
    valid_fits_filenames = []
    for filename in fits_filenames:
        fits = FITS(this_directory, '', filename)
        if fits.is_valid:
            valid_fits_filenames.append(filename)
            filter_counter[fits.filter] += 1
        else:
            print(' >>>>> WARNING: FITS file', filename, 'appears INVALID, is EXCLUDED.')
    for filter in filter_counter.keys():
        print('   ' + str(filter_counter[filter]), 'in filter', filter + '.')

    # Start dataframe for main FITS integrity checks:
    fits_extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in valid_fits_filenames])
    df = pd.DataFrame({'Filename': valid_fits_filenames,
                       'Extension': fits_extensions.values}).sort_values(by=['Filename'])
    df = df.set_index('Filename', drop=False)
    df['Valid'] = False
    df['PlateSolved'] = False
    df['Calibrated'] = True
    df['FWHM'] = np.nan
    df['FocalLength'] = np.nan

    # Try to open all fits filenames as FITS, collect all info relevant to errors and warnings:
    for filename in df['Filename']:
        fits = FITS(this_directory, '', filename)
        df.loc[filename, 'Valid'] = fits.is_valid
        if fits.is_valid:
            df.loc[filename, 'PlateSolved'] = fits.is_plate_solved
            df.loc[filename, 'Calibrated'] = fits.is_calibrated
            df.loc[filename, 'FWHM'] = fits.fwhm
            df.loc[filename, 'FocalLength'] = fits.focal_length
            df.loc[filename, 'UTC_mid'] = fits.utc_mid  # needed below for control.txt stub.

    # Invalid FITS files: should be none; report and REMOVE THEM from df:
    invalid_fits = df.loc[~ df['Valid'], 'Filename']
    if len(invalid_fits) >= 1:
        print('\nINVALID FITS files:')
        for f in invalid_fits:
            print('    ' + f)
        print('\n')
        df = df.loc[df['Valid'], :]  # keep only rows for valid FITS files.
        del df['Valid']  # all rows in df now refer to valid FITS files.
    n_warnings += len(invalid_fits)

    # Now assess all FITS, and report errors & warnings:
    not_platesolved = df.loc[~ df['PlateSolved'], 'Filename']
    if len(not_platesolved) >= 1:
        print('NO PLATE SOLUTION:')
        for f in not_platesolved:
            print('    ' + f)
        print('\n')
    else:
        print('All platesolved.')
    n_warnings += len(not_platesolved)

    not_calibrated = df.loc[~ df['Calibrated'], 'Filename']
    if len(not_calibrated) >= 1:
        print('\nNOT CALIBRATED:')
        for f in not_calibrated:
            print('    ' + f)
        print('\n')
    else:
        print('All calibrated.')
    n_warnings += len(not_calibrated)

    odd_fwhm_list = []
    for f in df['Filename']:
        fwhm = df.loc[f, 'FWHM']
        if fwhm < MIN_FWHM or fwhm > MAX_FWHM:  # too small or large:
            odd_fwhm_list.append((f, fwhm))
    if len(odd_fwhm_list) >= 1:
        print('\nUnusual FWHM (in pixels):')
        for f, fwhm in odd_fwhm_list:
            print('    ' + f + ' has unusual FWHM of ' + '{0:.2f}'.format(fwhm) + ' pixels.')
        print('\n')
    else:
        print('All FWHM values seem OK.')
    n_warnings += len(odd_fwhm_list)

    odd_fl_list = []
    mean_fl = df['FocalLength'].mean()
    for f in df['Filename']:
        fl = df.loc[f, 'FocalLength']
        focus_length_pct_deviation = 100.0 * abs((fl - mean_fl)) / mean_fl
        if focus_length_pct_deviation > FOCUS_LENGTH_MAX_PCT_DEVIATION:
            odd_fl_list.append((f, fl))
    if len(odd_fl_list) >= 1:
        print('\nUnusual FocalLength (vs mean of ' + '{0:.1f}'.format(mean_fl) + ' mm:')
        for f, fl in odd_fl_list:
            print('    ' + f + ' has unusual Focal length of ' + str(fl))
        print('\n')
    else:
        print('All Focal Lengths seem OK.')
    n_warnings += len(odd_fl_list)

    # Summarize and write instructions for next steps:
    if n_warnings == 0:
        print('\n >>>>> ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.')
        print('Next: (1) enter MP pixel positions in', MP_PHOT_CONTROL_FILENAME, 'AND SAVE it,',
              '\n      (2) make_dfs()')
        log_file.write('assess(): ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.' + '\n')
    else:
        print('\n >>>>> ' + str(n_warnings) + ' warnings (see listing above).')
        print('        Correct these and rerun assess() until no warnings remain.')
        log_file.write('assess(): ' + str(n_warnings) + ' warnings.' + '\n')

    defaults_dict = make_defaults_dict()
    if write_mp_phot_control_stub:
        write_control_txt_stub(this_directory, log_file, df)        # if it doesn't already exist.
    log_file.close()


def make_dfs(color_control_dict=None):
    """ For one MP on one night: gather images and ATLAS refcat2 catalog data,
            make df_comps, df_obs, and df_images. Modified 20200704 to include all filters.
            Intent: to make dataframes usable for all later purposes,
               esp. for measuring session lightcurves, system transforms, and MP color index.
    :param color_control_dict: None if for mp_phot (lightcurve), a dict of relevant parm values if for color.
    :return: [None]
    USAGE: make_dfs() or make_dfs(color_parm_dict)
    """
    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)

    # Gather key parameters, before computation:
    defaults_dict = make_defaults_dict()
    if color_control_dict is None:
        # Case: this run is for mp_phot, so use parm values local to this module, just as originally done:
        log_filename = MP_PHOT_LOG_FILENAME
        control_filename = MP_PHOT_CONTROL_FILENAME
        mp_location_filenames, x_pixels, y_pixels = read_mp_locations()
        df_obs_all_filename = MP_PHOT_DF_OBS_ALL_FILENAME
        df_images_all_filename = MP_PHOT_DF_IMAGES_ALL_FILENAME
        df_comps_all_filename = MP_PHOT_DF_COMPS_ALL_FILENAME
    else:
        # Case: this run is for color, so get parm values from the color control dict passed in:
        log_filename = defaults_dict['color log filename']
        control_filename = defaults_dict['color control filename']
        mp_location_filenames = color_control_dict.get('mp location filenames')
        x_pixels = color_control_dict.get('x pixels')
        y_pixels = color_control_dict.get('y pixels')
        df_obs_all_filename = color_control_dict.get('df_obs_all filename')
        df_images_all_filename = color_control_dict.get('df_images_all filenames')
        df_comps_all_filename = color_control_dict.get('df_comps_all filenames')

    # Start log file, do initial parm validation:
    log_file = open(log_filename, mode='a')  # set up append to log file.
    log_file.write('\n===== make_dfs()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    if mp_location_filenames is None or x_pixels is None or y_pixels is None:  # save time if parms wrong.
        print(' >>>>> ' + control_filename + ': something wrong with #MP lines. Stopping.')
        log_file.write(' >>>>> ' + control_filename + ': something wrong with #MP lines. Stopping.\n')
        return None

    # Get all relevant FITS filenames, make lists of FITS objects and Image objects:
    fits_names = get_fits_filenames(this_directory)
    fits_list = [FITS(this_directory, '', fits_name) for fits_name in fits_names]  # list of FITS objects
    fits_list = [fits for fits in fits_list if fits.is_valid]  # list of *valid* FITS objects
    image_list = [Image(fits_object) for fits_object in fits_list]  # Image objects
    if len(image_list) <= 0:
        print(' >>>>> ERROR: no FITS files found in this directory.')
        return None

    # Get time range of all MP images:
    utc_mids = [i.fits.utc_mid for i in image_list]
    min_session_utc = min(utc_mids)
    max_session_utc = max(utc_mids)
    mid_session_utc = min_session_utc + (max_session_utc - min_session_utc) / 2

    # Get coordinates for catalog retrieval (from all images' outermost bounding RA, Dec):
    ra_deg_min_list, ra_deg_max_list, dec_deg_min_list, dec_deg_max_list = [], [], [], []
    for fits in fits_list:
        ra_deg_min, ra_deg_max, dec_deg_min, dec_deg_max = get_bounding_ra_dec(fits)
        ra_deg_min_list.append(ra_deg_min)
        ra_deg_max_list.append(ra_deg_max)
        dec_deg_min_list.append(dec_deg_min)
        dec_deg_max_list.append(dec_deg_max)

    straddling_ra_zero = any([(abs(ramin)) < MARGIN_RA_ZERO or (ramin > 360 - MARGIN_RA_ZERO) or
                              (abs(ramax) < MARGIN_RA_ZERO) or (ramax > 360 - MARGIN_RA_ZERO)
                              for (ramin, ramax) in zip(ra_deg_min_list, ra_deg_max_list)])
    if straddling_ra_zero:
        rotated_ra_min = min([((ramin + 180) % 360) for ramin in ra_deg_min_list])
        rotated_ra_max = max([((ramax + 180) % 360) for ramax in ra_deg_max_list])
        ra_min_all = (180 + rotated_ra_min) % 360
        ra_max_all = (180 + rotated_ra_max) % 360
    else:
        ra_min_all = min(ra_deg_min_list)
        ra_max_all = min(ra_deg_max_list)
    dec_min_all = min(dec_deg_min_list)
    dec_max_all = min(dec_deg_max_list)

    # Get ATLAS refcat2 stars covering bounding region of all images:
    refcat2 = Refcat2(ra_deg_range=(ra_min_all, ra_max_all), dec_deg_range=(dec_min_all, dec_max_all))
    lines = screen_comps_for_photometry(refcat2)  # works IN-PLACE.
    refcat2.update_epoch(mid_session_utc)    # apply proper motions to update star positions.
    print('\n'.join(lines), '\n')
    log_file.write('\n'.join(lines) + '\n')

    # Add apertures for all comps within all Image objects:
    df_radec = refcat2.selected_columns(['RA_deg', 'Dec_deg'])
    for image in image_list:
        for i_comp in df_radec.index:
            ra = df_radec.loc[i_comp, 'RA_deg']
            dec = df_radec.loc[i_comp, 'Dec_deg']
            x0, y0 = image.fits.xy_from_radec(RaDec(ra, dec))
            image.add_aperture(str(i_comp), x0, y0)
        print(image.fits.filename + ':', str(len(image.apertures)), 'comp apertures.')

    # Extract MP *RA,Dec* positions from the 2 images:
    # FITS objects in next line are temporary only.
    mp_location_fits = [FITS(this_directory, '', mp_filename) for mp_filename in mp_location_filenames]
    mp_datetime, mp_ra_deg, mp_dec_deg = [], [], []
    for i in range(2):
        ps = mp_location_fits[i].plate_solution
        dx = x_pixels[i] - ps['CRPIX1']
        dy = y_pixels[i] - ps['CRPIX2']
        d_east_west = dx * ps['CD1_1'] + dy * ps['CD1_2']
        d_ra = d_east_west / cos(mp_location_fits[0].dec / DEGREES_PER_RADIAN)
        d_dec = dx * ps['CD2_1'] + dy * ps['CD2_2']
        mp_datetime.append(mp_location_fits[i].utc_mid)
        mp_ra_deg.append(ps['CRVAL1'] + d_ra)  # do NOT normalize this to (0, 360).
        mp_dec_deg.append(ps['CRVAL2'] + d_dec)

    # Get reference RA,Dec and motion rates:
    utc0 = mp_datetime[0]
    ra0 = mp_ra_deg[0]
    dec0 = mp_dec_deg[0]
    span_seconds = (mp_datetime[1] - utc0).total_seconds()
    ra_per_second = (mp_ra_deg[1] - ra0) / span_seconds
    dec_per_second = (mp_dec_deg[1] - dec0) / span_seconds
    mp_id = 'MP_' + mp_string
    log_file.write('MP at JD ' + '{0:.5f}'.format(jd_from_datetime_utc(utc0)) + ':  RA,Dec='
                   + '{0:.5f}'.format(ra0) + u'\N{DEGREE SIGN}' + ', '
                   + '{0:.5f}'.format(dec0) + u'\N{DEGREE SIGN}' + ',  d(RA,Dec)/hour='
                   + '{0:.6f}'.format(ra_per_second * 3600.0) + ', '
                   + '{0:.6f}'.format(dec_per_second * 3600.0) + '\n')

    # Add all MP apertures to Image objects:
    mp_radec_dict = dict()
    for image in image_list:
        dt = (image.fits.utc_mid - utc0).total_seconds()
        ra = ra0 + dt * ra_per_second
        dec = dec0 + dt * dec_per_second
        x0, y0 = image.fits.xy_from_radec(RaDec(ra, dec))
        image.add_aperture(mp_id, x0, y0)
        mp_radec_dict[image.fits.filename] = (ra, dec)  # in degrees; later inserted into df_obs.

    # Gather data for df_image and df_obs (this code adapted from photrix.process.make_df_master()):
    image_dict_list = []
    df_image_obs_list = []
    for image in image_list:
        # Collect info from all Aperture objects:
        ap_names = [k for k in image.apertures.keys()]
        ap_list = [dict(image.results_from_aperture(ap_name)) for ap_name in ap_names]
        df_apertures = pd.DataFrame(data=ap_list, index=ap_names)
        df_apertures['SourceID'] = df_apertures.index.values
        df_apertures.rename(columns={'r_disc': 'DiscRadius',
                                     'r_inner': 'SkyRadiusInner',
                                     'r_outer': 'SkyRadiusOuter',
                                     'x_centroid': 'Xcentroid',
                                     'y_centroid': 'Ycentroid',
                                     'annulus_flux': 'SkyADU',
                                     'annulus_flux_sigma': 'SkySigma',
                                     'fwhm': 'FWHM',
                                     'x1024': 'X1024',
                                     'y1024': 'Y1024',
                                     'vignette': 'Vignette',
                                     'sky_bias': 'SkyBias'},
                            inplace=True)

        # Remove apertures with no flux or with saturated pixel(s):
        has_positive_flux = df_apertures['net_flux'] > 0.0
        df_apertures = df_apertures.loc[has_positive_flux, :]
        adus_saturated = [adu_sat_from_xy(x1024, y1024)
                          for (x1024, y1024) in zip(df_apertures['X1024'], df_apertures['Y1024'])]
        is_not_saturated = [mx <= sat for (mx, sat) in zip(df_apertures['max_adu'], adus_saturated)]
        df_apertures = df_apertures.loc[is_not_saturated, :]

        # Complete df_apertures (requires positive flux, screened for in lines just above:
        df_apertures['InstMag'] = -2.5 * np.log10(df_apertures['net_flux'])\
            + 2.5 * log10(image.fits.exposure)
        df_apertures['InstMagSigma'] = (2.5 / log(10)) * \
                                       (df_apertures['net_flux_sigma'] / df_apertures['net_flux'])

        # Remove COMP apertures with low signal-to-noise ratio (keep all MP apertures whatever their SNRs):
        aperture_snr_ok = [sig < MAX_MAG_UNCERT for sig in df_apertures['InstMagSigma']]
        is_mp_aperture = [name.upper().startswith('MP_') for name in df_apertures.index]
        apertures_to_keep = [ok | mp for (ok, mp) in zip(aperture_snr_ok, is_mp_aperture)]
        df_apertures = df_apertures.loc[apertures_to_keep, :]
        df_apertures.drop(['n_disc_pixels', 'n_annulus_pixels', 'max_adu', 'net_flux', 'net_flux_sigma'],
                          axis=1, inplace=True)  # delete unneeded columns.
        df_image_obs = df_apertures.copy()
        df_image_obs['FITSfile'] = image.fits.filename
        df_image_obs['JD_mid'] = image.fits.utc_mid  # to help sort the obs by time; could discard later.
        df_image_obs_list.append(df_image_obs)

        # Add image-specific data to image_dict_list:
        image_dict = dict()
        image_dict['FITSfile'] = image.fits.filename
        image_dict['JD_start'] = jd_from_datetime_utc(image.fits.utc_start)
        image_dict['UTC_start'] = image.fits.utc_start
        image_dict['Exposure'] = image.fits.exposure
        image_dict['UTC_mid'] = image.fits.utc_mid
        image_dict['JD_mid'] = jd_from_datetime_utc(image.fits.utc_mid)
        image_dict['Filter'] = image.fits.filter
        image_dict['Airmass'] = image.fits.airmass
        image_dict['JD_fract'] = np.nan  # placeholder (actual value requires that all JDs be known).
        image_dict_list.append(image_dict)

    # Make df_obs (one row per observation across all images):
    df_obs = pd.DataFrame(pd.concat(df_image_obs_list, ignore_index=True, sort=True))
    df_obs['Type'] = ['MP' if id.startswith('MP_') else 'Comp' for id in df_obs['SourceID']]
    df_obs.sort_values(by=['JD_mid', 'Type', 'SourceID'], inplace=True)
    df_obs.drop(['JD_mid'], axis=1, inplace=True)
    df_obs.insert(0, 'Serial', [str(s) for s in range(1, 1 + len(df_obs))])
    df_obs.index = list(df_obs['Serial'])  # list to prevent naming the index
    df_obs = reorder_df_columns(df_obs, ['Serial', 'FITSfile', 'SourceID', 'Type',
                                         'InstMag', 'InstMagSigma'])
    print('   ' + str(len(df_obs)) + ' obs retained.')

    # Make df_comps (one row per comp star, mostly catalog data):
    df_comps = refcat2.selected_columns(['RA_deg', 'Dec_deg', 'RP1', 'R1', 'R10',
                                         'g', 'dg', 'r', 'dr', 'i', 'di', 'z', 'dz',
                                         'BminusV', 'APASS_R', 'T_eff'])
    comp_ids = [str(i) for i in df_comps.index]
    df_comps.index = comp_ids
    df_comps.insert(0, 'CompID', comp_ids)
    print('   ' + str(len(df_comps)) + ' comps retained.')

    # Make df_images (one row per FITS file):
    df_images = pd.DataFrame(data=image_dict_list)
    df_images.index = list(df_images['FITSfile'])  # list to prevent naming the index
    jd_floor = floor(df_images['JD_mid'].min())  # requires that all JD_mid values be known.
    df_images['JD_fract'] = df_images['JD_mid'] - jd_floor
    df_images.sort_values(by='JD_mid')
    df_images = reorder_df_columns(df_images, ['FITSfile', 'JD_mid', 'Filter', 'Exposure', 'Airmass'])
    print('   ' + str(len(df_images)) + ' images retained.')

    # Remove from all 3 dataframes all reference to images with: missing MP flux, or no comps:
    # Nested function:
    def remove_image_rows(filename, df_images, df_obs):
        df_images_rows_to_keep = list(df_images.index != filename)
        df_images = df_images.loc[df_images_rows_to_keep, :]
        df_obs_rows_to_keep = [(fn != filename) for fn in df_obs['FITSfile']]
        df_obs = df_obs.loc[df_obs_rows_to_keep, :]
        return df_images, df_obs
    # Do the removal:
    for im in df_images.index:
        indices = list(df_obs.loc[df_obs['FITSfile'] == im].index)
        try:
            mp_index = [idx for idx in indices if df_obs.loc[idx, 'Type'] == 'MP'][0]
        except IndexError:
            mp_index = None
            df_images, df_obs = remove_image_rows(im, df_images, df_obs)
            print(' >>>>> WARNING: image', im, 'removed for absence of valid MP.')
            continue  # that's all we do for this image.
        im_comps = [idx for idx in indices if df_obs.loc[idx, 'Type'] == 'Comp']
        if len(im_comps) <= 0:
            df_images, df_obs = remove_image_rows(im, df_images, df_obs)
            print(' >>>>> WARNING: image', im, 'removed for absence of any valid comps.')
            continue

    # Calculate *per-observation* airmass, add column ObsAirmass to df_obs:
    site_data = DSW_SITE_DATA
    observer = Observer(longitude=site_data['longitude'] * u.deg,
                        latitude=site_data['latitude'] * u.deg,
                        elevation=site_data['elevation'] * u.m, name='DSW')  # astropy.
    df_obs['ObsAirmass'] = None
    for im in df_images.index:
        time = Time(df_images.loc[im, 'JD_mid'], format='jd')
        altaz_frame = observer.altaz(time)
        indices = list(df_obs.loc[df_obs['FITSfile'] == im].index)
        # >>> Add ObsAirmass for this image's comp stars:
        comp_indices = [idx for idx in indices if df_obs.loc[idx, 'Type'] == 'Comp']
        source_ids = df_obs.loc[comp_indices, 'SourceID']
        ra_list = df_comps.loc[source_ids, 'RA_deg']
        dec_list = df_comps.loc[source_ids, 'Dec_deg']
        skycoord_list = [make_skycoord(ra, dec) for (ra, dec) in zip(ra_list, dec_list)]
        alt_list = [skycoord.transform_to(altaz_frame).alt.value for skycoord in skycoord_list]
        airmass_list = [1.0 / sin(alt / DEGREES_PER_RADIAN) for alt in alt_list]
        df_obs.loc[comp_indices, 'ObsAirmass'] = airmass_list
        # >>> Add ObsAirmass for this image's MP:
        ra, dec = mp_radec_dict[im]
        skycoord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        alt = skycoord.transform_to(altaz_frame).alt.value
        airmass = 1.0 / sin(alt / DEGREES_PER_RADIAN)
        try:
            mp_index = [idx for idx in indices if df_obs.loc[idx, 'Type'] == 'MP'][0]
        except IndexError:
            print(' >>>>> ERROR: cannot parse MP index.')
            exit(0)
        df_obs.loc[mp_index, 'ObsAirmass'] = airmass
        print('ObsAirmass done for', im)

    # Write df_obs to CSV file (rather than returning the df):
    fullpath_df_obs_all = os.path.join(this_directory, df_obs_all_filename)
    df_obs.to_csv(fullpath_df_obs_all, sep=';', quotechar='"',
                  quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    n_comp_obs = sum([t.lower() == 'comp' for t in df_obs['Type']])
    n_mp_obs = sum([t.lower() == 'mp' for t in df_obs['Type']])
    print('obs written to', fullpath_df_obs_all)
    log_file.write(df_obs_all_filename + ' written: ' + str(n_comp_obs) + ' comp obs & ' +
                   str(n_mp_obs) + ' MP obs.\n')

    # Write df_comps to CSV file (rather than returning the df):
    fullpath_df_comps_all = os.path.join(this_directory, df_comps_all_filename)
    df_comps.to_csv(fullpath_df_comps_all, sep=';', quotechar='"',
                    quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('comps written to', fullpath_df_comps_all)
    log_file.write(df_comps_all_filename + ' written: ' + str(len(df_comps)) + ' comps.\n')

    # Write df_images to CSV file (rather than returning the df):
    fullpath_df_images_all = os.path.join(this_directory, df_images_all_filename)
    df_images.to_csv(fullpath_df_images_all, sep=';', quotechar='"',
                     quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('images written to', fullpath_df_images_all)
    log_file.write(df_images_all_filename + ' written: ' + str(len(df_images)) + ' images.\n')

    log_file.close()
    print('\nNext: (1) enter comp selection limits and model options in ' + control_filename,
          '\n      (2) run do_mp_phot()')



_____SESSION_LIGHTCURVE_PHOTOMETRY________________________________________________ = 0


def do_mp_phot(fits_filter='Clear'):
    """ Primary lightcurve photometry for one session. Takes all data incl. color index, generates:
    (1) canopus-import format results, (2) ALCDEF-format file, and (3) all diagnostic plots.
    Typically iterated, pruning comp-star ranges and outliers, until converged and then simply stop.
    NB: As of 20200704, one may choose the FITS files by filter (typically 'Clear' or 'BB'), but
        lightcurve passband is fixed as Sloan 'r', and
        color index is fixed as Sloan (r-i).
    :param fits_filter: filter allowed for this session's FITS files. Typically 'Clear' or 'BB' [string]
    :returns None. Writes all info to files.
    USAGE: do_mp_phot()   [no return value]
    """
    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
    log_file = open(MP_PHOT_LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n===== do_mp_phot(' + fits_filter + ')  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Load required data:
    df_all = make_df_all(filters_to_include=fits_filter, comps_only=False, require_mp_obs_each_image=True)
    state = get_session_state()  # for extinction and transform values.
    mp_color_ri, source_string_ri = read_mp_ri_color()  # read from control file, returns a tuple.

    # Make df_model: keep only obs which are (1) MP type OR (2) of a comp that is present in every image:
    rows_with_mp_ids = (df_all['Type'] == 'MP')
    df_all_image_list = df_all['FITSfile'].drop_duplicates()
    n_all_images = len(df_all_image_list)
    rows_with_comp_ids = (df_all['Type'] == 'Comp')
    comp_id_list = df_all.loc[rows_with_comp_ids, 'SourceID'].drop_duplicates()
    df_image_count = df_all.groupby('SourceID')[['FITSfile', 'SourceID']].count()
    comp_ids_in_every_image = [id for id in comp_id_list
                               if df_image_count.loc[id, 'FITSfile'] == n_all_images]
    rows_with_qualified_comp_ids = df_all['SourceID'].isin(comp_ids_in_every_image)
    rows_to_keep = rows_with_mp_ids | rows_with_qualified_comp_ids
    df_model = df_all.loc[rows_to_keep, :].copy()
    # Note: df_model now contains all obs, comp, and image data, for both comp stars and minor planets.

    # Mark df_model with user selections, sync comp and image dfs:
    user_selections = read_selection_criteria(MP_PHOT_CONTROL_FILENAME, DO_PHOT_COMP_SELECTION_DEFAULTS)
    apply_do_phot_selections(df_model, user_selections)  # modifies in-place.

    # # Sync the comp and image dataframes (esp. user selections); may be needed later for plotting:
    # df_model_comps, df_model_images = sync_comps_and_images(df_model, df_comps_all, df_images_all)
    # print(str(len(df_model_comps)), 'comps retained in model.')

    # Make photometric model via mixed-model regression, using only selected observations:
    options_dict = read_regression_options(MP_PHOT_CONTROL_FILENAME)
    model = SessionModel(df_model, fits_filter, mp_color_ri, state, options_dict)

    make_session_diagnostic_plots(model, df_model, mp_color_ri, state, user_selections)

    write_canopus_file(model)
    write_alcdef_file(model, mp_color_ri, source_string_ri)
    # model_jds = df_model['JD_mid']
    model_jds = model.df_mp_mags['JD_mid']
    print(' >>>>> Please add this line to MPfile', mp_string + ':',
          '  #OBS', '{0:.5f}'.format(model_jds.min()), ' {0:.5f}'.format(model_jds.max()),
          ' ;', an_string,  '::', mp_string)


class SessionModel:
    def __init__(self, df_model, fits_filter, mp_color_ri, state, options_dict):
        """  Makes and holds photometric model via mixed-model regression. Affords prediction for MP mags.
        :param df_model:
        :param fits_filter: filter allowed for session's FITS files. Typically 'Clear' or 'BB' [string]
        :param mp_color_ri: color index of the Minor Planet target, however derived in do_phot(). [float]
        :param state:
        :param options_dict: holds options for making comp fit; its elements are:
            fit_transform: True iff transform is to be fit; never True in actual photometry model,
                   True set only rarely, to extract transform from images of one field of view. [boolean]
            fit_extinction: True iff extinction is to be fit (uncommon; usually get known value from
                   Site object). [boolean]
            fit_vignette: True iff centered, parabolic term to be included in model. [boolean]
            fit_xy: True iff linear x and y terms to be included in model. [boolean]
            fit_jd: True iff linear time term (zero-point creep in time seen in plot of "cirrus"
                   random-effect term). [boolean]
        """
        self.df_model = df_model
        self.fits_filter = fits_filter
        self.mp_ci = mp_color_ri
        self.state = state
        self.df_used = df_model.copy().loc[df_model['UseInModel'], :]  # only observations used in model.
        self.df_used_comps_only = self.df_used.loc[(self.df_used['Type'] == 'Comp'), :].copy()
        self.df_used_mps_only = self.df_used.loc[(self.df_used['Type'] == 'MP'), :].copy()

        defaults = DEFAULT_MODEL_OPTIONS
        self.fit_transform = options_dict.get('fit_transform', defaults['fit_transform'])
        self.fit_extinction = options_dict.get('fit_extinction', defaults['fit_extinction'])
        self.fit_vignette = options_dict.get('fit_vignette', defaults['fit_vignette'])
        self.fit_xy = options_dict.get('fit_xy', defaults['fit_xy'])
        self.fit_jd = options_dict.get('fit_jd', defaults['fit_jd'])

        self.dep_var_name = 'InstMag_with_offsets'
        self.mm_fit = None      # placeholder for the fit result [photrix MixedModelFit object].
        self.transform = None   # placeholder for this fit parameter result [scalar].
        self.transform_fixed = None  # "
        self.extinction = None  # "
        self.vignette = None    # "
        self.x = None           # "
        self.y = None           # "
        self.jd1 = None         # "

        self._prep_and_do_regression()
        self.df_mp_mags = self._calc_mp_mags()

    def _prep_and_do_regression(self):
        """ Using photrix.util.MixedModelFit class (which wraps statsmodels.MixedLM.from_formula() etc).
            Use ONLY comp data in the model. Later use model's .predict() to calculate best MP mags.
            This is adapted from photrix package SkyModel.
            This function uses comp data only (no minor planet data).
        :return: [None]
        """
        fit_summary_lines = []

        self.df_used_comps_only['CI'] = self.df_used_comps_only['r'] - self.df_used_comps_only['i']
        self.df_used_comps_only['CI2'] = [ci ** 2 for ci in self.df_used_comps_only['CI']]

        # Initiate dependent-variable offset, which will aggregate all such offset terms:
        dep_var_offset = self.df_used_comps_only['r'].copy()  # *copy* CatMag, or it will be damaged

        fixed_effect_var_list = []

        # Handle transform (Color Index) options separately here:
        fit_transform_handled = False  # default, to be set to True when handled.
        if self.fit_transform == True:
            self.fit_transform = 'fit=1'  # convert True to first-order fit.
            fit_transform_handled = True
        elif self.fit_transform == False:
            self.fit_transform = ['use', '0', '0']  # convert False to exclusion of color index (rare).
            fit_transform_handled = True
        if self.fit_transform == 'fit=1':
            fixed_effect_var_list.append('CI')
            fit_transform_handled = True
        elif self.fit_transform == 'fit=2':
            fixed_effect_var_list.append('CI')
            fixed_effect_var_list.append('CI2')
            fit_transform_handled = True
        elif isinstance(self.fit_transform, list):
            if self.fit_transform[0] == 'use':
                try:
                    transform_1 = float(self.fit_transform[1])
                    transform_2 = float(self.fit_transform[2])
                except ValueError:
                    raise ValueError('#FIT_TRANSFORM use value(s) cannot be converted to float value(s).')
                dep_var_offset += transform_1 * self.df_used_comps_only['CI'] +\
                    transform_2 * self.df_used_comps_only['CI2']
                msg = ' Transform (Color Index) not fit: 1st, 2nd order values fixed at' +\
                      ' {0:.3f}'.format(transform_1) + ' {0:.3f}'.format(transform_2)
                print(msg)
                fit_summary_lines.append(msg)
                fit_transform_handled = True
        if not fit_transform_handled:
            print(' >>>>> WARNING: _prep_and_do_regression() cannot interpret #FIT_TRANSFORM choice.')

        # Handle extinction (ObsAirmass) options separately here:
        fit_extinction_handled = False    # default, to be set to True when handled.
        if self.fit_extinction == True:
            fixed_effect_var_list.append('ObsAirmass')
            fit_extinction_handled = True
        if self.fit_extinction == False:
            extinction = self.state['extinction'][self.fits_filter]
            dep_var_offset += extinction * self.df_used_comps_only['ObsAirmass']
            msg = ' Extinction (ObsAirmass) not fit: value fixed at default of' +\
                  ' {0:.3f}'.format(extinction)
            print(msg)
            fit_summary_lines.append(msg)
            fit_extinction_handled = True
        if isinstance(self.fit_extinction, list):
            if self.fit_extinction[0] == 'use':
                try:
                    extinction = float(self.fit_extinction[1])
                except ValueError:
                    raise ValueError('#FIT_EXTINCTION use value cannot be converted to float value.')
                dep_var_offset += extinction * self.df_used_comps_only['ObsAirmass']
                msg = ' Extinction (ObsAirmass) not fit: value fixed at control.txt value of' +\
                      ' {0:.3f}'.format(extinction)
                print(msg)
                fit_summary_lines.append(msg)
                fit_extinction_handled = True
        if not fit_extinction_handled:
            print(' >>>>> WARNING: _prep_and_do_regression() cannot interpret #FIT_EXTINCTION choice.')

        # Build all other fixed-effect (x) variable lists and dep-var offsets:
        if self.fit_vignette:
            fixed_effect_var_list.append('Vignette')
        if self.fit_xy:
            fixed_effect_var_list.extend(['X1024', 'Y1024'])
        if self.fit_jd:
            fixed_effect_var_list.append('JD_fract')
        if len(fixed_effect_var_list) == 0:
            fixed_effect_var_list = ['JD_fract']  # as statsmodels requires >= 1 fixed-effect variable.

        # TEST ONLY #############################################################################
        # Remove later: ADD Sloan g-i color index as variable CIGI:
        # self.df_used_comps_only['CI'] = self.df_used_comps_only['r'] - self.df_used_comps_only['i']
        # self.df_used_comps_only['CI2'] = [ci ** 2 for ci in self.df_used_comps_only['CI']]
        # self.df_used_comps_only['CIGI'] = self.df_used_comps_only['g'] - self.df_used_comps_only['i']
        # self.df_used_comps_only['CIGI2'] = [cigi ** 2 for cigi in self.df_used_comps_only['CIGI']]
        # fixed_effect_var_list.append('CI')
        # fixed_effect_var_list.append('CI2')
        # fixed_effect_var_list.append('CIGI')
        # fixed_effect_var_list.append('CIGI2')
        # END TEST ONLY. #########################################################################

        # Build 'random-effect' variable:
        random_effect_var_name = 'FITSfile'  # cirrus effect is per-image

        # Build dependent (y) variable:
        self.df_used_comps_only[self.dep_var_name] = self.df_used_comps_only['InstMag'] - dep_var_offset

        # Execute regression:
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
        self.mm_fit = MixedModelFit(data=self.df_used_comps_only,
                                    dep_var=self.dep_var_name,
                                    fixed_vars=fixed_effect_var_list,
                                    group_var=random_effect_var_name)
        print(self.mm_fit.statsmodels_object.summary())
        print('sigma =', '{0:.1f}'.format(1000.0 * self.mm_fit.sigma), 'mMag.')
        if not self.mm_fit.converged:
            msg = ' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.'
            print(msg)
            fit_summary_lines.append(msg)
        write_text_file('fit_summary.txt',
                        'Regression for directory ' + get_context()[0] + '\n\n' +
                        '\n'.join(fit_summary_lines) +
                        self.mm_fit.statsmodels_object.summary().as_text() +
                        '\n\nsigma = ' + '{0:.1f}'.format(1000.0 * self.mm_fit.sigma) + ' mMag.')

    def _calc_mp_mags(self):
        bogus_cat_mag = 0.0  # we'll need this below, to correct raw predictions.
        self.df_used_mps_only['CatMag'] = bogus_cat_mag  # totally bogus local value, corrected for later.
        self.df_used_mps_only['CI'] = self.mp_ci
        self.df_used_mps_only['CI2'] = self.mp_ci ** 2
        raw_predictions = self.mm_fit.predict(self.df_used_mps_only, include_random_effect=True)

        # Compute dependent-variable offsets for MP:
        dep_var_offsets = pd.Series(len(self.df_used_mps_only) * [0.0], index=raw_predictions.index)
        if self.fit_transform == False:
            dep_var_offsets += self.transform_fixed * self.df_used_mps_only['CI']
        if self.fit_extinction == False:
            dep_var_offsets += self.state['extinction'][self.fits_filter] * \
                               self.df_used_mps_only['ObsAirmass']
        if isinstance(self.fit_extinction, list):
            if self.fit_extinction[0] == 'use':
                extinction = float(self.fit_extinction[1])  # no try/except (was prev. converted).
                dep_var_offsets += extinction * self.df_used_mps_only['ObsAirmass']

        # Extract best MP mag (in Sloan r):
        mp_mags = self.df_used_mps_only['InstMag'] \
            - dep_var_offsets - raw_predictions + bogus_cat_mag  # correct for use of bogus cat mag.
        df_mp_mags = pd.DataFrame({'MP_Mags': mp_mags}, index=list(mp_mags.index))
        df_mp_mags = pd.merge(left=df_mp_mags,
                              right=self.df_used_mps_only.loc[:, ['JD_mid', 'FITSfile',
                                                                  'InstMag', 'InstMagSigma']],
                              how='left', left_index=True, right_index=True, sort=False)
        return df_mp_mags


def make_session_diagnostic_plots(model, df_model, mp_color_ri, state, user_selections):
    """  Display and write to file several diagnostic plots, to help decide which obs, comps, images
         might need removal by editing control file.
    :param model: mixed model summary object. [photrix.MixedModelFit object]
    :param df_model: dataframe of all data including UseInModel (user selection) column. [pandas DataFrame]
    :param mp_color_ri: Sloan r-i color of minor planet target. [float]
    :param state: session state for this observing session [dict]
    :param user_selections: comp selection criteria, used for drawing limits on plots [python dict]
    :return: [None]
    """
    this_directory, mp_string, an_string = get_context()

    # Delete any previous image files from current directory:
    image_filenames = [f for f in os.listdir('.')
                       if f.startswith(SESSION_IMAGE_PREFIX) and f.endswith('.png')]
    for f in image_filenames:
        os.remove(f)

    # Wrangle needed data into convenient forms:
    df_plot = pd.merge(left=df_model.loc[df_model['UseInModel'], :].copy(),
                       right=model.mm_fit.df_observations,
                       how='left', left_index=True, right_index=True, sort=False)  # add col 'Residuals'.
    is_comp_obs = (df_plot['Type'] == 'Comp')
    df_plot_comp_obs = df_plot.loc[is_comp_obs, :]
    # df_plot_mp_obs = df_plot.loc[(~ is_comp_obs), :]
    df_image_effect = model.mm_fit.df_random_effects
    df_image_effect.rename(columns={"GroupName": "FITSfile", "Group": "ImageEffect"}, inplace=True)
    # intercept = model.mm_fit.df_fixed_effects.loc['Intercept', 'Value']
    # jd_slope = model.mm_fit.df_fixed_effects.loc['JD_fract', 'Value']  # undefined if FIT_JD is False.
    sigma = model.mm_fit.sigma
    if 'CI' in model.mm_fit.df_fixed_effects.index:
        transform = model.mm_fit.df_fixed_effects.loc['CI', 'Value']  # if fit in model
    else:
        transform = TRANSFORM_CLEAR_SR_SR_SI  # default if not fit in model (normal case)
    # if model.fit_jd:
    #     jd_coefficient = model.mm_fit.df_fixed_effects.loc['JD_fract', 'Value']
    # else:
    #     jd_coefficient = 0.0
    comp_ids = df_plot_comp_obs['SourceID'].drop_duplicates()
    n_comps = len(comp_ids)
    # comp_color, mp_color = 'dimgray', 'orangered'
    # obs_colors = [comp_color if i is True else mp_color for i in is_comp_obs]
    jd_floor = floor(min(df_model['JD_mid']))
    # obs_jd_fract = df_plot['JD_mid'] - jd_floor
    xlabel_jd = 'JD(mid)-' + str(jd_floor)

    # ################ SESSION FIGURE 1: Q-Q plot of mean comp effects (1 pt per comp star used in model),
    #    code heavily adapted from photrix.process.SkyModel.plots():
    window_title = 'Q-Q Plot (by comp):  MP ' + mp_string + '   AN ' + an_string
    page_title = 'MP ' + mp_string + '   AN ' + an_string + '   ::   Q-Q plot by comp (mean residual)'
    plot_annotation = str(n_comps) + ' comps used in model.' + \
        '\n(tags: comp SourceID)'
    df_y = df_plot_comp_obs.loc[:, ['SourceID', 'Residual']].groupby(['SourceID']).mean()
    df_y = df_y.sort_values(by='Residual')
    y_data = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y.index.values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
                          SESSION_IMAGE_PREFIX + '1_QQ_comps.png')

    # ################ SESSION FIGURE 2: Q-Q plot of comp residuals (one point per comp obs),
    #    code heavily adapted from photrix.process.SkyModel.plots():
    window_title = 'Q-Q Plot (by comp observation):  MP ' + mp_string + '   AN ' + an_string
    page_title = 'MP ' + mp_string + '   AN ' + an_string + '   ::   Q-Q plot by comp observation'
    plot_annotation = str(len(df_plot_comp_obs)) + ' observations of ' + \
        str(n_comps) + ' comps used in model.' + \
        '\n (tags: observation Serial numbers)'
    df_y = df_plot_comp_obs.loc[:, ['Serial', 'Residual']]
    df_y = df_y.sort_values(by='Residual')
    y_data = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y['Serial'].values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
                          SESSION_IMAGE_PREFIX + '2_QQ_obs.png')

    # ################ SESSION FIGURE 3: Catalog and Time plots:
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(11, 8.5))  # (width, height) in "inches", was 15,9
    fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
    fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
    fig.suptitle('MP ' + mp_string + '   AN ' + an_string + '     ::     catalog and time plots',
                 color='darkblue', fontsize=20)
    fig.canvas.set_window_title('Catalog and Time Plots: ' + 'MP ' + mp_string + '   AN ' + an_string)
    subplot_text = 'rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
    fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=12, color='dimgray')

    # Catalog mag uncertainty plot (comps only, one point per comp, x=cat r mag, y=cat r uncertainty):
    ax = axes[0, 0]
    make_9_subplot(ax, 'Catalog Mag Uncertainty (dr)', 'Catalog Mag (r)', 'mMag', '', False,
                   x_data=df_plot_comp_obs['r'], y_data=df_plot_comp_obs['dr'])

    # Catalog color plot (comps only, one point per comp, x=cat r mag, y=cat color (r-i)):
    ax = axes[0, 1]
    make_9_subplot(ax, 'Catalog Color Index', 'Catalog Mag (r)', 'CI Mag', '', zero_line=False,
                   x_data=df_plot_comp_obs['r'], y_data=(df_plot_comp_obs['r'] - df_plot_comp_obs['i']))
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=len(model.df_mp_mags) * [mp_color_ri],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)  # add MP points.

    # Inst Mag plot (comps only, one point per obs, x=cat r mag, y=InstMagSigma):
    ax = axes[0, 2]
    make_9_subplot(ax, 'Instrument Magnitude Uncertainty', 'Catalog Mag (r)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['r'], y_data=df_plot_comp_obs['InstMagSigma'])
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=model.df_mp_mags['InstMagSigma'],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)  # add MP points.

    # Cirrus plot (comps only, one point per image, x=JD_fract, y=Image Effect):
    ax = axes[1, 0]
    df_this_plot = pd.merge(df_image_effect, df_plot_comp_obs.loc[:, ['FITSfile', 'JD_fract']],
                            how='left', on='FITSfile', sort=False).drop_duplicates()
    make_9_subplot(ax, 'Image effect (cirrus plot)', xlabel_jd, 'mMag', '', False,
                   x_data=df_this_plot['JD_fract'], y_data=1000.0 * df_this_plot['ImageEffect'],
                   alpha=1.0, jd_locators=True)
    ax.invert_yaxis()  # per custom of plotting magnitudes brighter=upward

    # SkyADU plot (comps only, one point per obs: x=JD_fract, y=SkyADU):
    ax = axes[1, 1]
    make_9_subplot(ax, 'SkyADU vs time', xlabel_jd, 'ADU', '', False,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=df_plot_comp_obs['SkyADU'],
                   jd_locators=True)

    # FWHM plot (comps only, one point per obs: x=JD_fract, y=FWHM):
    ax = axes[1, 2]
    make_9_subplot(ax, 'FWHM vs time', xlabel_jd, 'FWHM (pixels)', '', False,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=df_plot_comp_obs['FWHM'],
                   jd_locators=True)

    # InstMagSigma plot (comps only, one point per obs; x=JD_fract, y=InstMagSigma):
    ax = axes[2, 0]
    make_9_subplot(ax, 'Inst Mag Sigma vs time', xlabel_jd, 'mMag', '', False,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=1000.0 * df_plot_comp_obs['InstMagSigma'],
                   jd_locators=True)

    # Obs.Airmass plot (comps only, one point per obs; x=JD_fract, y=ObsAirmass):
    ax = axes[2, 1]
    make_9_subplot(ax, 'Obs.Airmass vs time', xlabel_jd, 'Obs.Airmass', '', False,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=df_plot_comp_obs['ObsAirmass'],
                   jd_locators=True)

    # Session Lightcurve plot (comps only, one point per obs; x=JD_fract, y=MP best magnitude):
    ax = axes[2, 2]
    make_9_subplot(ax, 'MP Lightcurve for this session', xlabel_jd, 'Mag (r)', '', False,
                   x_data=model.df_mp_mags['JD_mid'] - jd_floor, y_data=model.df_mp_mags['MP_Mags'],
                   alpha=1.0, jd_locators=True)
    ax.errorbar(x=model.df_mp_mags['JD_mid'] - jd_floor, y=model.df_mp_mags['MP_Mags'],
                yerr=model.df_mp_mags['InstMagSigma'], fmt='none', color='black',
                linewidth=0.5, capsize=3, capthick=0.5, zorder=-100)
    ax.invert_yaxis()  # per custom of plotting magnitudes brighter=upward

    plt.show()
    fig.savefig(SESSION_IMAGE_PREFIX + '3_Catalog_and_Time.png')

    # ################ SESSION FIGURE 4: Residual plots:
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(11, 8.5))  # (width, height) in "inches", was 15, 9
    fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
    fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
    fig.suptitle('MP ' + mp_string + '   AN ' + an_string + '     ::     residual plots',
                 color='darkblue', fontsize=20)
    fig.canvas.set_window_title('Residual Plots: ' + 'MP ' + mp_string + '   AN ' + an_string)
    subplot_text = str(len(df_plot_comp_obs)) + ' obs   ' +\
        str(n_comps) + ' comps    ' +\
        'sigma=' + '{0:.0f}'.format(1000.0 * sigma) + ' mMag' +\
        (12 * ' ') + ' rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
    fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=12, color='dimgray')

    # Comp residual plot (comps only, one point per obs: x=catalog r mag, y=model residual):
    ax = axes[0, 0]
    make_9_subplot(ax, 'Model residual vs r (catalog)', 'Catalog Mag (r)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['r'], y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=len(model.df_mp_mags) * [0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)  # add MP points.
    draw_x_line(ax, user_selections['min_r_mag'])
    draw_x_line(ax, user_selections['max_r_mag'])

    # Comp residual plot (comps only, one point per obs: x=raw Instrument Mag, y=model residual):
    ax = axes[0, 1]
    make_9_subplot(ax, 'Model residual vs raw Instrument Mag', 'Raw instrument mag', 'mMag', '', True,
                   x_data=df_plot_comp_obs['InstMag'], y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter(x=model.df_mp_mags['InstMag'], y=len(model.df_mp_mags) * [0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)  # add MP points.

    # Comp residual plot (comps only, one point per obs: x=catalog r-i color, y=model residual):
    ax = axes[0, 2]
    make_9_subplot(ax, 'Model residual vs Color Index (cat)', 'Catalog Color (r-i)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['r'] - df_plot_comp_obs['i'],
                   y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter(x=[mp_color_ri], y=[0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)  # add MP points.

    # Comp residual plot (comps only, one point per obs: x=Julian Date fraction, y=model residual):
    ax = axes[1, 0]
    make_9_subplot(ax, 'Model residual vs JD', xlabel_jd, 'mMag', '', True,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=1000.0 * df_plot_comp_obs['Residual'],
                   jd_locators=True)

    # Comp residual plot (comps only, one point per obs: x=ObsAirmass, y=model residual):
    ax = axes[1, 1]
    make_9_subplot(ax, 'Model residual vs Obs.Airmass', 'ObsAirmass', 'mMag', '', True,
                   x_data=df_plot_comp_obs['ObsAirmass'], y_data=1000.0 * df_plot_comp_obs['Residual'])

    # Comp residual plot (comps only, one point per obs: x=Sky Flux (ADUs), y=model residual):
    ax = axes[1, 2]
    make_9_subplot(ax, 'Model residual vs Sky Flux', 'Sky Flux (ADU)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['SkyADU'], y_data=1000.0 * df_plot_comp_obs['Residual'])

    # Comp residual plot (comps only, one point per obs: x=X in images, y=model residual):
    ax = axes[2, 0]
    make_9_subplot(ax, 'Model residual vs X in image', 'X from center (pixels)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['X1024'] * 1024.0, y_data=1000.0 * df_plot_comp_obs['Residual'])
    draw_x_line(ax, 0.0)

    # Comp residual plot (comps only, one point per obs: x=Y in images, y=model residual):
    ax = axes[2, 1]
    make_9_subplot(ax, 'Model residual vs Y in image', 'Y from center (pixels)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['Y1024'] * 1024.0, y_data=1000.0 * df_plot_comp_obs['Residual'])
    draw_x_line(ax, 0.0)

    # Comp residual plot (comps only, one point per obs: x=vignette (dist from center), y=model residual):
    ax = axes[2, 2]
    make_9_subplot(ax, 'Model residual vs distance from center', 'dist from center (pixels)', 'mMag',
                   '', True,
                   x_data=1024*np.sqrt(df_plot_comp_obs['Vignette']),
                   y_data=1000.0 * df_plot_comp_obs['Residual'])

    plt.show()
    fig.savefig(SESSION_IMAGE_PREFIX + '4_Residuals.png')

    # ################ FIGURE(S) 5: Variability plots:
    # Several comps on a subplot, vs JD, normalized by (minus) the mean of all other comps' responses.
    # Make df_offsets (one row per obs, at first with only raw offsets):
    make_comp_variability_plots(df_plot_comp_obs, xlabel_jd, transform, sigma,
                                image_prefix=SESSION_IMAGE_PREFIX)


def write_control_txt_stub(this_directory, log_file, df):
    fullpath = os.path.join(this_directory, MP_PHOT_CONTROL_FILENAME)
    if os.path.exists(fullpath):
        return

    defaults = DO_PHOT_COMP_SELECTION_DEFAULTS
    df['SecondsRelative'] = [(utc - df['UTC_mid'].min()).total_seconds() for utc in df['UTC_mid']]
    i_earliest = df['SecondsRelative'].nsmallest(n=1).index[0]
    i_latest = df['SecondsRelative'].nlargest(n=1).index[0]
    earliest_filename = df.loc[i_earliest, 'Filename']
    latest_filename = df.loc[i_latest, 'Filename']

    def yes_no(true_false):
        return 'Yes' if true_false else 'No'

    lines = [';----- This is ' + MP_PHOT_CONTROL_FILENAME + ' for directory:\n;      ' + this_directory,
             ';',
             ';===== MP LOCATIONS BLOCK ===========================================',
             ';===== Enter before make_dfs() ======================================',
             ';      MP x,y positions for aperture photometry:',
             '#MP_LOCATION  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
             'early filename, change if needed',
             '#MP_LOCATION  ' + latest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
             ' late filename, change if needed',
             ';',
             ';===== MP RI COLOR BLOCK ============================================',
             ';===== Enter before do_mp_phot(), get from do_color. ================',
             '#MP_RI_COLOR ' + '{0:+.3f}'.format(DEFAULT_MP_RI_COLOR) +
             '  Default MP color  ;  get by running do_color(), or leave as default=' +
             '{0:+.3f}'.format(DEFAULT_MP_RI_COLOR),
             ';',
             ';===== SELECTION CRITERIA BLOCK =====================================',
             ';===== Enter before do_mp_phot() ====================================',
             ';      Selection criteria for comp stars, observations, images:',
             ';#COMP  nnnn nn,   nnn        ; to omit comp(s) by comp ID',
             ';#OBS nnn,nnnn nnnn   nn      ; to omit observation(s) by Serial number',
             ';#IMAGE  MP_mmmm-00nn-Clear   ; to omit one FITS image (.fts at end optional)',
             (';#MIN_R_MAG ' + str(defaults['min_catalog_r_mag'])).ljust(30) +
             '; default=' + str(defaults['min_catalog_r_mag']),
             (';#MAX_R_MAG ' + str(defaults['max_catalog_r_mag'])).ljust(30) +
             '; default=' + str(defaults['max_catalog_r_mag']),
             (';#MAX_CATALOG_DR_MMAG ' + str(defaults['max_catalog_dr_mag'])).ljust(30) +
             '; default=' + str(defaults['max_catalog_dr_mag']),
             (';#MIN_SLOAN_RI_COLOR ' + str(defaults['min_catalog_ri_color'])).ljust(30) +
             '; default=' + str(defaults['min_catalog_ri_color']),
             (';#MAX_SLOAN_RI_COLOR ' + str(defaults['max_catalog_ri_color'])).ljust(30) +
             '; default=' + str(defaults['max_catalog_ri_color']),
             ';',
             ';===== REGRESSION OPTIONS BLOCK =====================================',
             ';===== Enter before do_mp_phot(): ===================================',
             ';----- OPTIONS for regression model:',

             ';Choices for #FIT_TRANSFORM: Fit=1; '
             + 'Fit=2; Use 0.2 0.4 [=tr1 & tr2 values]; Yes->Fit=1; No->Use 0 0',
             ';#FIT_TRANSFORM  Fit=2'.ljust(30) + '; default= Fit=2',

             ';Choices for #FIT_EXTINCTION: Yes->do fit; No->use default extinction;'
             + ' Use +0.34-> use this value ',
             (';#FIT_EXTINCTION ' + yes_no(DEFAULT_MODEL_OPTIONS['fit_extinction'])).ljust(30) +
             '; default='
             + yes_no(DEFAULT_MODEL_OPTIONS['fit_extinction']) + ' // choose Yes or No  (case-insensitive)',

             (';#FIT_VIGNETTE ' + yes_no(DEFAULT_MODEL_OPTIONS['fit_vignette'])).ljust(30) + '; default='
             + yes_no(DEFAULT_MODEL_OPTIONS['fit_vignette']) + ' // choose Yes or No  (case-insensitive)',
             (';#FIT_XY ' + yes_no(DEFAULT_MODEL_OPTIONS['fit_xy'])).ljust(30) + '; default='
             + yes_no(DEFAULT_MODEL_OPTIONS['fit_xy']) + ' // choose Yes or No  (case-insensitive)',
             (';#FIT_JD ' + yes_no(DEFAULT_MODEL_OPTIONS['fit_jd'])).ljust(30) + '; default='
             + yes_no(DEFAULT_MODEL_OPTIONS['fit_jd']) + ' // choose Yes or No  (case-insensitive)',
             ';'
             ]
    lines = [line + '\n' for line in lines]
    fullpath = os.path.join(this_directory, MP_PHOT_CONTROL_FILENAME)
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(lines)
            log_file.write('New ' + MP_PHOT_CONTROL_FILENAME + ' file written.\n')


def write_text_file(filename, lines):
    this_directory, mp_string, an_string = get_context()
    fullpath = os.path.join(this_directory, filename)
    with open(fullpath, 'w') as f:
        f.write(lines)


def write_canopus_file(model):
    """  Write file that can be imported into Canopus, to populate observation data of one Canopus session.
    :param model: mixed model summary object. [photrix.MixedModelFit object]
    :return: [None]
    """
    this_directory, mp_string, an_string = get_context()
    df = model.df_mp_mags
    fulltext = '\n'.join(['{0:.6f}'.format(jd) + ',' + '{0:.4f}'.format(mag) + ',' + '{0:.4f}'.format(s) +
                          ',' + f
                          for (jd, mag, s, f) in zip(df['JD_mid'], df['MP_Mags'], df['InstMagSigma'],
                                                     df['FITSfile'])])
    fullpath = os.path.join(this_directory, 'canopus_MP_' + mp_string + '_' + an_string + '.txt')
    with open(fullpath, 'w') as f:
        f.write(fulltext)


def write_alcdef_file(model, mp_color_ri, source_string_ri):
    """  Write file that can be uploaded to ALCDEF, to make one session's observation available to public.
    :param model: mixed model summary object. [photrix.MixedModelFit object]
    :param mp_color_ri: MP color in Sloan r-i. [float]
    :param source_string_ri: tells where mp_color_ri value came from. [string]
    :return: [None]
    """
    this_directory, mp_string, an_string = get_context()
    mpfile_names = all_mpfile_names()
    name_list = [name for name in mpfile_names if name.startswith('MP_' + mp_string)]
    if len(name_list) <= 0:
        print(' >>>>> ERROR: No MPfile can be found for MP', mp_string, '--> NO ALCDEF file written')
        return
    if len(name_list) >= 2:
        print(' >>>>> ERROR: Multiple MPfiles were found for MP', mp_string, '--> NO ALCDEF file written')
        return
    mpfile = MPfile(name_list[0])
    site_data = DSW_SITE_DATA
    df = model.df_mp_mags

    # Build data that will go into file:
    lines = list()
    lines.append('# ALCDEF file for MP ' + mp_string + '  AN ' + an_string)
    lines.append('STARTMETADATA')
    lines.append('REVISEDDATA=FALSE')
    lines.append('OBJECTNUMBER=' + mp_string)
    lines.append('OBJECTNAME=' + mpfile.name)
    lines.append('ALLOWSHARING=TRUE')
    # lines.append('MPCDESIG=')
    lines.append('CONTACTNAME=' + ALCDEF_DATA['contactname'])
    lines.append('CONTACTINFO=' + ALCDEF_DATA['contactinfo'])
    lines.append('OBSERVERS=' + ALCDEF_DATA['observers'])
    lines.append('OBSLONGITUDE=' + '{0:.4f}'.format(site_data['longitude']))
    lines.append('OBSLATITUDE=' + '{0:.4f}'.format(site_data['latitude']))
    lines.append('FACILITY=' + site_data['facility'])
    lines.append('MPCCODE=' + site_data['mpccode'])
    # lines.append('PUBLICATION=')
    jd_session_start = min(df['JD_mid'])
    jd_session_end = max(df['JD_mid'])
    jd_session_mid = (jd_session_start + jd_session_end) / 2.0
    utc_session_mid = datetime_utc_from_jd(jd_session_mid)  # (needed below)
    dt_split = utc_session_mid.isoformat().split('T')
    lines.append('SESSIONDATE=' + dt_split[0])
    lines.append('SESSIONTIME=' + dt_split[1].split('+')[0].split('.')[0])
    lines.append('FILTER=' + ALCDEF_DATA['filter'])
    lines.append('MAGBAND=' + ALCDEF_DATA['magband'])
    lines.append('LTCTYPE=NONE')
    # lines.append('LTCDAYS=0')
    # lines.append('LTCAPP=NONE')
    lines.append('REDUCEDMAGS=NONE')
    session_dict = mpfile.eph_from_utc(utc_session_mid)
    # earth_mp_au = session_dict['Delta']
    # sun_mp_au = session_dict['R']
    # reduced_mag_correction = -5.0 * log10(earth_mp_au * sun_mp_au)
    #  lines.append('UCORMAG=' + '{0:.4f}'.format(reduced_mag_correction))  # removed to avoid confusion.
    lines.append('OBJECTRA=' + ra_as_hours(session_dict['RA']).rsplit(':', 1)[0])
    lines.append('OBJECTDEC=' + ' '.join(dec_as_hex(round(session_dict['Dec'])).split(':')[0:2]))
    lines.append('PHASE=+' + '{0:.1f}'.format(abs(session_dict['Phase'])))
    lines.append('PABL=' + '{0:.1f}'.format(abs(session_dict['PAB_longitude'])))
    lines.append('PABB=' + '{0:.1f}'.format(abs(session_dict['PAB_latitude'])))
    lines.append('COMMENT=These results from submitter\'s '
                 'ATLAS-refcat2 based workflow: see SAS Symposium 2020.')
    lines.append('COMMENT=This session used ' +
                 str(len(model.df_used_comps_only['SourceID'].drop_duplicates())) + ' comp stars. '
                 'COMPNAME etc lines are omitted.')
    lines.append('CICORRECTION=TRUE')
    lines.append('CIBAND=SRI')
    lines.append('CITARGET=' + '{0:+.3f}'.format(mp_color_ri) + '  # origin: ' + source_string_ri)
    lines.append('DELIMITER=PIPE')
    lines.append('ENDMETADATA')
    data_lines = ['DATA=' + '|'.join(['{0:.6f}'.format(jd), '{0:.3f}'.format(mag), '{0:.3f}'.format(sigma)])
                  for (jd, mag, sigma) in zip(df['JD_mid'], df['MP_Mags'], df['InstMagSigma'])]
    lines.extend(data_lines)
    lines.append('ENDDATA')

    # Write the file and exit:
    fulltext = '\n'.join(lines) + '\n'
    fullpath = os.path.join(this_directory, 'alcdef_MP_' + mp_string + '_' + an_string + '.txt')
    with open(fullpath, 'w') as f:
        f.write(fulltext)


def combine_alcdef(mp, apparition_year):
    """ Append into one file all ALCDEF files in this MP Campaign directory, write to MP directory.
    :param mp: MP number. A subdirectory 'MP_[mpnumber]' must exist in the MP TOP DIRECTORY. [int or str]
    :param apparition_year
    :return: None. Writes new file to MP's directory.
    """
    mpdir = os.path.join(MP_PHOT_TOP_DIRECTORY, 'MP_' + str(mp))
    an_subdirs = [f.path for f in os.scandir(mpdir)
                  if (f.is_dir() and f.name.startswith('AN') and len(f.name) == 10)]
    all_lines = []
    n_files_read = 0
    for sd in an_subdirs:
        filenames = [f.path for f in os.scandir(sd)
                     if (f.name.startswith('alcdef_MP_') and f.name.endswith('.txt'))]
        for filename in filenames:
            with open(filename, 'r') as f:
                lines = f.readlines()
            all_lines.append('#====================================='
                             '========================================\n')
            all_lines.extend(lines)
            print('{0:6d}'.format(len(lines)), 'lines read from', filename)
            n_files_read += 1
    if n_files_read <= 0:
        print(' >>>>> WARNING: No ALCDEF files were found.')
        exit(0)

    # Write combined ALCDEF file to MP directory:
    fulltext = all_lines
    combined_filename = 'alcdef_combined_MP_' + str(mp) + '_' + str(apparition_year) + '.txt'
    fullpath = os.path.join(mpdir, combined_filename)
    with open(fullpath, 'w') as f:
        f.writelines(fulltext)
    print('Combined ALCDEF now at', fullpath, '= ', len(all_lines), 'lines.')


_____READING_LOG_and_CONTROL_FILES__________________________________ = 0


def get_context(log_filename=MP_PHOT_LOG_FILENAME):
    """ This is run at beginning of workflow functions (except start() or resume()) to orient the function.
    :param log_filename: name of log file from which to get context data. [string]
    :return: 3-tuple: (this_directory, mp_string, an_string) [3 strings]
    """
    this_directory = os.getcwd()
    if not os.path.isfile(log_filename):
        print(' >>>>> ERROR: no log file', log_filename ,
              'found ==> You probably need to run start() or resume().')
        return None
    log_file = open(log_filename, mode='r')  # for read only
    lines = log_file.readlines()
    log_file.close()
    if len(lines) < 3:
        return None
    if lines[0].strip().lower().replace('\\', '/').replace('//', '/') != \
            this_directory.strip().lower().replace('\\', '/').replace('//', '/'):
        print('Working directory does not match directory at top of log file.')
        return None
    mp_string = lines[1][3:].strip().upper()
    an_string = lines[2][3:].strip()
    return this_directory, mp_string, an_string


def read_mp_locations():
    """ Reads control file, parses exactly 2 #MP lines, returns filenames, x- and y-pixels.
    :return: mp_location_filenames [list of 2 strings], x_pixels, y_pixels [each list of 2 floats].
    """
    mp_location_key = '#MP_LOCATION'
    mp_location_filenames, x_pixels, y_pixels = [], [], []
    with open(MP_PHOT_CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines
        for line in lines:
            if len(mp_location_filenames) >= 2:
                break
            if line.upper().startswith(mp_location_key):
                words = line[len(mp_location_key):].rsplit(maxsplit=2)
                if any([w.startswith('[') for w in words]):
                    break
                mp_location_filenames.append(words[0].strip())
                try:
                    x_pixels.append(float(words[1]))
                    y_pixels.append(float(words[2]))
                except ValueError:
                    break
    if (len(mp_location_filenames), len(x_pixels), len(y_pixels)) != (2, 2, 2):
        mp_location_filenames, x_pixels, y_pixels = None, None, None
    return mp_location_filenames, x_pixels, y_pixels


def read_mp_ri_color():
    """  Gets minor planet's Sloan r-i color index from control file, or use default value if unreadable.
         Color index will typically have been calculated using do_color() applied to the first MP session.
    :return: tuple (color index (Sloan r-i), source of value). [2-tuple of float, string]
    """
    # First set defaults, to be overridden if valid color in fact read from file.
    mp_ri_color, mp_ri_color_source = DEFAULT_MP_RI_COLOR, 'Default MP color'
    with open(MP_PHOT_CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines
        for line in lines:
            if line.upper().startswith('#MP_RI_COLOR '):
                tokens = line[len('#MP_RI_COLOR '):].split(maxsplit=1)
                if len(tokens) < 2:
                    print(' >>>>> ERROR: Either color or color source is missing from ' + MP_PHOT_CONTROL_FILENAME)
                else:
                    ri_string = tokens[0]
                    mp_ri_color_source = tokens[1].strip()
                    try:
                        mp_ri_color = float(ri_string)  # parse color value.
                    except ValueError:
                        print(' >>>>> ERROR: cannot read #MP_RI_COLOR given in ' + MP_PHOT_CONTROL_FILENAME)
                        mp_ri_color, mp_ri_color_source = None, \
                            '[error in reading control file ' + MP_PHOT_CONTROL_FILENAME + ']'
        if not ((-0.25) <= mp_ri_color <= +0.75):
            print(' >>>>> WARNING: MP_RI_COLOR has unreasonable value of', '{0:.2f}'.format(mp_ri_color))
        if len(mp_ri_color_source) < 5:
            print(' >>>>> WARNING: MP_RI_COLOR source specification is missing or is too short.')
        return mp_ri_color, mp_ri_color_source


def read_selection_criteria(filename, defaults):
    """ Reads observation selection lines from control file, compiles lists of observations to remove
            (for use by do_color() and by apply_do_phot_selections()).
    :param filename: file name from which to read selection criteria, e.g., control.txt. [string]
    :param defaults: comp selection default values. [py dict]
    :return: criteria to remove observations from df_obs before use in model. [dict]
    """
    # Parse file:
    with open(filename, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines

    serial_list, comp_list, image_list = [], [], []
    # Preload values with defaults, to be overwritten if valid selection directive read from file:
    min_r_mag = defaults['min_catalog_r_mag']
    max_r_mag = defaults['max_catalog_r_mag']
    max_catalog_dr_mmag = defaults['max_catalog_dr_mag']
    min_sloan_ri_color = defaults['min_catalog_ri_color']
    max_sloan_ri_color = defaults['max_catalog_ri_color']

    for line in lines:
        content = line.strip().split(';')[0].strip()  # upper case, comments removed.
        content_upper = content.upper()
        if content_upper.startswith('#COMP'):
            values = content[len('#COMP'):].strip().replace(',', ' ').split()
            comp_list.extend(values)
        if content_upper.startswith('#OBS'):
            values = content[len('#OBS'):].strip().replace(',', ' ').split()
            serial_list.extend(values)
        if content_upper.startswith('#IMAGE'):
            image_filename = content[len('#IMAGE'):].strip()
            image_list.append(image_filename)
        if content_upper.startswith('#MIN_R_MAG'):
            try:
                min_r_mag = float(content[len('#MIN_R_MAG'):].strip())
            except ValueError:
                print(' >>>>> WARNING: #MIN_R_MAG in', filename, 'cannot be parsed as float; default used.')
        if content_upper.startswith('#MAX_R_MAG'):
            try:
                max_r_mag = float(content[len('#MAX_R_MAG'):].strip())
            except ValueError:
                print(' >>>>> WARNING: #MAX_R_MAG in', filename, 'cannot be parsed as float; default used.')
        if content_upper.startswith('#MAX_CATALOG_DR_MMAG'):
            try:
                max_catalog_dr_mmag = float(content[len('#MAX_CATALOG_DR_MMAG'):].strip())
            except ValueError:
                print(' >>>>> WARNING: #MAX_CATALOG_DR_MMAG in', filename, 'cannot be parsed as float; '
                      'default used.')
        if content_upper.startswith('#MIN_SLOAN_RI_COLOR'):
            try:
                min_sloan_ri_color = float(content[len('#MIN_SLOAN_RI_COLOR'):].strip())
            except ValueError:
                print(' >>>>> WARNING: #MIN_SLOAN_RI_COLOR in', filename, 'cannot be parsed as float; ' +
                      'default used.')
        if content_upper.startswith('#MAX_SLOAN_RI_COLOR'):
            try:
                max_sloan_ri_color = float(content[len('#MAX_SLOAN_RI_COLOR'):].strip())
            except ValueError:
                print(' >>>>> WARNING: #MAX_SLOAN_RI_COLOR in', filename, 'cannot be parsed as float; ' +
                      'default used.')
    return {'comps': comp_list, 'serials': serial_list, 'images': image_list,
            'min_r_mag': min_r_mag, 'max_r_mag': max_r_mag,
            'max_catalog_dr_mmag': max_catalog_dr_mmag,
            'min_sloan_ri_color': min_sloan_ri_color, 'max_sloan_ri_color': max_sloan_ri_color}


def read_regression_options(filename):
    """ Returns dict of options for SessionModel (mixed-model fit of comp data).
        Reads from filename, uses default for any option not read in from there.
        :param filename: file name from which options are read, e.g., 'control.txt'. [string]
        :return option_dict: options returned. [python dict]

        """
    # Parse file:
    with open(filename, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines
    option_dict = DEFAULT_MODEL_OPTIONS.copy()  # for those options not in filename.
    for line in lines:
        content = line.upper().strip().split(';')[0].strip()  # upper case, comments removed.
        option_keys = list(option_dict.keys())                # keys for dict
        target_keys = ['#' + k.upper() for k in option_keys]  # as in control file.
        for option_key, target_key in zip(option_keys, target_keys):
            if content.startswith(target_key):
                value = content[len(target_key):].strip().lower()
                this_value = None
                if value in ['yes', 'true']:
                    this_value = True
                if value in ['no', 'false']:
                    this_value = False
                # Extended options for FIT_TRANSFORM (esp. 'Use'):
                if target_key == '#FIT_TRANSFORM':
                    if value in ['fit=1', 'fit=2']:
                        this_value = value
                    if value.startswith('use '):
                        if len(value.split()) == 3:
                            this_value = value.split()
                # Extended options for FIT_EXTINCTION (i.e., 'Use'):
                if target_key == '#FIT_EXTINCTION':
                    if value.startswith('use '):
                        if len(value.split()) == 2:
                            this_value = value.split()
                if this_value is None:
                    print(' >>>>> WARNING:', filename, target_key, 'value not understood.')
                    option_dict[option_key] = None
                else:
                    option_dict[option_key] = this_value
    return option_dict


_____READING_DATA_FILES_____________________________________ = 0


def make_df_all(filters_to_include=None, comps_only=False, require_mp_obs_each_image=False):
    """ General, reusable function to merge df_obs_all, df_comps_all, and df_images_all. OK 20200702.
    :param filters_to_include: either one filter name, or a list of filters. Only observations in
                that filter or filters will be retained.
                Use None to include ALL filters [None, or string, or list of strings]
    :param comps_only: True iff MPs to be removed (only rows for comp stars remain). [boolean]
    :param require_mp_obs_each_image: True to remove all obs from images without MP observation. [boolean]
    NOTE: cannot have comps_only=True and require_mp_obs_per_image=True.
    :return: df_all, the master table of data, one row per observation. [pandas DataFrame]
    """
    if comps_only is True and require_mp_obs_each_image is True:
        print(' >>>>> ERROR, make_df_all(): comps_only and '
              'require_mp_obs_each_image may not both be True.')
        return None
    if isinstance(filters_to_include, str):
        filters_to_include = [filters_to_include]
    df_obs_all = read_df_obs_all()
    df_comps_all = read_df_comps_all()
    df_images_all = read_df_images_all()
    df_obs_selected = df_obs_all.copy()  # start with all obs, make selections below.
    if filters_to_include is None:
        filters_to_include = df_images_all['Filter'].drop_duplicates()

    # Keep only obs from images in requested filter(s):
    image_is_in_filter = [(f in filters_to_include) for f in df_images_all['Filter']]
    if sum(image_is_in_filter) <= 0:
        print(' >>>>> ERROR, make_df_all(): no images found for the chosen filter(s).')
        return None
    images_to_keep = list(df_images_all.loc[image_is_in_filter, 'FITSfile'])
    obs_to_keep = [(im in images_to_keep) for im in df_obs_selected['FITSfile']]
    df_obs_selected = df_obs_selected.loc[obs_to_keep, :]

    # Remove any obs from images having no MP obs, if requested:
    if require_mp_obs_each_image is True:
        obs_is_type_mp = list(df_obs_selected['Type'] == 'MP')
        images_with_mp_obs = list(df_obs_selected.loc[obs_is_type_mp, 'FITSfile'])
        obs_is_in_qualified_image = [(i in images_with_mp_obs) for i in df_obs_selected['FITSfile']]
        df_obs_selected = df_obs_selected.loc[obs_is_in_qualified_image, :]

    # Keep comp obs only, if requested (this has to go last, or above selections don't work):
    if comps_only is True:
        obs_is_comp = [t == 'Comp' for t in df_obs_selected['Type']]
        df_obs_selected = df_obs_selected.loc[obs_is_comp, :]

    # Perform merges:
    df_obs_and_comps = pd.merge(left=df_obs_selected, right=df_comps_all,
                                how='left', left_on='SourceID', right_on='CompID', sort=False)
    df_all = pd.merge(left=df_obs_and_comps, right=df_images_all,
                      how='left', on='FITSfile', sort=False)
    df_all.index = list(df_all['Serial'])
    return df_all


def read_df_obs_all():
    """  Simple utility to read df_obs.csv file and return the original DataFrame.
    :return: df_obs from make_dfs() [pandas Dataframe]
    """
    this_directory, _, _ = get_context()
    fullpath = os.path.join(this_directory, MP_PHOT_DF_OBS_ALL_FILENAME)
    df_obs = pd.read_csv(fullpath, sep=';', index_col=0)
    serials = [str(s) for s in df_obs['Serial']]  # ensure strings.
    df_obs.loc[:, 'Serial'] = serials
    df_obs.index = serials  # list, to ensure index not named.
    return df_obs


def read_df_images_all():
    """  Simple utility to read df_images.csv file and return the original DataFrame.
    :return: df_images from make_dfs() [pandas Dataframe]
    """
    this_directory, _, _ = get_context()
    fullpath = os.path.join(this_directory, MP_PHOT_DF_IMAGES_ALL_FILENAME)
    df_images = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_images


def read_df_comps_all():
    """  Simple utility to read df_comps.csv file and return the original DataFrame.
    :return: df_comps from make_dfs() [pandas Dataframe]
    """
    this_directory, _, _ = get_context()
    fullpath = os.path.join(this_directory, MP_PHOT_DF_COMPS_ALL_FILENAME)
    df_comps = pd.read_csv(fullpath, sep=';', index_col=0)
    comp_ids = [str(id) for id in df_comps['CompID']]  # ensure strings.
    df_comps.loc[:, 'CompID'] = comp_ids
    df_comps.index = comp_ids  # list, to ensure index not named.
    return df_comps


_____PLOTTING_and_SUPPORT________________________________________________ = 0


def make_qq_plot_fullpage(window_title, page_title, plot_annotation,
                          y_data, y_labels, filename, figsize=(11, 8.5)):  # was 12, 9
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)  # (width, height) in "inches"
    ax = axes  # not subscripted if just one subplot in Figure
    ax.set_title(page_title, color='darkblue', fontsize=20, pad=30)
    ax.set_xlabel('t (sigma.residuals = ' + str(round(pd.Series(y_data).std(), 1)) + ' mMag)')
    ax.set_ylabel('Residual (mMag)')
    ax.grid(True, color='lightgray', zorder=-1000)
    df_y = pd.DataFrame({'Y': y_data, 'Label': y_labels}).sort_values(by='Y')
    n = len(df_y)
    t_values = [norm.ppf((k - 0.5) / n) for k in range(1, n + 1)]
    ax.scatter(x=t_values, y=df_y['Y'], alpha=0.6, color='dimgray', zorder=+1000)
    # Label potential outliers:
    z_score_y = (df_y['Y'] - df_y['Y'].mean()) / df_y['Y'].std()
    is_outlier = (abs(z_score_y) >= 2.0)
    for x, y, label, add_label in zip(t_values, df_y['Y'], df_y['Label'], is_outlier):
        if add_label:
            ax.annotate(label, xy=(x, y), xytext=(4, -4),
                        textcoords='offset points', ha='left', va='top', rotation=-40)
    # Add reference line:
    x_low = 1.10 * min(t_values)
    x_high = 1.10 * max(t_values)
    y_low = x_low * df_y['Y'].std()
    y_high = x_high * df_y['Y'].std()
    ax.plot([x_low, x_high], [y_low, y_high], color='gray', zorder=-100, linewidth=1)
    # Finish FIGURE 1:
    fig.text(x=0.5, y=0.87, s=plot_annotation,
             verticalalignment='top', horizontalalignment='center', fontsize=12)
    fig.canvas.set_window_title(window_title)
    plt.show()
    fig.savefig(filename)


def make_9_subplot(ax, title, x_label, y_label, text, zero_line, x_data, y_data,
                   size=14, alpha=0.3, color='black', jd_locators=False):
    """ Make a subplot sized to 3x3 subplots/page. Frame w/labels only if x_data is None or y_data is None.
    :param ax: axis location of this subplot. [matplotlib Axes object]
    :param title: text atop the plot. [string]
    :param x_label: x-axis label. [string]
    :param y_label: y_axis label. [string]
    :param text: text inside top border (rarely used). [string]
    :param zero_line: iff True, plot a light line along Y=0. [boolean]
    :param x_data: vector of x-values to plot. [iterable of floats]
    :param y_data: vector of y-values to plot, must equal x_data in length. [iterable of floats]
    :param size: size of points to plot each x,y. [float, weird matplotlib scale]
    :param alpha: opacity of each point. [float, 0 to 1]
    :param color: name of color to plot each point. [string, matplotlib color]
    :param jd_locators: iff True, make x-axis ticks convenient to plotting JDs of one night. [boolean]
    :return: [no return value]
    """
    ax.set_title(title, loc='center', pad=-3)  # pad in points
    ax.set_xlabel(x_label, labelpad=-29)  # labelpad in points
    ax.set_ylabel(y_label, labelpad=-5)  # "
    ax.text(x=0.5, y=0.95, s=text,
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    if zero_line is True:
        ax.axhline(y=0, color='lightgray', linewidth=1, zorder=-100)
    if x_data is not None and y_data is not None:
        ax.scatter(x=x_data, y=y_data, s=size, alpha=alpha, color=color)
    if jd_locators:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))


def make_comp_variability_plots(df_plot_comp_obs, xlabel_jd, transform, sigma, image_prefix):
    """ Make pages of 9 subplots, 4 comps per subplot, showing jackknife comp residuals.
    :param df_plot_comp_obs: table of data from which to generate all subplots. [pandas Dataframe]
    :param xlabel_jd: label to go under x-axis (JD axis) for each subplot. [string]
    :param transform: Sloan r-i transform used for data. [float]
    :param sigma: std deviation of comp responses from fit. [float]
    :param image_prefix: prefix for output images' filenames. [string]
    :return:
    """
    _, mp_string, an_string = get_context()
    comp_ids = df_plot_comp_obs['SourceID'].drop_duplicates()
    n_comps = len(comp_ids)
    dict_list = []
    for comp_id in comp_ids:
        is_comp_id = df_plot_comp_obs['SourceID'] == comp_id
        serials = df_plot_comp_obs.loc[is_comp_id, 'Serial']
        jd_fracts = df_plot_comp_obs.loc[is_comp_id, 'JD_fract']
        inst_mags = df_plot_comp_obs.loc[is_comp_id, 'InstMag']
        r_catmag = df_plot_comp_obs.loc[is_comp_id, 'r']
        color_ri = r_catmag - df_plot_comp_obs.loc[is_comp_id, 'i']
        raw_offsets = inst_mags - r_catmag - transform * color_ri  # pd Series
        for i in range(len(serials)):
            comp_dict = dict()
            comp_dict['SourceID'] = comp_id
            comp_dict['Serial'] = serials.iloc[i]
            comp_dict['JD_fract'] = jd_fracts.iloc[i]
            comp_dict['RawOffset'] = raw_offsets.iloc[i]
            dict_list.append(comp_dict)
    df_comp_offsets = pd.DataFrame(data=dict_list)
    df_comp_offsets.index = df_comp_offsets['Serial']

    # Normalize offsets by subtracting mean of *other* comps' offsets at each JD_fract:
    all_jd_fracts = df_comp_offsets['JD_fract'].copy().drop_duplicates().sort_values()
    df_comp_offsets['NormalizedOffset'] = None
    df_comp_offsets['LatestNormalizedOffset'] = None
    for comp_id in comp_ids:
        is_comp_id = (df_comp_offsets['SourceID'] == comp_id)
        valid_jd_fracts = []
        for jd_fract in all_jd_fracts:
            is_this_jd_fract = (df_comp_offsets['JD_fract'] == jd_fract)
            mean_other_offsets = df_comp_offsets.loc[(~is_comp_id) & is_this_jd_fract, 'RawOffset'].mean()
            is_this_obs = is_comp_id & is_this_jd_fract
            this_raw_offset = df_comp_offsets.loc[is_this_obs, 'RawOffset'].mean()
            normalized_offset = this_raw_offset - mean_other_offsets
            df_comp_offsets.loc[is_this_obs, 'NormalizedOffset'] = normalized_offset
            if not np.isnan(normalized_offset):
                valid_jd_fracts.append(jd_fract)
        is_latest_jd_fract = df_comp_offsets['JD_fract'] == valid_jd_fracts[-1]
        latest_normalized_offset = df_comp_offsets.loc[is_comp_id & is_latest_jd_fract, 'NormalizedOffset']
        df_comp_offsets.loc[is_comp_id, 'LatestNormalizedOffset'] = latest_normalized_offset.iloc[0]

    df_comp_offsets = df_comp_offsets.sort_values(by=['LatestNormalizedOffset', 'SourceID', 'JD_fract'],
                                                  ascending=[False, True, True])
    df_plot_index = df_comp_offsets[['SourceID']].drop_duplicates()
    df_plot_index['PlotIndex'] = range(len(df_plot_index))
    df_comp_offsets = pd.merge(left=df_comp_offsets, right=df_plot_index,
                               how='left', on='SourceID', sort=False)

    # Plot the normalized offsets vs JD for each comp_id, 4 comp_ids to a subplot:
    n_cols, n_rows = 3, 3
    n_plots_per_figure = n_cols * n_rows
    n_comps_per_plot = 4
    n_plots = ceil(n_comps / n_comps_per_plot)
    plot_colors = ['r', 'g', 'm', 'b']
    n_figures = ceil(n_plots / n_plots_per_figure)
    jd_range = max(all_jd_fracts) - min(all_jd_fracts)
    jd_low_limit = min(all_jd_fracts) - 0.05 * jd_range
    jd_high_limit = max(all_jd_fracts) + 0.40 * jd_range
    normalized_offset_mmag = 1000.0 * df_comp_offsets['NormalizedOffset']
    offset_range = max(normalized_offset_mmag) - min(normalized_offset_mmag)
    offset_low_limit = min(normalized_offset_mmag) - 0.05 * offset_range
    offset_high_limit = max(normalized_offset_mmag) + 0.05 * offset_range
    plotted_comp_ids = []
    for i_figure in range(n_figures):
        n_plots_remaining = n_plots - (i_figure * n_plots_per_figure)
        n_plots_this_figure = min(n_plots_remaining, n_plots_per_figure)
        if n_plots_this_figure >= 1:
            # Start new Figure:
            fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(11, 8.5))  # was 15, 9
            fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
            fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
            fig.suptitle('MP ' + mp_string + '   AN ' + an_string + '     ::     Comp Variability Page ' +
                         str(i_figure + 1) + ' of ' + str(n_figures),
                         color='darkblue', fontsize=20)
            fig.canvas.set_window_title(
                'Comp Variability Plots: ' + 'MP ' + mp_string + '   AN ' + an_string)
            subplot_text = str(n_comps) + ' comps    ' + \
                'sigma=' + '{0:.0f}'.format(1000.0 * sigma) + ' mMag' + (12 * ' ') + \
                ' rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
            fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=12,
                     color='dimgray')
            for i_plot in range(n_plots_this_figure):
                first_index_in_plot = i_figure * n_plots_per_figure + i_plot
                i_col = i_plot % n_cols
                i_row = int(floor(i_plot / n_cols))
                ax = axes[i_row, i_col]
                # Make a frame and axes only (x and y vectors are None):
                make_9_subplot(ax, 'Comp Variability plot', xlabel_jd, 'mMag', '', True, None, None)
                # Make a scatter plot for each chosen comp:
                scatterplots = []
                legend_labels = []
                for i_plot_comp in range(n_comps_per_plot):
                    i_plot_index = first_index_in_plot + i_plot_comp * n_plots
                    if i_plot_index <= n_comps - 1:
                        is_this_plot = (df_comp_offsets['PlotIndex'] == i_plot_index)
                        x = df_comp_offsets.loc[is_this_plot, 'JD_fract']
                        y = 1000.0 * df_comp_offsets.loc[is_this_plot, 'NormalizedOffset']
                        ax.plot(x, y, linewidth=2, alpha=0.8, color=plot_colors[i_plot_comp])
                        sc = ax.scatter(x=x, y=y, s=24, alpha=0.8, color=plot_colors[i_plot_comp])
                        scatterplots.append(sc)
                        this_comp_id = (df_comp_offsets.loc[is_this_plot, 'SourceID']).iloc[0]
                        # print('i_figure ' + str(i_figure) +\
                        #       '  i_plot ' + str(i_plot) +\
                        #       '  i_plot_comp ' + str(i_plot_comp) +\
                        #       '  i_plot_index ' + str(i_plot_index) +\
                        #       '  this_comp_id ' + this_comp_id)  # (debug)
                        legend_labels.append(this_comp_id)
                        plotted_comp_ids.append(this_comp_id)
                ax.set_xlim(jd_low_limit, jd_high_limit)
                ax.set_ylim(offset_low_limit, offset_high_limit)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
                ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))
                ax.legend(scatterplots, legend_labels, loc='upper right')
            # Remove any empty subplots from this (last) Figure:
            for i_plot in range(n_plots_this_figure, n_plots_per_figure):
                i_col = i_plot % n_cols
                i_row = int(floor(i_plot / n_cols))
                ax = axes[i_row, i_col]
                ax.remove()
        plt.show()
        plt.savefig(image_prefix + '5_Comp Variability_' + '{:02d}'.format(i_figure + 1) + '.png')

    # Verify that all comps were plotted exactly once (debug):
    all_comps_plotted_once = (sorted(comp_ids) == sorted(plotted_comp_ids))
    # print('all comp ids were plotted exactly once = ', str(all_comps_plotted_once))
    if not all_comps_plotted_once:
        print('comp ids plotted more than once',
              [item for item, count in Counter(plotted_comp_ids).items() if count > 1])


def draw_x_line(ax, x_value, color='lightgray'):
    ax.axvline(x=x_value, color=color, linewidth=1, zorder=-100)


_____SUPPORT________________________________________________ = 0



def get_fits_filenames(directory):
    all_filenames = pd.Series([e.name for e in os.scandir(directory) if e.is_file()])
    extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in all_filenames])
    is_fits = [ext.lower() in VALID_FITS_FILE_EXTENSIONS for ext in extensions]
    fits_filenames = all_filenames[is_fits]
    return fits_filenames


def get_session_state(site_name='DSW', instrument_name='Borea'):
    """ Return dict containing extinction & transforms (etc?).
    :param site_name: name of site for class Site, e.g., 'DSW' [string].
    :param instrument_name: name of instrument for class Instrument, e.g., 'Borea' [string].
    :return: Session state data (dict of dicts)
    Access an extinction via: state['extinction'][filter_string].
    Access a filter=passband transform (color index V-I) via: state['transform'][filter_string].
    """
    # TODO: we probably don't need (and should remove) transforms, as we're not using Johnson-Cousins.
    # TODO: *OR* we should really put all the transforms into photrix Instrument file(s).
    from photrix.user import Site, Instrument
    state = dict()
    site = Site(site_name)
    state['extinction'] = site.extinction
    inst = Instrument(instrument_name)
    transform_dict = dict()
    for this_filter in ['V', 'R', 'I']:
        transform_dict[this_filter] = inst.transform(this_filter, 'V-I')
    state['transform'] = transform_dict
    return state


def reorder_df_columns(df, left_column_list=None, right_column_list=None):
    left_column_list = [] if left_column_list is None else left_column_list
    right_column_list = [] if right_column_list is None else right_column_list
    new_column_order = left_column_list +\
                       [col_name for col_name in df.columns
                        if col_name not in (left_column_list + right_column_list)] +\
                       right_column_list
    df = df[new_column_order]
    return df


def adu_sat_from_xy(x1024, y1024):
    """ Return estimated saturation ADU limit from aperture's distances from image center.
    :param x1024: pixels/1024 in x-direction of aperture center from image center. [float]
    :param y1024: pixels/1024 in y-direction of aperture center from image center. [float]
    :return: estimated saturation ADU at given image position. [float]
    """
    r2 = x1024 ** 2 + y1024 ** 2
    fract_dist2_to_vign_pt = r2 / ((VIGNETTING[0] / 1024.0) ** 2)
    fract_decr = (1.0 - VIGNETTING[1]) * fract_dist2_to_vign_pt
    return ADU_SATURATED * (1.0 - fract_decr)


def screen_comps_for_photometry(refcat2):
    """ Applies ATLAS refcat2 screens to refcat2 object itself.
    :param refcat2: ATLAS refcat2 catalog from catalog.py. [catalog.py:Refcat2 object]
    :return lines: list of advisory test lines. [list of strings]
    """
    lines = []
    refcat2.select_min_r_mag(MIN_R_MAG)
    lines.append('Refcat2: min(g) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_max_r_mag(MAX_R_MAG)
    lines.append('Refcat2: max(g) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_max_g_uncert(MAX_G_UNCERT)
    lines.append('Refcat2: max(dg) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_max_r_uncert(MAX_R_UNCERT)
    lines.append('Refcat2: max(dr) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_max_i_uncert(MAX_I_UNCERT)
    lines.append('Refcat2: max(di) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_sloan_ri_color(MIN_SLOAN_RI_COLOR, MAX_SLOAN_RI_COLOR)
    lines.append('Refcat2: Sloan ri color screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    # refcat2.select_dgaia()
    # lines.append('Refcat2: dgaia screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.remove_overlapping()
    lines.append('Refcat2: overlaps removed to ' + str(len(refcat2.df_selected)) + ' stars.')
    return lines


def apply_do_phot_selections(df_model, user_selections):
    """ Applies user selections to set desired df_model's 'UseInModel' row(s) to False, in-place.
    :param df_model: observations dataframe. [pandas DataFrame]
    :param user_selections: dict of lists of items to remove from df_obs before modeling. [dict of lists]
    :return df_obs: [None] (both dataframes are modified in-place.)
    """
    # Apply user selections to observations:
    is_comp = (df_model['Type'] == 'Comp')
    deselect_for_serial = df_model['Serial'].isin(user_selections['serials']) & is_comp
    deselect_for_comp_id = df_model['SourceID'].isin(user_selections['comps']) & is_comp
    images_to_omit = [i for i in user_selections['images']] +\
                     [i + '.fts' for i in user_selections['images']]  # allow entry w/ or w/ '.fts' at end.
    deselect_for_image = df_model['FITSfile'].isin(images_to_omit)  # remove MPs as well, here.

    deselect_for_low_r_mag = (df_model['r'] < user_selections['min_r_mag']) & is_comp
    deselect_for_high_r_mag = (df_model['r'] > user_selections['max_r_mag']) & is_comp
    deselect_for_high_catalog_dr_mmag = (df_model['dr'] > user_selections['max_catalog_dr_mmag']) & is_comp
    sloan_ri_color = df_model['r'] - df_model['i']
    deselect_for_low_sloan_ri_color = (sloan_ri_color < user_selections['min_sloan_ri_color']) & is_comp
    deselect_for_high_sloan_ri_color = (sloan_ri_color > user_selections['max_sloan_ri_color']) & is_comp
    obs_to_deselect = list(deselect_for_serial | deselect_for_comp_id | deselect_for_image
                           | deselect_for_low_r_mag | deselect_for_high_r_mag
                           | deselect_for_high_catalog_dr_mmag
                           | deselect_for_low_sloan_ri_color | deselect_for_high_sloan_ri_color)
    df_model['UseInModel'] = True
    df_model.loc[obs_to_deselect, ['UseInModel']] = False


    # def az_alt_at_datetime_utc(longitude, latitude, target_radec, datetime_utc):
    #     obs = ephem.Observer()  # for local use.
    #     if isinstance(longitude, str):
    #         obs.lon = longitude
    #     else:
    #         # next line wrong?: if string should be in deg not radians?? (masked by long passed as hex string?)
    #         obs.lon = str(longitude * pi / 180)
    #     if isinstance(latitude, str):
    #         obs.lat = latitude
    #     else:
    #         # next line wrong?: if string should be in deg not radians?? (masked by long passed as hex string?)
    #         obs.lat = str(latitude * pi / 180)
    #     obs.date = datetime_utc
    #     target_ephem = ephem.FixedBody()  # so named to suggest restricting its use to ephem.
    #     target_ephem._epoch = '2000'
    #     target_ephem._ra, target_ephem._dec = target_radec.as_hex  # text: RA in hours, Dec in deg
    #     target_ephem.compute(obs)
    #     return target_ephem.az * 180 / pi, target_ephem.alt * 180 / pi


from functools import lru_cache
@lru_cache(maxsize=800)
def make_skycoord(ra_deg, dec_deg):
    return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)


_____TEST_AND_PROTOTYPE_temporary______________________________________ = 0


# def t():
#     """ Try astroplan package. """
#     from astropy.coordinates import SkyCoord
#     from astropy.time import Time
#     from astropy import units as u
#     from astroplan import Observer, FixedTarget
#
#     dsw = Observer(longitude=DSW_SITE_DATA['longitude'] * u.deg,
#                    latitude=DSW_SITE_DATA['latitude'] * u.deg,
#                    elevation=2210 * u.m, name='DSW')
#
#     jd = 2459018.676703 + 67 / 24 / 3600  # end JD!
#     utc = datetime_utc_from_jd(jd)
#     time = Time(jd, format='jd')
#
#     coord = SkyCoord(ra=267.529167 * u.deg, dec=-6.953611 * u.deg)
#     target = FixedTarget(name='6077', coord=coord)
#
#     altaz = dsw.altaz(time, coord)
#     airmass = 1.0 / sin(altaz.alt.value / DEGREES_PER_RADIAN)
#     iiii = 4


# def sync_comps_and_images(df_all_columns, df_comps, df_images):
#     """ Utility to return df_comps and df_images
#             whose rows are selected by representation in df_all_columns.
#     :param df_all_columns: the master dataframe, immutable [pandas DataFrame]
#     :param df_comps: the comps dataframe to select from. [pandas DataFrame]
#     :param df_images: the images dataframe to select from. [pandas DataFrame]
#     :return: df_comps_synced, df_images_synced
#     """
#     comp_list = df_all_columns['CompID'].drop_duplicates().copy()
#     is_in_comp_list = list(df_comps['CompID'].isin(comp_list))
#     df_comps_synced = df_comps.loc[is_in_comp_list, :].copy()
#     image_list = df_all_columns['FITSfile'].drop_duplicates().copy()
#     is_in_image_list = list(df_images['FITSfile'].isin(image_list))
#     df_images_synced = df_images.loc[is_in_image_list, :].copy()
#     return df_comps_synced, df_images_synced


_____ANCILLARY_CODE________________________________________________ = 0


def remove_pinpoint_plate_solution(fullpath):
    """ Remove PinPoint's plate solution entirely from one FITS file."""
    from astropy.io import fits
    try:
        hdulist = fits.open(fullpath)
    except IOError:
        print(' >>>>> ERROR: cannot read FITS file', fullpath)
        return
    hdr = hdulist[0].header

    keys_to_remove = ('PA', 'CRVAL*', 'CRPIX*', 'CDELT*', 'CROTA*', 'CTYPE*', 'CD1_*', 'CD2_*',
                      'TR1_*', 'TR2_*', 'PLTSOLVD')
    for key_to_remove in keys_to_remove:
        for key in hdr.keys():
            if key_to_remove.endswith('*'):
                if key.startswith(key_to_remove[:-1]):
                    del hdr[key]
            else:
                if key == key_to_remove:
                    del hdr[key]
    hdr['HISTORY'] = 'Pinpoint plate solution removed.'
    fn, ext = tuple(fullpath.rsplit('.', maxsplit=1))
    outpath = '.'.join([fn + 'noPP', ext])
    hdulist.writeto(outpath)


# def mp_phot_test():
#     from random import random, gauss, uniform, seed
#     _, _, _, _ = get_context()   # forces exit if working directory is invalid.
#     state = get_session_state()  # for extinction and transform values.
#     seed(3423)
#     n_comps = 40
#     df_comps = read_df_comps_all()[0:n_comps]
#     df_comps['Vignette'] = [uniform(-1, 1) for i in range(n_comps)]
#     df_comps['X1024'] = [uniform(-1, 1) for i in range(n_comps)]
#     df_comps['Y1024'] = [uniform(-1, 1) for i in range(n_comps)]
#     df_comps['CatError'] = [0.2, -0.15] + [gauss(0, 0.01) for i in range(n_comps - 2)]
#     n_images = 20
#     df_images = pd.DataFrame(
#         {'Serial': [1 + i for i in range(n_images)],
#          'FITSfile': ['a' + str(i+1) + '.fts' for i in range(n_images)],
#          'Airmass': [1.2 + 0.01 * i for i in range(n_images)],
#          'JD_fract': [.6 + 0.005 * i for i in range(n_images)],
#          'Cirrus': [gauss(0, 0.05) for i in range(n_images)]})
#     z = -21.000
#     extinction = state['extinction']['Clear']
#     transform = TRANSFORM_CLEAR_SR_SR_SI  # overrides value read from state.
#     dict_obs_list = []
#     for i_comp in df_comps.index:
#         for i_image in df_images.index:
#             dict_obs = {'FITSfile': df_images.loc[i_image, 'FITSfile'],
#                         'SourceID': df_comps.loc[i_comp, 'CompID'],
#                         'Type': 'Comp',
#                         'Airmass': df_images.loc[i_image, 'Airmass'],
#                         'JD_fract': df_images.loc[i_image, 'JD_fract'],
#                         'Vignette': df_comps.loc[i_comp, 'Vignette'],
#                         'X1024': df_comps.loc[i_comp, 'X1024'],
#                         'Y1024': df_comps.loc[i_comp, 'Y1024'],
#                         'CI': df_comps.loc[i_comp, 'r'] - df_comps.loc[i_comp, 'i'],
#                         'R': df_comps.loc[i_comp, 'R'],
#                         'InstMag': z + df_comps.loc[i_comp, 'R']
#                                 + extinction * df_images.loc[i_image, 'Airmass']
#                                 + transform * (df_comps.loc[i_comp, 'r'] - df_comps.loc[i_comp, 'i'])
#                                 + 0.04 * df_comps.loc[i_comp, 'Vignette']
#                                 + df_images.loc[i_image, 'Cirrus']
#                                 - df_comps['CatError']
#                                 + gauss(0, 0.002)}
#             dict_obs_list.append(dict_obs)
#     df_obs = pd.DataFrame(data=dict_obs_list)
#     df_comps = df_comps.drop(['Vignette', 'X1024', 'Y1024'], axis=1)
#
#     smodel = SessionModel(df_obs, df_comps, state)
