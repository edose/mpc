# Python core packages:
import os
from math import cos, pi, log10, log, floor, ceil, sqrt
from collections import Counter
from datetime import datetime, timezone
# from statistics import pstdev, mean

# External packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
# import statsmodels.api as sm
from scipy.stats import norm

# From this (mpc) package:
from mpc.catalogs import Refcat2, get_bounding_ra_dec
from mpc.mp_planning import all_mpfile_names, MPfile

# From external (EVD) package photrix:
from photrix.image import Image, FITS
from photrix.util import RaDec, jd_from_datetime_utc, datetime_utc_from_jd, \
    MixedModelFit, ra_as_hours, dec_as_hex

__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

DAYS_PER_YEAR_NOMINAL = 365.25
VALID_FITS_FILE_EXTENSIONS = ['.fits', '.fit', '.fts']
DEGREES_PER_RADIAN = 180.0 / pi

# To assess FITS files in assess():
MP_PHOTOMETRY_FILTER = 'Clear'
MIN_MP_PHOTOMETRY_FILES = 5
MIN_FWHM = 1.5  # in pixels.
MAX_FWHM = 14  # "
FOCUS_LENGTH_MAX_PCT_DEVIATION = 3.0

MARGIN_RA_ZERO = 5  # in degrees, to handle RA ~zero; s/be well larger than images' height or width.

# For color handling:
FILTERS_FOR_MP_COLOR_INDEX = ('R', 'I')
DEFAULT_MP_RI_COLOR = 0.2  # close to known Sloan mean (r-i) for MPs.

# To screen observations:
MAX_MAG_UNCERT = 0.03  # min signal/noise for COMP obs (as InstMagSigma).
# The next two probably should go in Instrument class, but ok for now.
ADU_SATURATED = 56000  # Max ADU allowable in original (Ur) images.
VIGNETTING = (1846, 0.62)  # (px from center, max fract of ADU_SATURATED allowed) both at corner.

# For this package:
MP_TOP_DIRECTORY = 'C:/Astro/MP Photometry/'
LOG_FILENAME = 'mp_photometry.log'
CONTROL_FILENAME = 'control.txt'
DF_OBS_ALL_FILENAME = 'df_obs_all.csv'
DF_IMAGES_ALL_FILENAME = 'df_images_all.csv'
DF_COMPS_ALL_FILENAME = 'df_comps_all.csv'
TRANSFORM_CLEAR_SR_SR_SI = 0.025  # estimate from MP 1074 20191109 (37 images).
DEFAULT_MODEL_OPTIONS = {'fit_transform': False, 'fit_extinction': False,
                         'fit_vignette': True, 'fit_xy': False,
                         'fit_jd': True}  # defaults if not found in file control.txt.

# For selecting comps within Refcat2 object (intentionally wide; can narrow with control.txt later):
MIN_R_MAG = 10
MAX_R_MAG = 16
MAX_G_UNCERT = 20  # millimagnitudes
MAX_R_UNCERT = 20  # "
MAX_I_UNCERT = 20  # "
MIN_SLOAN_RI_COLOR = -0.4
MAX_SLOAN_RI_COLOR = 0.8

# Defaults (user-changeable) for selecting comps and obs in do_mp_phot():
PHOT_MIN_R_MAG = MIN_R_MAG
PHOT_MAX_R_MAG = MAX_R_MAG
PHOT_MAX_CATALOG_DR_MMAG = 15
PHOT_MIN_SLOAN_RI_COLOR = 0.0
PHOT_MAX_SLOAN_RI_COLOR = 0.4

# For ALCDEF File generation:
DSW_SITE_DATA = {'longitude': -105.6536, 'latitude': +35.3311,
                 'facility': 'Deep Sky West Observatory',
                 'mpccode': 'V28'}
ALCDEF_DATA = {'contactname': 'Eric V. Dose',
               'contactinfo': 'MP@ericdose.com',
               'observers': 'Dose, E.V.',
               'filter': 'C',
               'magband': 'SR'}


_____ATLAS_BASED_WORKFLOW________________________________________________ = 0

"""  ***************************************************************************
     WORKFLOW STEPS (example lines):
     * Ensure at least 5 MP photometry files IN Clear filter.
     >>> start(MP_TOP_DIRECTORY + '/Test', 1074, 20191109)
     >>> assess()
     * Edit control.txt: add 2 MP positions (x_pixel, y_pixel), at early and late times.
     >>> make_dfs()
     * Edit control.txt: (1) comp-selection limits, (2) model options [both optional].
     >>> do_mp_phot()
     ********************************************
     (at any time:)
     >>> resume(MP_TOP_DIRECTORY + '/Test', 1074, 20191109), esp. after Console restart.
     ***************************************************************************
"""


def start(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Preliminaries to begin MP photometry workflow.
    :param mp_top_directory: path of lowest directory common to all MP photometry FITS, e.g.,
               'C:/Astro/MP Photometry' [string]
    :param mp_number: number of target MP, e.g., 1602 for Indiana. [integer or string].
    :param an_string: Astronight string representation, e.g., '20191106' [string].
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
    log_file = open(LOG_FILENAME, mode='w')  # new file; wipe old one out if it exists.
    log_file.write(mp_directory + '\n')
    log_file.write('MP: ' + mp_string + '\n')
    log_file.write('AN: ' + an_string + '\n')
    log_file.write('This log started: ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    log_file.close()
    print('Log file started.')
    print('Next: assess()')


def resume(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Restart a workflow in its correct working directory,
         but keep the previous log file--DO NOT overwrite it.
    parameters as for start().
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


def assess(min_mp_photometry_files=MIN_MP_PHOTOMETRY_FILES):
    """  First, verify that all required files are in the working directory or otherwise accessible.
         Then, perform checks on FITS files in this directory before performing the photometry proper.
         Modeled after and extended from assess() found in variable-star photometry package 'photrix'.
    :param min_mp_photometry_files: minimum required number of mp photometry files.
                                    May be zero for MP color index determination only. [int]
    :return: [None]
    """
    this_directory, mp_string, an_string = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    log_file.write('\n===== access()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    n_warnings = 0

    # Get FITS file names in current directory:
    fits_filenames = get_fits_filenames(this_directory)
    print(str(len(fits_filenames)) + ' FITS files found:')
    log_file.write(str(len(fits_filenames)) + ' FITS files found:' + '\n')

    # Verify that all required FITS file types exist within this directory and are valid:
    filter_counter = Counter()
    filters_to_use = [MP_PHOTOMETRY_FILTER] + list(FILTERS_FOR_MP_COLOR_INDEX)
    filenames_to_use = []
    for filename in fits_filenames:
        fits = FITS(this_directory, '', filename)
        if fits.is_valid:
            filter_counter[fits.filter] += 1
            if fits.filter in filters_to_use:
                filenames_to_use.append(filename)

    if filter_counter[MP_PHOTOMETRY_FILTER] < min_mp_photometry_files:
        print(' >>>>> ERROR:', str(min_mp_photometry_files),
              'valid files in main photometry filter (' + MP_PHOTOMETRY_FILTER + ') are required, but only'
              + str(filter_counter[MP_PHOTOMETRY_FILTER]), 'found.')
        exit(1)
    for filter in filter_counter.keys():
        if filter == MP_PHOTOMETRY_FILTER:
            filter_category = 'MAIN PHOTOMETRIC filter'
        elif filter in FILTERS_FOR_MP_COLOR_INDEX:
            filter_category = 'for MP COLOR INDEX'
        else:
            filter_category = '(ignored)'
        print('   ' + str(filter_counter[filter]), 'in filter', filter + '. ', filter_category)

    # Start dataframe for main FITS integrity checks:
    fits_extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in filenames_to_use])
    df = pd.DataFrame({'Filename': filenames_to_use,
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
        print('Next: (1) enter MP pixel positions in control.txt AND SAVE it,\n      (2) make_dfs()')
        log_file.write('assess(): ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.' + '\n')
    else:
        print('\n >>>>> ' + str(n_warnings) + ' warnings (see listing above).')
        print('        Correct these and rerun assess() until no warnings remain.')
        log_file.write('assess(): ' + str(n_warnings) + ' warnings.' + '\n')

    # Write control.txt stub *IF* it doesn't already exist:
    df['SecondsRelative'] = [(utc - df['UTC_mid'].min()).total_seconds() for utc in df['UTC_mid']]
    i_earliest = df['SecondsRelative'].nsmallest(n=1).index[0]
    i_latest = df['SecondsRelative'].nlargest(n=1).index[0]
    earliest_filename = df.loc[i_earliest, 'Filename']
    latest_filename = df.loc[i_latest, 'Filename']
    lines = [';----- This is control.txt for directory:\n      ' + this_directory,
             ';',
             ';===== Enter before make_dfs() ===================================',
             ';      MP x,y positions for aperture photometry:',
             '#MP  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
                                           'early filename, change if needed',
             '#MP  ' + latest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
                                         ' late filename, change if needed',
             ';',
             ';===== Enter before do_mp_phot() ====================================',
             ';#MP_RI_COLOR UseDefault      ; or give a *measured* r-i (ATLAS refcat2) color',
             ';',
             ';===== Enter before do_mp_phot() ====================================',
             ';      Selection criteria for comp stars:',
             ';#COMP  2245 144,   781       ; to omit comp by comp ID (many per line OK)',
             ';#SERIAL 123,7776 2254   16   ; to omit observations by Serial number (many per line OK)',
             ';#IMAGE  MP_mmmm-00nn-Clear   ; to omit FITS image (.fts at end optional)',
             (';#MIN_R_MAG ' + str(PHOT_MIN_R_MAG)).ljust(30) + '; default=' + str(PHOT_MIN_R_MAG),
             (';#MAX_R_MAG ' + str(PHOT_MAX_R_MAG)).ljust(30) + '; default=' + str(PHOT_MAX_R_MAG),
             (';#MAX_CATALOG_DR_MMAG ' + str(PHOT_MAX_CATALOG_DR_MMAG)).ljust(30) +
             '; default=' + str(PHOT_MAX_CATALOG_DR_MMAG),
             (';#MIN_SLOAN_RI_COLOR ' + str(PHOT_MIN_SLOAN_RI_COLOR)).ljust(30) +
             '; default=' + str(PHOT_MIN_SLOAN_RI_COLOR),
             (';#MAX_SLOAN_RI_COLOR ' + str(PHOT_MAX_SLOAN_RI_COLOR)).ljust(30) +
             '; default=' + str(PHOT_MAX_SLOAN_RI_COLOR),
             ';',
             ';===== Enter before do_mp_phot(): ===================================',
             ';----- OPTIONS for regression model:',
             (';#FIT_TRANSFORM ' + str(DEFAULT_MODEL_OPTIONS['fit_transform'])).ljust(30) + '; default='
             + str(DEFAULT_MODEL_OPTIONS['fit_transform']) + ' or yes, False, No  (case-insensitive)',
             (';#FIT_EXTINCTION ' + str(DEFAULT_MODEL_OPTIONS['fit_extinction'])).ljust(30) + '; default='
             + str(DEFAULT_MODEL_OPTIONS['fit_extinction']) + ' or yes, False, No  (case-insensitive)',
             (';#FIT_VIGNETTE ' + str(DEFAULT_MODEL_OPTIONS['fit_vignette'])).ljust(30) + '; default='
             + str(DEFAULT_MODEL_OPTIONS['fit_vignette']) + ' or yes, False, No  (case-insensitive)',
             (';#FIT_XY ' + str(DEFAULT_MODEL_OPTIONS['fit_xy'])).ljust(30) + '; default='
             + str(DEFAULT_MODEL_OPTIONS['fit_xy']) + ' or yes, False, No  (case-insensitive)',
             (';#FIT_JD ' + str(DEFAULT_MODEL_OPTIONS['fit_jd'])).ljust(30) + '; default='
             + str(DEFAULT_MODEL_OPTIONS['fit_jd']) + ' or yes, False, No  (case-insensitive)',
             ';'
             ]
    lines = [line + '\n' for line in lines]
    fullpath = os.path.join(this_directory, 'control.txt')
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(lines)
            log_file.write('New control.txt file written.\n')

    log_file.close()


def make_dfs():
    """ For one MP on one night: gather images and ATLAS refcat2 catalog data, make df_comps and df_obs.
    :return: [None]
    USAGE: make_dfs()
    """
    this_directory, mp_string, an_string = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n===== make_dfs()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Get all relevant FITS filenames, make lists of FITS and Image objects (per photrix):
    fits_names = get_fits_filenames(this_directory)
    fits_list = [FITS(this_directory, '', fits_name) for fits_name in fits_names]  # FITS objects
    filters_to_use = [MP_PHOTOMETRY_FILTER] + list(FILTERS_FOR_MP_COLOR_INDEX)
    fits_list = [fits for fits in fits_list if fits.filter in filters_to_use]
    image_list = [Image(fits_object) for fits_object in fits_list]  # Image objects

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
    # TODO: may need an option (control.txt?) for less restrictive screens, for get_transforms().
    lines = screen_comps_for_photometry(refcat2)  # works IN-PLACE.
    refcat2.update_epoch(mid_session_utc)    # apply proper motions to update star positions.
    print('\n'.join(lines), '\n')
    log_file.write('\n'.join(lines) + '\n')

    # Add apertures for all comps within all Image objects:
    df_radec = refcat2.selected_columns(['RA_deg', 'Dec_deg'])
    for image in image_list:
        if image.fits.filter in [MP_PHOTOMETRY_FILTER] + list(FILTERS_FOR_MP_COLOR_INDEX):
            for i_comp in df_radec.index:
                ra = df_radec.loc[i_comp, 'RA_deg']
                dec = df_radec.loc[i_comp, 'Dec_deg']
                x0, y0 = image.fits.xy_from_radec(RaDec(ra, dec))
                image.add_aperture(str(i_comp), x0, y0)
            print(image.fits.filename + ':', str(len(image.apertures)), 'comp apertures.')

    # Read user's MP *pixel* positions in 2 images, from control.txt:
    mp_position_filenames, x_pixels, y_pixels = read_mp_positions()
    if mp_position_filenames is None:
        print(' >>>>> ' + CONTROL_FILENAME + ': something wrong with #MP lines. Stopping.')
        log_file.write(' >>>>> ' + CONTROL_FILENAME + ': something wrong with #MP lines. Stopping.\n')
        exit(1)

    # Extract MP *RA,Dec* positions from the 2 images:
    # FITS objects in next line are temporary only.
    mp_position_fits = [FITS(this_directory, '', mp_filename) for mp_filename in mp_position_filenames]
    mp_datetime, mp_ra_deg, mp_dec_deg = [], [], []
    for i in range(2):
        ps = mp_position_fits[i].plate_solution
        dx = x_pixels[i] - ps['CRPIX1']
        dy = y_pixels[i] - ps['CRPIX2']
        d_east_west = dx * ps['CD1_1'] + dy * ps['CD1_2']
        d_ra = d_east_west / cos(mp_position_fits[0].dec / DEGREES_PER_RADIAN)
        d_dec = dx * ps['CD2_1'] + dy * ps['CD2_2']
        mp_datetime.append(mp_position_fits[i].utc_mid)
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

    # Write df_obs to CSV file (rather than returning the df):
    fullpath_df_obs_all = os.path.join(this_directory, DF_OBS_ALL_FILENAME)
    df_obs.to_csv(fullpath_df_obs_all, sep=';', quotechar='"',
                  quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    n_comp_obs = sum([t.lower() == 'comp' for t in df_obs['Type']])
    n_mp_obs = sum([t.lower() == 'mp' for t in df_obs['Type']])
    print('obs written to', fullpath_df_obs_all)
    log_file.write(DF_OBS_ALL_FILENAME + ' written: ' + str(n_comp_obs) + ' comp obs & ' +
                   str(n_mp_obs) + ' MP obs.\n')

    # Write df_comps to CSV file (rather than returning the df):
    fullpath_df_comps_all = os.path.join(this_directory, DF_COMPS_ALL_FILENAME)
    df_comps.to_csv(fullpath_df_comps_all, sep=';', quotechar='"',
                    quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('comps written to', fullpath_df_comps_all)
    log_file.write(DF_COMPS_ALL_FILENAME + ' written: ' + str(len(df_comps)) + ' comps.\n')

    # Write df_images to CSV file (rather than returning the df):
    fullpath_df_images_all = os.path.join(this_directory, DF_IMAGES_ALL_FILENAME)
    df_images.to_csv(fullpath_df_images_all, sep=';', quotechar='"',
                     quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('images written to', fullpath_df_images_all)
    log_file.write(DF_IMAGES_ALL_FILENAME + ' written: ' + str(len(df_images)) + ' images.\n')

    log_file.close()
    print('\nNext: (1) enter comp selection limits and model options in control.txt,'
          '\n      (2) run do_mp_phot()')


def do_mp_phot():
    """ Use obs quality and user's control.txt to screen obs for photometry and diagnostic plotting.
    USAGE: do_mp_phot()   [no return value]
    """
    this_directory, mp_string, an_string = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n===== do_mp_phot()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Load required data:
    df_obs_all = read_df_obs_all()
    df_comps_all = read_df_comps_all()
    df_images_all = read_df_images_all()
    state = get_session_state()  # for extinction and transform values.

    # Get MP color index, if images present for filter pair (R, I).
    return_values = get_mp_color_index(df_obs_all, df_images_all, df_comps_all)  # returns a tuple.
    mp_color_ri, mp_color_ri_sigma, mp_color_n_comps, source_string_ri = return_values  # unpack the tuple.
    if mp_color_ri_sigma is not None:
        print('MP color index (r-i) =', '{0:.3f}'.format(mp_color_ri),
              u'\u00B1' + ' {0:.3f}'.format(mp_color_ri_sigma), 'from', source_string_ri,
              '(' + str(mp_color_n_comps) + ' comps used)')
        log_file.write('MP color index (r-i) = ' + '{0:.3f}'.format(mp_color_ri) +
                       ' ' + u'\u00B1' + ' {0:.3f}'.format(mp_color_ri_sigma) +
                       ' from ' + source_string_ri +
                       ' (' + str(mp_color_n_comps) + ' comps used)')
    else:
        print('MP color index (r-i) =', '{0:.3f}'.format(mp_color_ri), 'from', source_string_ri)
        log_file.write('MP color index (r-i) = ' + '{0:.3f}'.format(mp_color_ri) +
                       ' from ' + source_string_ri)

    # Make df_full: all data (obs, comp, image) only from images that...
    #     (1) are taken in main photometric filter AND (2) have a valid MP obs:
    is_in_photometry_filter = (df_images_all['Filter'] == MP_PHOTOMETRY_FILTER)
    photometry_image_list = list(df_images_all.loc[is_in_photometry_filter, 'FITSfile'])
    is_mp_obs = list(df_obs_all['Type'] == 'MP')
    mp_image_list = list(df_obs_all.loc[is_mp_obs, 'FITSfile'])
    qualified_image_list = [f for f in photometry_image_list if (f in mp_image_list)]  # list intersection.
    has_qualified_image = [(f in qualified_image_list) for f in df_obs_all['FITSfile']]
    df_qualified_image = df_obs_all.loc[has_qualified_image, :].copy()
    df_obs_and_comps = pd.merge(left=df_qualified_image, right=df_comps_all,
                                how='left', left_on='SourceID', right_on='CompID', sort=False)
    df_full = pd.merge(left=df_obs_and_comps, right=df_images_all, how='left', on='FITSfile', sort=False)

    # Make df_model: rows only for (1) MP obs and (2) obs with comps present in every selected image:
    is_comp = (df_full['Type'] == 'Comp')
    df_image_count = df_full.loc[is_comp, :].groupby('SourceID')[['FITSfile', 'SourceID']].count()
    is_in_every_image = (df_image_count['FITSfile'] == len(qualified_image_list))
    comp_ids_in_every_image = list(df_image_count.index[is_in_every_image])
    rows_with_qualified_comp_ids = df_full['SourceID'].isin(comp_ids_in_every_image)
    rows_with_mp_ids = (df_full['Type'] == 'MP')
    rows_to_keep = rows_with_qualified_comp_ids | rows_with_mp_ids
    df_model = df_full.loc[rows_to_keep, :].copy()
    df_model['UseInModel'] = True
    # Note: df_model now contains all obs, comp, and image data, for both comp stars and minor planets.

    # Mark df_model with user selections, sync comp and image dfs:
    user_selections = read_user_selections()
    apply_user_selections(df_model, user_selections)  # modifies in-place.

    # # Sync the comp and image dataframes (esp. user selections); may be needed later for plotting:
    # df_model_comps, df_model_images = sync_comps_and_images(df_model, df_comps_all, df_images_all)
    # print(str(len(df_model_comps)), 'comps retained in model.')

    # # Make data diagnostics for plotting:
    # df_model = make_diagnostics(df_model)
    #
    # # Do data-only (pre-model) plots; use them to update (in-place) 'UseInModel' column in the 3 dfs:
    # do_pre_model_plots(df_model, df_model_comps, df_model_images, user_selections, mp_color_ri)

    # Make photometric model via mixed-model regression, using only selected observations:
    options_dict = read_model_options()
    model = SessionModel(df_model, mp_color_ri, state, options_dict)

    do_plots(model, df_model, mp_color_ri, state, user_selections)

    write_canopus_file(model)
    write_alcdef_file(model, mp_color_ri, source_string_ri)

    # Write last info lines:
    if mp_color_ri_sigma is not None:
        print('MP color index (r-i) =', '{0:.3f}'.format(mp_color_ri),
              u'\u00B1' + ' {0:.3f}'.format(mp_color_ri_sigma), 'from', source_string_ri,
              '(' + str(mp_color_n_comps) + ' comps used)')
    else:
        print('MP color index (r-i) =', '{0:.3f}'.format(mp_color_ri), 'from', source_string_ri)
    model_jds = df_model['JD_mid']
    print('Add this line to MPfile', mp_string + ':',
          '    #OBS ', '{0:.5f}'.format(model_jds.min()), '  {0:.5f}'.format(model_jds.max()),
          '  ; ', an_string)


def do_plots(model, df_model, mp_color_ri, state, user_selections):
    """  Produce diagnostic plots, to help decide which obs, comps, images might need removal by
         editing control.txt.
    :param model: mixed model summary object. [photrix.MixedModelFit object]
    :param df_model: dataframe of all data including UseInModel (user selection) column. [pandas DataFrame]
    :param mp_color_ri: Sloan r-i color of minor planet target. [float]
    :param state: session state for this observing session [dict]
    :param user_selections: comp selection criteria, used for drawing limits on plots [python dict]
    :return: [None]
    """
    def make_labels_9_subplots(ax, title, xlabel, ylabel, text='', zero_line=True):
        ax.set_title(title, loc='center', pad=-3)  # pad in points
        ax.set_xlabel(xlabel, labelpad=-29)  # labelpad in points
        ax.set_ylabel(ylabel, labelpad=-5)   # "
        ax.text(x=0.5, y=0.95, s=text,
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if zero_line is True:
            ax.axhline(y=0, color='lightgray', linewidth=1, zorder=-100)

    def draw_y_line(ax, y_value, color='lightgray'):
        ax.axhline(y=y_value, color=color, linewidth=1, zorder=-100)

    def draw_x_line(ax, x_value, color='lightgray'):
        ax.axvline(x=x_value, color=color, linewidth=1, zorder=-100)

    def make_qq_plot_fullpage(window_title, page_title, plot_annotation,
                              y_values, y_labels, filename, figsize=(11, 8.5)):  # was 12, 9
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)  # (width, height) in "inches"
        ax = axes  # not subscripted if just one subplot in Figure
        ax.set_title(page_title, color='darkblue', fontsize=20, pad=30)
        ax.set_xlabel('t (sigma.residuals = ' + str(round(pd.Series(y_values).std(), 1)) + ' mMag)')
        ax.set_ylabel('Residual (mMag)')
        ax.grid(True, color='lightgray', zorder=-1000)
        df_y = pd.DataFrame({'Y': y_values, 'Label': y_labels}).sort_values(by='Y')
        n = len(df_y)
        t_values = [norm.ppf((k - 0.5) / n) for k in range(1, n + 1)]
        ax.scatter(x=t_values, y=df_y['Y'], alpha=0.6, color=comp_color, zorder=+1000)
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

    this_directory, mp_string, an_string = get_context()

    # Delete any previous image files from current directory:
    image_filenames = [f for f in os.listdir('.')
                       if f.startswith('Image') and f.endswith('.png')]
    for f in image_filenames:
        os.remove(f)

    # Wrangle needed data into convenient forms:
    df_plot = pd.merge(left=df_model.loc[df_model['UseInModel'], :].copy(),
                       right=model.mm_fit.df_observations,
                       how='left', left_index=True, right_index=True, sort=False)  # add col 'Residuals'.
    is_comp_obs = (df_plot['Type'] == 'Comp')
    df_plot_comp_obs = df_plot.loc[is_comp_obs, :]
    df_plot_mp_obs = df_plot.loc[(~ is_comp_obs), :]
    df_image_effect = model.mm_fit.df_random_effects
    df_image_effect.rename(columns={"GroupName": "FITSfile", "Group": "ImageEffect"}, inplace=True)
    intercept = model.mm_fit.df_fixed_effects.loc['Intercept', 'Value']
    # jd_slope = model.mm_fit.df_fixed_effects.loc['JD_fract', 'Value']  # undefined if FIT_JD is False.
    sigma = model.mm_fit.sigma
    if 'Airmass' in model.mm_fit.df_fixed_effects.index:
        extinction = model.mm_fit.df_fixed_effects.loc['Airmass', 'Value']  # if fit in model
    else:
        extinction = state['extinction']['Clear']  # default if not fit in model (normal case)
    if 'CI' in model.mm_fit.df_fixed_effects.index:
        transform = model.mm_fit.df_fixed_effects.loc['CI', 'Value']  # if fit in model
    else:
        transform = TRANSFORM_CLEAR_SR_SR_SI  # default if not fit in model (normal case)
    if model.fit_jd:
        jd_coefficient = model.mm_fit.df_fixed_effects.loc['JD_fract', 'Value']
    else:
        jd_coefficient = 0.0
    comp_ids = df_plot_comp_obs['SourceID'].drop_duplicates()
    n_comps = len(comp_ids)

    comp_color, mp_color = 'dimgray', 'orangered'
    obs_colors = [comp_color if i is True else mp_color for i in is_comp_obs]

    jd_floor = floor(min(df_model['JD_mid']))
    obs_jd_fract = df_plot['JD_mid'] - jd_floor
    xlabel_jd = 'JD(mid)-' + str(jd_floor)

    # ################ FIGURE 1: Q-Q plot of mean comp effects (one point per comp star used in model),
    #    code heavily adapted from photrix.process.SkyModel.plots():
    window_title = 'Q-Q Plot (by comp):  MP ' + mp_string + '   AN ' + an_string
    page_title = 'MP ' + mp_string + '   AN ' + an_string + '   ::   Q-Q plot by comp (mean residual)'
    plot_annotation = str(n_comps) + ' comps used in model.' + \
        '\n(tags: comp SourceID)'
    df_y = df_plot_comp_obs.loc[:, ['SourceID', 'Residual']].groupby(['SourceID']).mean()
    df_y = df_y.sort_values(by='Residual')
    y_values = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y.index.values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_values, y_labels,
                          'Image1_QQ_comps.png')

    # ################ FIGURE 2: Q-Q plot of comp residuals (one point per comp obs),
    #    code heavily adapted from photrix.process.SkyModel.plots():
    window_title = 'Q-Q Plot (by comp observation):  MP ' + mp_string + '   AN ' + an_string
    page_title = 'MP ' + mp_string + '   AN ' + an_string + '   ::   Q-Q plot by comp observation'
    plot_annotation = str(len(df_plot_comp_obs)) + ' observations of ' + \
        str(n_comps) + ' comps used in model.' + \
        '\n (tags: observation Serial numbers)'
    df_y = df_plot_comp_obs.loc[:, ['Serial', 'Residual']]
    df_y = df_y.sort_values(by='Residual')
    y_values = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y['Serial'].values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_values, y_labels,
                          'Image2_QQ_obs.png')

    # ################ FIGURE 3: Catalog and Time plots:
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
    make_labels_9_subplots(ax, 'Catalog Mag Uncertainty (dr)', 'Catalog Mag (r)', 'mMag', '',
                           zero_line=False)
    ax.scatter(x=df_plot_comp_obs['r'], y=df_plot_comp_obs['dr'], s=14, alpha=0.3, color='black')

    # Catalog color plot (comps only, one point per comp, x=cat r mag, y=cat color (r-i)):
    ax = axes[0, 1]
    make_labels_9_subplots(ax, 'Catalog Color Index', 'Catalog Mag (r)', 'CI Mag', '', zero_line=False)
    ax.scatter(x=df_plot_comp_obs['r'], y=(df_plot_comp_obs['r'] - df_plot_comp_obs['i']),
               s=14, alpha=0.3, color='black')
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=len(model.df_mp_mags) * [mp_color_ri],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)

    # Inst Mag plot (comps only, one point per obs, x=cat r mag, y=InstMagSigma):
    ax = axes[0, 2]
    make_labels_9_subplots(ax, 'Instrument Magnitude Uncertainty', 'Catalog Mag (r)', 'mMag', '',
                           zero_line=True)
    ax.scatter(x=df_plot_comp_obs['r'], y=df_plot_comp_obs['InstMagSigma'], s=14, alpha=0.3, color='black')
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=model.df_mp_mags['InstMagSigma'],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)

    # Cirrus plot (comps only, one point per image, x=JD_fract, y=Image Effect):
    ax = axes[1, 0]
    make_labels_9_subplots(ax, 'Image effect (cirrus plot)', xlabel_jd, 'mMag', '', zero_line=False)
    df_this_plot = pd.merge(df_image_effect, df_plot_comp_obs.loc[:, ['FITSfile', 'JD_fract']],
                            how='left', on='FITSfile', sort=False).drop_duplicates()
    ax.scatter(x=df_this_plot['JD_fract'], y=1000.0 * df_this_plot['ImageEffect'],
               s=14, alpha=1, color='black')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))

    # SkyADU plot (comps only, one point per obs: x=JD_fract, y=SkyADU):
    ax = axes[1, 1]
    make_labels_9_subplots(ax, 'SkyADU vs time', xlabel_jd, 'ADU', '', zero_line=False)
    ax.scatter(x=df_plot_comp_obs['JD_fract'], y=df_plot_comp_obs['SkyADU'],
               s=14, alpha=0.3, color='black')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))

    # FWHM plot (comps only, one point per obs: x=JD_fract, y=FWHM):
    ax = axes[1, 2]
    make_labels_9_subplots(ax, 'FWHM vs time', xlabel_jd, 'FWHM (pixels)', '', zero_line=False)
    ax.scatter(x=df_plot_comp_obs['JD_fract'], y=df_plot_comp_obs['FWHM'],
               s=14, alpha=0.3, color='black')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))

    # InstMagSigma plot (comps only, one point per obs; x=JD_fract, y=InstMagSigma):
    ax = axes[2, 0]
    make_labels_9_subplots(ax, 'Inst Mag Sigma vs time', xlabel_jd, 'mMag', '', zero_line=False)
    ax.scatter(x=df_plot_comp_obs['JD_fract'], y=1000.0 * df_plot_comp_obs['InstMagSigma'],
               s=14, alpha=0.3, color='black')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))

    # Airmass plot (comps only, one point per obs; x=JD_fract, y=Airmass):
    ax = axes[2, 1]
    make_labels_9_subplots(ax, 'Airmass vs time', xlabel_jd, 'Airmass', '', zero_line=False)
    ax.scatter(x=df_plot_comp_obs['JD_fract'], y=df_plot_comp_obs['Airmass'],
               s=14, alpha=0.3, color='black')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))

    # Session Lightcurve plot (comps only, one point per obs; x=JD_fract, y=MP best magnitude):
    ax = axes[2, 2]
    make_labels_9_subplots(ax, 'MP Lightcurve for this session', xlabel_jd, 'Mag (r)', '', zero_line=False)
    ax.errorbar(x=model.df_mp_mags['JD_mid'] - jd_floor, y=model.df_mp_mags['MP_Mags'],
                yerr=model.df_mp_mags['InstMagSigma'], fmt='none', color='black',
                linewidth=0.5, capsize=3, capthick=0.5, zorder=-100)
    ax.scatter(x=model.df_mp_mags['JD_mid'] - jd_floor, y=model.df_mp_mags['MP_Mags'],
               s=14, alpha=1, color='black')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))
    ax.invert_yaxis()  # per custom of plotting magnitudes brighter=upward

    plt.show()
    fig.savefig('Image3_Catalog_and_Time.png')

    # ################ FIGURE 4: Residual plots:
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
    make_labels_9_subplots(ax, 'Model residual vs r (catalog)',
                           'Catalog Mag (r)', 'mMag',
                           '', zero_line=True)
    ax.scatter(x=df_plot_comp_obs['r'], y=1000.0 * df_plot_comp_obs['Residual'],
               s=14, alpha=0.3, color='black')
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=len(model.df_mp_mags) * [0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)
    draw_x_line(ax, user_selections['min_r_mag'])
    draw_x_line(ax, user_selections['min_r_mag'])

    # Comp residual plot (comps only, one point per obs: x=raw Instrument Mag, y=model residual):
    ax = axes[0, 1]
    make_labels_9_subplots(ax, 'Model residual vs raw Instrument Mag',
                           'Raw instrument mag', 'mMag',
                           '', zero_line=True)
    ax.scatter(x=df_plot_comp_obs['InstMag'], y=1000.0 * df_plot_comp_obs['Residual'],
               s=14, alpha=0.3, color='black')
    ax.scatter(x=model.df_mp_mags['InstMag'], y=len(model.df_mp_mags) * [0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)

    # Comp residual plot (comps only, one point per obs: x=catalog r-i color, y=model residual):
    ax = axes[0, 2]
    make_labels_9_subplots(ax, 'Model residual vs Color Index (cat)',
                           'Catalog Color (r-i)', 'mMag',
                           '', zero_line=True)
    ax.scatter(x=(df_plot_comp_obs['r'] - df_plot_comp_obs['i']), y=1000.0 * df_plot_comp_obs['Residual'],
               s=14, alpha=0.3, color='black')
    ax.scatter(x=[mp_color_ri], y=[0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)

    # Comp residual plot (comps only, one point per obs: x=Julian Date fraction, y=model residual):
    ax = axes[1, 0]
    make_labels_9_subplots(ax, 'Model residual vs JD',
                           xlabel_jd, 'mMag',
                           '', zero_line=True)
    ax.scatter(x=df_plot_comp_obs['JD_fract'],
               y=1000.0 * df_plot_comp_obs['Residual'],
               s=14, alpha=0.3, color='black')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))

    # Comp residual plot (comps only, one point per obs: x=Airmass, y=model residual):
    ax = axes[1, 1]
    make_labels_9_subplots(ax, 'Model residual vs Airmass',
                           'Airmass', 'mMag',
                           '', zero_line=True)
    ax.scatter(x=df_plot_comp_obs['Airmass'],
               y=1000.0 * df_plot_comp_obs['Residual'],
               s=14, alpha=0.3, color='black')

    # Comp residual plot (comps only, one point per obs: x=Sky Flux (ADUs), y=model residual):
    ax = axes[1, 2]
    make_labels_9_subplots(ax, 'Model residual vs Sky Flux',
                           'Sky Flux (ADU)', 'mMag',
                           '', zero_line=True)
    ax.scatter(x=df_plot_comp_obs['SkyADU'],
               y=1000.0 * df_plot_comp_obs['Residual'],
               s=14, alpha=0.3, color='black')

    # Comp residual plot (comps only, one point per obs: x=X in images, y=model residual):
    ax = axes[2, 0]
    make_labels_9_subplots(ax, 'Model residual vs X in image',
                           'X from center (pixels)', 'mMag',
                           '', zero_line=True)
    ax.scatter(x=df_plot_comp_obs['X1024'] * 1024.0,
               y=1000.0 * df_plot_comp_obs['Residual'],
               s=14, alpha=0.3, color='black')
    draw_x_line(ax, 0.0)

    # Comp residual plot (comps only, one point per obs: x=Y in images, y=model residual):
    ax = axes[2, 1]
    make_labels_9_subplots(ax, 'Model residual vs Y in image',
                           'Y from center (pixels)', 'mMag',
                           '', zero_line=True)
    ax.scatter(x=df_plot_comp_obs['Y1024'] * 1024.0,
               y=1000.0 * df_plot_comp_obs['Residual'],
               s=14, alpha=0.3, color='black')
    draw_x_line(ax, 0.0)

    # Comp residual plot (comps only, one point per obs: x=vignette (dist from center), y=model residual):
    ax = axes[2, 2]
    make_labels_9_subplots(ax, 'Model residual vs distance from center',
                           'dist from center (pixels)', 'mMag',
                           '', zero_line=True)
    ax.scatter(x=1024*np.sqrt(df_plot_comp_obs['Vignette']),
               y=1000.0 * df_plot_comp_obs['Residual'],
               s=14, alpha=0.3, color='black')

    plt.show()
    fig.savefig('Image4_Residuals.png')

    # ################ FIGURE(S) 5: Variability plots:
    # Several comps on a subplot, vs JD, normalized by (minus) the mean of all other comps' responses.
    # Make df_offsets (one row per obs, at first with only raw offsets):
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
                make_labels_9_subplots(ax, 'Comp Variability plot', xlabel_jd, 'mMag', '', zero_line=True)
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
        fig.savefig('Image5_Comp Variability_' + '{:02d}'.format(i_figure + 1) + '.png')
    # Verify that all comps were plotted exactly once (debug):
    all_comps_plotted_once = (sorted(comp_ids) == sorted(plotted_comp_ids))
    # print('all comp ids were plotted exactly once = ', str(all_comps_plotted_once))
    if not all_comps_plotted_once:
        print('comp ids plotted more than once',
              [item for item, count in Counter(plotted_comp_ids).items() if count > 1])


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
    lines.append('# ALCDEF file for MP' + mp_string + '  AN ' + an_string)
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
    earth_mp_au = session_dict['Delta']
    sun_mp_au = session_dict['R']
    reduced_mag_correction = -5.0 * log10(earth_mp_au * sun_mp_au)
    lines.append('UCORMAG=' + '{0:.4f}'.format(reduced_mag_correction))
    lines.append('OBJECTRA=' + ra_as_hours(session_dict['RA']).rsplit(':', 1)[0])
    lines.append('OBJECTDEC=' + dec_as_hex(round(session_dict['Dec'])).split(':')[0])
    lines.append('PHASE=+' + '{0:.1f}'.format(abs(session_dict['Phase'])))
    lines.append('PABL=' + '{0:.1f}'.format(abs(session_dict['PAB_longitude'])))
    lines.append('PABB=' + '{0:.1f}'.format(abs(session_dict['PAB_latitude'])))
    lines.append('# Comp stars omitted: our ATLAS workflow uses many more than ALCDEF\'s max of 10.')
    lines.append('CICORRECTION=TRUE')
    lines.append('CIBAND=SRI')
    lines.append('CITARGET=' + '{0:.2f}'.format(mp_color_ri) + '  # originating from ' + source_string_ri)
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


class SessionModel:
    def __init__(self, df_model, mp_color_ri, state, options_dict):
        """  Makes and holds photometric model via mixed-model regression. Affords prediction for MP mags.
        :param df_model:
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

        self.dep_var_name = 'InstMag_SR_with_offsets'
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
        self.df_used_comps_only['CI'] = self.df_used_comps_only['r'] - self.df_used_comps_only['i']

        # Initiate dependent-variable offset, which will aggregate all such offset terms:
        dep_var_offset = self.df_used_comps_only['r'].copy()  # *copy* CatMag, or it will be damaged

        # Build fixed-effect (x) variable list and construct dep-var offset:
        fixed_effect_var_list = []
        if self.fit_transform:
            fixed_effect_var_list.append('CI')
        else:
            self.transform_fixed = TRANSFORM_CLEAR_SR_SR_SI
            dep_var_offset += self.transform_fixed * self.df_used_comps_only['CI']
            print(' >>>>> Transform (Color Index) not fit: value fixed at',
                  '{0:.3f}'.format(self.transform_fixed))
        if self.fit_extinction:
            fixed_effect_var_list.append('Airmass')
        else:
            extinction = self.state['extinction']['Clear']
            dep_var_offset += extinction * self.df_used_comps_only['Airmass']
            print(' >>>>> Extinction (Airmass) not fit: value fixed at',
                  '{0:.3f}'.format(extinction))
        if self.fit_vignette:
            fixed_effect_var_list.append('Vignette')
        if self.fit_xy:
            fixed_effect_var_list.extend(['X1024', 'Y1024'])
        if self.fit_jd:
            fixed_effect_var_list.append('JD_fract')
        if len(fixed_effect_var_list) == 0:
            fixed_effect_var_list = ['JD_fract']  # as statsmodels requires >= 1 fixed-effect varialble.

        # Build 'random-effect' variable:
        random_effect_var_name = 'FITSfile'  # cirrus effect is per-image

        # Build dependent (y) variable:
        self.df_used_comps_only[self.dep_var_name] = self.df_used_comps_only['InstMag'] - dep_var_offset

        # Execute regression:
        self.mm_fit = MixedModelFit(data=self.df_used_comps_only,
                                    dep_var=self.dep_var_name,
                                    fixed_vars=fixed_effect_var_list,
                                    group_var=random_effect_var_name)
        print(self.mm_fit.statsmodels_object.summary())

    def _calc_mp_mags(self):
        bogus_cat_mag = 0.0  # we'll need this below, to correct raw predictions.
        self.df_used_mps_only['CatMag'] = bogus_cat_mag  # totally bogus local value, corrected for later.
        self.df_used_mps_only['CI'] = self.mp_ci
        raw_predictions = self.mm_fit.predict(self.df_used_mps_only, include_random_effect=True)

        # Compute dependent-variable offsets for MP:
        dep_var_offsets = pd.Series(len(self.df_used_mps_only) * [0.0], index=raw_predictions.index)
        if self.fit_transform is False:
            dep_var_offsets += self.transform_fixed * self.df_used_mps_only['CI']
        if self.fit_extinction is False:
            dep_var_offsets += self.state['extinction']['Clear'] * self.df_used_mps_only['Airmass']

        # Extract best MP mag (in Sloan r):
        mp_mags = self.df_used_mps_only['InstMag'] \
            - dep_var_offsets - raw_predictions + bogus_cat_mag  # correct for use of bogus cat mag.
        df_mp_mags = pd.DataFrame({'MP_Mags': mp_mags}, index=list(mp_mags.index))
        df_mp_mags = pd.merge(left=df_mp_mags,
                              right=self.df_used_mps_only.loc[:, ['JD_mid', 'FITSfile',
                                                                  'InstMag', 'InstMagSigma']],
                              how='left', left_index=True, right_index=True, sort=False)
        return df_mp_mags


_____SUPPORT________________________________________________ = 0


def get_context():
    """ This is run at beginning of workflow functions (except start()) to orient the function.
    :return: 3-tuple: (this_directory, mp_string, an_string) [3 strings]
    """
    this_directory = os.getcwd()
    if not os.path.isfile(LOG_FILENAME):
        print(' >>>>> ERROR: no log file found ==> You probably need to run start() or resume().')
        return None
    log_file = open(LOG_FILENAME, mode='r')  # for read only
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


def read_mp_positions():
    """ Reads control.txt, parses 2 #MP lines, returns MP-position filenames, x-pixels, and y_pixels.
    :return: mp_position_filenames [list of 2 strings], x_pixels, y_pixels [each list of 2 floats].
    """
    mp_position_filenames, x_pixels, y_pixels = [], [], []
    with open(CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines
        for line in lines:
            if len(mp_position_filenames) >= 2:
                break
            if line.upper().startswith('#MP'):
                words = line[3:].rsplit(maxsplit=2)
                if any([w.startswith('[') for w in words]):
                    break
                mp_position_filenames.append(words[0].strip())
                try:
                    x_pixels.append(float(words[1]))
                    y_pixels.append(float(words[2]))
                except ValueError:
                    break
    if (len(mp_position_filenames), len(x_pixels), len(y_pixels)) != (2, 2, 2):
        mp_position_filenames, x_pixels, y_pixels = None, None, None
    return mp_position_filenames, x_pixels, y_pixels


def read_df_obs_all():
    """  Simple utility to read df_obs.csv file and return the original DataFrame.
    :return: df_obs from make_dfs() [pandas Dataframe]
    """
    this_directory, _, _ = get_context()
    fullpath = os.path.join(this_directory, DF_OBS_ALL_FILENAME)
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
    fullpath = os.path.join(this_directory, DF_IMAGES_ALL_FILENAME)
    df_images = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_images


def read_df_comps_all():
    """  Simple utility to read df_comps.csv file and return the original DataFrame.
    :return: df_comps from make_dfs() [pandas Dataframe]
    """
    this_directory, _, _ = get_context()
    fullpath = os.path.join(this_directory, DF_COMPS_ALL_FILENAME)
    df_comps = pd.read_csv(fullpath, sep=';', index_col=0)
    comp_ids = [str(id) for id in df_comps['CompID']]  # ensure strings.
    df_comps.loc[:, 'CompID'] = comp_ids
    df_comps.index = comp_ids  # list, to ensure index not named.
    return df_comps


def get_mp_color_index(df_obs_all, df_images_all, df_comps_all):
    """  Gets minor planet's Sloan r-i color index from best source available.
    Uses first (most preferred) of these color index value possibilities that actually exists:
        (1) value extracted from (Johnson-Cousins) R & I images, that is, r-i color index from
            regression on R & I instrument magnitudes; or
        (2) value given in control.txt #MP_RI_COLOR; or
        (3) DEFAULT_RI_COLOR as constant atop this python file.
    :param df_obs_all:
    :param df_images_all:
    :param df_comps_all:
    :return: tuple (color index (Sloan r-i), color sigma, n color comps,
             source of value). Sigma and n are None if color not calculated in this run (i.e.,
             if default or control.txt value is used. [4-tuple of float, string]
    """
    # Case 1: R & I images are available in this directory:
    is_r_image = (df_images_all['Filter'] == 'R')
    is_i_image = (df_images_all['Filter'] == 'I')
    ri_both_available = any(is_r_image) and any(is_i_image)
    if ri_both_available:
        r_filenames = list(df_images_all.loc[is_r_image, 'FITSfile'])
        i_filenames = list(df_images_all.loc[is_i_image, 'FITSfile'])
        ci_filenames = r_filenames + i_filenames

        # Keep only color-index images with a valid MP observation:
        valid_mp_filenames = []
        for filename in ci_filenames:
            is_filename_mp_row = (df_obs_all['FITSfile'] == filename) & (df_obs_all['Type'] == 'MP')
            if sum(is_filename_mp_row) == 1:
                mp_instmag = df_obs_all.loc[is_filename_mp_row, 'InstMag'].iloc[0]
                mp_instmagsigma = df_obs_all.loc[is_filename_mp_row, 'InstMagSigma'].iloc[0]
                if (mp_instmag is not None) and (mp_instmagsigma is not None):
                    if (not np.isnan(mp_instmag)) and (not np.isnan(mp_instmagsigma)):
                        if mp_instmagsigma < MAX_MAG_UNCERT:
                            valid_mp_filenames.append(filename)
        ci_filenames = valid_mp_filenames.copy()

        # Make list of all comps in color-index images, and having high signal-to-noise:
        is_ci_obs = [(f in ci_filenames) for f in df_obs_all['FITSfile']]
        df_ci_obs = df_obs_all[is_ci_obs].copy()  # includes MP observations.
        is_ci_comp_obs = [type == 'Comp' for type in df_ci_obs['Type']]
        ci_comp_ids = df_ci_obs.loc[is_ci_comp_obs, 'SourceID'].drop_duplicates()
        is_high_snr = [(df_comps_all.loc[c, 'dr'] < 10.0) and (df_comps_all.loc[c, 'di'] < 10.0)
                       for c in ci_comp_ids]
        ci_comp_ids = ci_comp_ids[is_high_snr].copy()

        # Keep only comps which are represented in every color-index image:
        ci_comp_list = []
        n_ci_images = len(r_filenames) + len(i_filenames)
        for this_comp_id in ci_comp_ids:
            n_images_this_comp = sum(df_ci_obs['SourceID'] == this_comp_id)
            if n_images_this_comp == n_ci_images:
                ci_comp_list.append(this_comp_id)
        is_comp_in_list = [(id in ci_comp_list) for id in df_ci_obs['SourceID']]
        df_ci_obs = df_ci_obs[is_comp_in_list].copy()

        # Make one merged dataframe, one row per eligible comp observation, but only selected rows:
        df_fit_obs = pd.merge(left=df_ci_obs.loc[:, ['FITSfile', 'SourceID', 'Type', 'InstMag']],
                              right=df_comps_all.loc[:, ['CompID', 'r', 'i']],
                              how='left', left_on='SourceID', right_on='CompID', sort=False)
        df_fit_obs = pd.merge(left=df_fit_obs, right=df_images_all.loc[:, ['FITSfile', 'Filter']],
                              how='left', on='FITSfile', sort=False)
        df_fit_obs = df_fit_obs.sort_values(by=['SourceID', 'Filter', 'FITSfile'])  # for review/debugging.

        # Make variables for regression fit:
        fit_dict_list = []
        is_r_row = (df_fit_obs['Filter'] == 'R')
        is_i_row = (df_fit_obs['Filter'] == 'I')
        for this_comp_id in ci_comp_list:
            fit_dict = dict()
            fit_dict['comp_id'] = this_comp_id
            fit_dict['CI'] = df_comps_all.loc[this_comp_id, 'r'] - df_comps_all.loc[this_comp_id, 'i']

            is_comp_id = (df_fit_obs['SourceID'] == this_comp_id)
            is_comp_id_r = is_comp_id & is_r_row
            is_comp_id_i = is_comp_id & is_i_row
            comp_id_mean_r_instmag = df_fit_obs.loc[is_comp_id_r, 'InstMag'].mean()  # averaged over images.
            comp_id_mean_i_instmag = df_fit_obs.loc[is_comp_id_i, 'InstMag'].mean()  # "
            fit_dict['InstMagDiff'] = comp_id_mean_r_instmag - comp_id_mean_i_instmag
            fit_dict_list.append(fit_dict)
        df_fit = pd.DataFrame(data=fit_dict_list)
        df_fit.index = list(df_fit['comp_id'])

        # Perform regression fit:
        result = smf.ols(formula='CI ~ InstMagDiff', data=df_fit).fit()
        # print(result.summary())
        # fig = plt.figure(figsize=(12, 8))
        # fig = sm.graphics.plot_regress_exog(result, "indep_var", fig=fig)
        # plt.show()

        # Build MP input data for color-index (Sloan r-i) prediction:
        instmag_list_r, instmag_list_i, sigma_list_r, sigma_list_i = [], [], [], []
        mp_dict = dict()
        for filename in df_fit_obs['FITSfile'].drop_duplicates():
            mp_instmag = df_obs_all.loc[(df_obs_all['FITSfile'] == filename) &
                                        (df_obs_all['Type'] == 'MP'), 'InstMag'].iloc[0]
            mp_sigma = df_obs_all.loc[(df_obs_all['FITSfile'] == filename) &
                                      (df_obs_all['Type'] == 'MP'), 'InstMagSigma'].iloc[0]
            if df_images_all.loc[filename, 'Filter'] == 'R':
                instmag_list_r.append(mp_instmag)
                sigma_list_r.append(mp_sigma)
            if df_images_all.loc[filename, 'Filter'] == 'I':
                instmag_list_i.append(mp_instmag)
                sigma_list_i.append(mp_sigma)
        mp_dict['InstMagDiff'] = sum(instmag_list_r) / len(instmag_list_r)\
            - sum(instmag_list_i) / len(instmag_list_i)
        df_prediction = pd.DataFrame(data=mp_dict, index=range(len(mp_dict)))

        # Predict MP color index and its uncertainty, then return them:
        mp_color = result.predict(df_prediction)
        mag_diff = df_prediction['InstMagDiff'].iloc[0]
        sigma_intercept = result.bse.Intercept
        sigma_slope = result.bse.InstMagDiff
        mean_diff = result.model.data.frame['InstMagDiff'].mean()
        variance_color = sigma_intercept**2 + sigma_slope**2 * (mag_diff - mean_diff)**2 + \
                         sum([sigma ** 2 for sigma in sigma_list_r]) / (len(sigma_list_r) ** 2) + \
                         sum([sigma ** 2 for sigma in sigma_list_i]) / (len(sigma_list_i) ** 2)
        sigma_color = sqrt(variance_color)
        n_ci_comps = len(ci_comp_list)
        return mp_color[0], sigma_color, n_ci_comps, 'R & I images'

    # Case 2: return MP_COLOR_INDEX from control.txt:
    with open(CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines
        ri_color = None
        for line in lines:
            if line.upper().startswith('#MP_RI_COLOR '):
                ri_string = line[len('#MP_RI_COLOR '):].split()[0]
                try:
                    ri_color = float(ri_string)
                except ValueError:
                    ri_color = None
        if ri_color is not None:
            return ri_color, None, None, 'control.txt'

    # Default case: return value hard-coded atop this python file:
    return DEFAULT_MP_RI_COLOR, None, None, 'default'


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
    refcat2.select_dgaia()
    lines.append('Refcat2: dgaia screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.remove_overlapping()
    lines.append('Refcat2: overlaps removed to ' + str(len(refcat2.df_selected)) + ' stars.')
    return lines


def read_user_selections():
    """ Reads observation selection lines from control.txt, compiles lists of observations to remove
            (for use by apply_user_selections()).
    :return: criteria to remove observations from df_obs before use in model. [dict]
    """
    # Parse control.txt:
    with open(CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines

    serial_list, comp_list, image_list = [], [], []
    min_r_mag, max_r_mag = PHOT_MIN_R_MAG, PHOT_MAX_R_MAG
    max_catalog_dr_mmag = PHOT_MAX_CATALOG_DR_MMAG
    min_sloan_ri_color, max_sloan_ri_color = PHOT_MIN_SLOAN_RI_COLOR, PHOT_MAX_SLOAN_RI_COLOR
    for line in lines:
        content = line.strip().split(';')[0].strip()  # upper case, comments removed.
        content_upper = content.upper()
        if content_upper.startswith('#SERIAL'):
            values = content[len('#SERIAL'):].strip().replace(',', ' ').split()
            serial_list.extend(values)
        if content_upper.startswith('#COMP'):
            values = content[len('#COMP'):].strip().replace(',', ' ').split()
            comp_list.extend(values)
        if content_upper.startswith('#IMAGE'):
            image_filename = content[len('#IMAGE'):].strip()
            image_list.append(image_filename)
        if content_upper.startswith('#MIN_R_MAG'):
            try:
                min_r_mag = float(content[len('#MIN_R_MAG'):].strip())
            except ValueError:
                print(' >>>>> WARNING: #MIN_R_MAG in control.txt cannot be parsed as float; default used.')
        if content_upper.startswith('#MAX_R_MAG'):
            try:
                max_r_mag = float(content[len('#MAX_R_MAG'):].strip())
            except ValueError:
                print(' >>>>> WARNING: #MAX_R_MAG in control.txt cannot be parsed as float; default used.')
        if content_upper.startswith('#MAX_CATALOG_DR_MMAG'):
            try:
                max_catalog_dr_mmag = float(content[len('#MAX_CATALOG_DR_MMAG'):].strip())
            except ValueError:
                print(' >>>>> WARNING: #MAX_CATALOG_DR_MMAG in control.txt cannot be parsed as float; '
                      'default used.')
        if content_upper.startswith('#MIN_SLOAN_RI_COLOR'):
            try:
                min_sloan_ri_color = float(content[len('#MIN_SLOAN_RI_COLOR'):].strip())
            except ValueError:
                print(' >>>>> WARNING: #MIN_SLOAN_RI_COLOR in control.txt cannot be parsed as float; ' +
                      'default used.')
        if content_upper.startswith('#MAX_SLOAN_RI_COLOR'):
            try:
                max_sloan_ri_color = float(content[len('#MAX_SLOAN_RI_COLOR'):].strip())
            except ValueError:
                print(' >>>>> WARNING: #MAX_SLOAN_RI_COLOR in control.txt cannot be parsed as float; ' +
                      'default used.')
    return {'serials': serial_list, 'comps': comp_list, 'images': image_list,
            'min_r_mag': min_r_mag, 'max_r_mag': max_r_mag,
            'max_catalog_dr_mmag': max_catalog_dr_mmag,
            'min_sloan_ri_color': min_sloan_ri_color, 'max_sloan_ri_color': max_sloan_ri_color}


def apply_user_selections(df_model, user_selections):
    """ Reads file control.txt and sets df_model's 'UseInModel' row to False, in-place.
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
    df_model.loc[obs_to_deselect, ['UseInModel']] = False


def sync_comps_and_images(df_all_columns, df_comps, df_images):
    """ Utility to return df_comps and df_images
            whose rows are selected by representation in df_all_columns.
    :param df_all_columns: the master dataframe, immutable [pandas DataFrame]
    :param df_comps: the comps dataframe to select from. [pandas DataFrame]
    :param df_images: the images dataframe to select from. [pandas DataFrame]
    :return: df_comps_synced, df_images_synced
    """
    comp_list = df_all_columns['CompID'].drop_duplicates().copy()
    is_in_comp_list = list(df_comps['CompID'].isin(comp_list))
    df_comps_synced = df_comps.loc[is_in_comp_list, :].copy()
    image_list = df_all_columns['FITSfile'].drop_duplicates().copy()
    is_in_image_list = list(df_images['FITSfile'].isin(image_list))
    df_images_synced = df_images.loc[is_in_image_list, :].copy()
    return df_comps_synced, df_images_synced


def read_model_options():
    """ Returns dict of options for SessionModel (mixed-model fit of comp data).
        Reads from control.txt, uses default for any option not read in from there."""
    # Parse control.txt:
    with open(CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines
    option_dict = DEFAULT_MODEL_OPTIONS.copy()
    for line in lines:
        content = line.upper().strip().split(';')[0].strip()  # upper case, comments removed.
        for key in option_dict.keys():
            directive_key = '#' + key.upper()
            if content.startswith(directive_key):
                value = content[len(directive_key):].strip().lower()
                if value.startswith('y') or value == 'true':
                    option_dict[key] = True
                elif value.startswith('n') or value == 'false':
                    option_dict[key] = False
                else:
                    print(' >>>>> control.txt:', directive_key, 'value not understood.')
    return option_dict


# def make_diagnostics(df_model):
#     """ Use mixed-model regression to separate image effects ("random effect")
#             from comp effects (comp-averaged residual). Present data ~ready for plotting etc.
#     :param df_model: merged (obs, comps, images) dataframe of observation data [pandas DataFrame]
#     :return: df_model(updated with image effect etc), df_image_diagnostics,
#              df_comp_diagnostics. [yes, 3 pandas DataFrames]
#     """
#     # Extract comps & their statistics, assemble other data:
#     is_comp_id = (df_model['Type'] == 'Comp')
#     is_phot_filter = (df_model['Filter'] == MP_PHOTOMETRY_FILTER)
#     to_keep = is_comp_id & is_phot_filter
#     df = df_model.loc[to_keep, :].copy()  # (maybe cut down list of columns, later)
#     # comps = df['SourceID'].drop_duplicates().copy()
#     # images = df['FITSfile'].drop_duplicates().copy()
#     transform = TRANSFORM_CLEAR_SR_SR_SI
#     state = get_session_state()
#     extinction = state['extinction']['Clear']
#
#     # Make dep. variable from InstMags adjusted for terms: r mag, transform, extinction:
#     r_mag_term = df['r']
#     transform_term = transform * df['Airmass']
#     extinction_term = extinction * (df['r'] - df['i'])
#     df['DepVar'] = df['InstMag'] - r_mag_term - transform_term - extinction_term
#
#     # Do mixed-model regression with
#     fit = MixedModelFit(data=df, dep_var='DepVar', fixed_vars=['JD_fract'], group_var='FITSfile')
#
#     # Make df_image_diagnostics & merge onto df_model:
#     df_image_diagnostics = fit.df_random_effects.sort_values(by='GroupName').copy()
#     df_image_diagnostics = df_image_diagnostics.rename(columns={"GroupName": "FITSfile",
#                                                                 "Group": "ImageEffect"})
#     df_model = pd.merge(df_model, df_image_diagnostics, how='left', on='FITSfile', sort=False)
#
#     # Make df_comp_diagnostics; merge residual and CompEffect onto df_model:
#     df_comp_obs = fit.df_observations.copy().drop(columns=['FittedValue'])
#     df_comp_obs['Serial'] = list(df_comp_obs.index)
#     df = pd.merge(df, df_comp_obs, how='left', on='Serial', sort=False)
#     df_comp_effect = df.loc[:, ['SourceID', 'Residual']].groupby(['SourceID']).mean()  # excludes NaNs ++ !
#     df_comp_effect = df_comp_effect.rename(columns={'Residual': 'CompEffect'})
#     df = pd.merge(df, df_comp_effect, how='left', on='SourceID', sort=False)
#
#     # Make new df_model columns:
#     df['InstMagAdjusted'] = df['InstMag'] - transform_term - extinction_term
#
#     return df_model











# def make_comp_diagnostics(df_model):
#     """ Return for each comp: offset from InstMag expected, mean offset, offset metric,
#             where "expected" IM is calc from mixed-model fit on all *other* comps.
#     :param df_model: merged (obs, comps, images) dataframe of observation data [pandas DataFrame]
#     :return: df of one row/comp, columns=CompID, offsets (list), mean_offset, offset_metric;
#                   suitable for use in dignostic plotting. [pandas DataFrame]
#     """
#     # Extract comps & their statistics:
#     is_comp_id = (df_model['Type'] == 'Comp')
#     is_phot_filter = (df_model['Filter'] == MP_PHOTOMETRY_FILTER)
#     to_keep = is_comp_id & is_phot_filter
#     df = df_model.loc[to_keep, :].copy()  # (maybe cut down list of columns, later)
#     comps = df['SourceID'].drop_duplicates().copy()
#     images = df['FITSfile'].drop_duplicates().copy()
#
#     transform = TRANSFORM_CLEAR_SR_SR_SI
#     state = get_session_state()
#     extinction = state['extinction']['Clear']
#
#     dict_list = []
#     # Loop over comps:
#     for comp in comps:
#         df_other = df.loc[df['SourceID'] != comp, :].copy()  # (maybe cut down list of columns, later)
#
#         # Get Z (estimated nightly zero-point) from *other* comps
#         mean_z = (df_other['InstMag'] - df_other['r'] -
#                   transform * (df_other['r'] - df_other['i']) - extinction * df_other['Airmass']).mean()
#
#         # Get mean random effect (per-image general variation) form *other* comps:
#         offsets = []
#         for i, image in enumerate(images):
#             df_image = df_other.loc[df_other['FITSfile'] == image, :]
#             image_effect = (df_image['InstMag'] - mean_z - df_image['r'] -
#                             transform * (df_image['r'] - df_image['i']) -
#                             extinction * df_image['Airmass']).mean()
#
#             # Get offset for this comp, this image:
#             is_this_obs = (df['SourceID'] == comp) & (df['FITSfile'] == image)
#             inst_mag = df.loc[is_this_obs, 'InstMag']
#             r_catmag = df.loc[is_this_obs, 'r']
#             i_catmag = df.loc[is_this_obs, 'i']
#             airmass = df.loc[is_this_obs, 'Airmass']
#             offset = inst_mag - mean_z - r_catmag - transform * (r_catmag - i_catmag) \
#                 - extinction * airmass - image_effect
#             offsets.append(offset.iloc[0])
#         this_dict = {'CompID': comp, 'Offsets': offsets, 'MeanOffset': mean(offsets)}
#         dict_list.append(this_dict)
#     df_comp_diagnostics = pd.DataFrame(data=dict_list)
#     df_comp_diagnostics.index = list(df_comp_diagnostics['CompID'])
#     return df_comp_diagnostics


_____ANCILLARY_CODE________________________________________________ = 0


def get_transform(filter='Clear', passband='r'):
    """ Get transform (filter=Clear, passband=SloanR, color=(SloanR-SloanI) from one directory's df_obs.
        Must have 2 or more images in chosen filter (usually moot, as we will have the whole night's set).
        First, user must ensure that current working directory is correct (prob. by running resume().
        Color index hard-coded as Sloan r - Sloan i.
    :param filter: filter in which images were taken (to select from df_obs). [string]
    :param passband: passband (e.g., Johnson-Cousins 'R' or Sloan 'i') to target. [string]
    :return: dataframe, each row one image, columns=T, dT (transform & its uncertainty). [pandas DataFrame]
    USAGE: fit = get_transform()
    """
    df_obs_all = read_df_obs_all()
    df_comps = read_df_comps_all()
    df_images = read_df_images_all()
    df_merged = pd.merge(left=df_obs_all, right=df_comps,
                         how='left', left_on='SourceID', right_on='CompID', sort=False)
    df_merged = pd.merge(left=df_merged, right=df_images,
                         how='left', on='FITSfile', sort=False).copy()
    is_comp = pd.Series([t.lower() == 'comp' for t in df_merged['Type']])
    is_filter = pd.Series([f.lower() == filter.lower() for f in df_merged['Filter']])
    to_keep = is_comp & is_filter
    df = df_merged[list(to_keep)].copy()

    df['CI'] = df['r'] - df['i']
    df['Difference'] = df['InstMag'] - df[passband]
    n_images = len(df['FITSfile'].drop_duplicates())
    if n_images < 2:
        print(' >>>>> ERROR: get_transform() must get more than one image in filter \'' + filter + '\'.')
        return None
    else:
        print('MixedModel (' + str(n_images) + ' images):')
        # fit = MixedModelFit(data=df, dep_var='Difference', fixed_vars=['CI', 'CI2'], group_var='FITSfile')
        fit = MixedModelFit(data=df, dep_var='Difference', fixed_vars=['CI'], group_var='FITSfile')
        print('Transform(' + passband + '->' + filter + ') =',
              str(fit.df_fixed_effects.loc['CI', 'Value']),
              'stdev =', str(fit.df_fixed_effects.loc['CI', 'Stdev']))
    return fit


def mp_phot_test():
    from random import random, gauss, uniform, seed
    _, _, _, _ = get_context()   # forces exit if working directory is invalid.
    state = get_session_state()  # for extinction and transform values.
    seed(3423)
    n_comps = 40
    df_comps = read_df_comps_all()[0:n_comps]
    df_comps['Vignette'] = [uniform(-1, 1) for i in range(n_comps)]
    df_comps['X1024'] = [uniform(-1, 1) for i in range(n_comps)]
    df_comps['Y1024'] = [uniform(-1, 1) for i in range(n_comps)]
    df_comps['CatError'] = [0.2, -0.15] + [gauss(0, 0.01) for i in range(n_comps - 2)]
    n_images = 20
    df_images = pd.DataFrame(
        {'Serial': [1 + i for i in range(n_images)],
         'FITSfile': ['a' + str(i+1) + '.fts' for i in range(n_images)],
         'Airmass': [1.2 + 0.01 * i for i in range(n_images)],
         'JD_fract': [.6 + 0.005 * i for i in range(n_images)],
         'Cirrus': [gauss(0, 0.05) for i in range(n_images)]})
    z = -21.000
    extinction = state['extinction']['Clear']
    transform = TRANSFORM_CLEAR_SR_SR_SI  # overrides value read from state.
    dict_obs_list = []
    for i_comp in df_comps.index:
        for i_image in df_images.index:
            dict_obs = {'FITSfile': df_images.loc[i_image, 'FITSfile'],
                        'SourceID': df_comps.loc[i_comp, 'CompID'],
                        'Type': 'Comp',
                        'Airmass': df_images.loc[i_image, 'Airmass'],
                        'JD_fract': df_images.loc[i_image, 'JD_fract'],
                        'Vignette': df_comps.loc[i_comp, 'Vignette'],
                        'X1024': df_comps.loc[i_comp, 'X1024'],
                        'Y1024': df_comps.loc[i_comp, 'Y1024'],
                        'CI': df_comps.loc[i_comp, 'r'] - df_comps.loc[i_comp, 'i'],
                        'R': df_comps.loc[i_comp, 'R'],
                        'InstMag': z + df_comps.loc[i_comp, 'R']
                                + extinction * df_images.loc[i_image, 'Airmass']
                                + transform * (df_comps.loc[i_comp, 'r'] - df_comps.loc[i_comp, 'i'])
                                + 0.04 * df_comps.loc[i_comp, 'Vignette']
                                + df_images.loc[i_image, 'Cirrus']
                                - df_comps['CatError']
                                + gauss(0, 0.002)}
            dict_obs_list.append(dict_obs)
    df_obs = pd.DataFrame(data=dict_obs_list)
    df_comps = df_comps.drop(['Vignette', 'X1024', 'Y1024'], axis=1)

    smodel = SessionModel(df_obs, df_comps, state)


# ----- Ultimately, the code modeled in ml_2_groups won't do.
# Statsmodels "variance components" are just variance pools,
#    and effects per group (per-image, or per-comp) CANNOT be extracted, just don't exist.
# Choices, then are: (1) settle for one random effect (prob. image), or
#    (2) use R+lmer via rpy2 or maybe pymer4 package. Groan.
# def ml_2_groups():
#     # This converges to right values, but fails to yield random-effect values. Sigh.
#     from random import seed, randint, random, shuffle
#     import statsmodels.api as sm
#     # Make data:
#     n = 100
#     n1 = int(n/2)
#     n2 = int(n) - n1
#     # seed(2465)
#     a = pd.Series([i + 0.1 * random() for i in range(n)])      # indep fixed var
#     b = pd.Series([0.5 * random() for i in range(n)])   # "
#     gp1 = n1 * ['gp1_a'] + n2 * ['gp1_b']
#     gp2 = n1 * ['gp2_a'] + n2 * ['gp2_b']
#     shuffle(gp1)  # works in place
#     shuffle(gp2)  # "
#     val_gp1 = pd.Series([-2 if g.endswith('a') else 2 for g in gp1])
#     val_gp2 = pd.Series([-0.5 if g.endswith('a') else 0.5 for g in gp2])
#     y_random_error = pd.Series([0.1 * random() for i in range(n)])
#     intercept = pd.Series(n * [123.000])
#     y = intercept + a + b + val_gp1 + val_gp2 + y_random_error
#     df_x = pd.DataFrame({'Y': y, 'A': a, 'B': b, 'GP1': gp1, 'GP2': gp2})
#     # df_x['Intercept'] = 1
#     df_x['group'] = 'No group'  # because statsmodels cannot handle >1 group, but uses variance components.
#
#     # Make model with 2 crossed random variables:
#     model_formula = 'Y ~ A + B'
#     # variance_component_formula = {'Group1': '0 + C(GP1)',
#     #                               'Group2': '0 + C(GP2)'}
#     variance_component_formula = {'Group1': '0 + GP1',
#                                   'Group2': '0 + GP2'}
#     random_effect_formula = '0'
#     model = sm.MixedLM.from_formula(model_formula,
#                                     data=df_x,
#                                     groups='group',
#                                     vc_formula=variance_component_formula,
#                                     re_formula=random_effect_formula)
#     result = model.fit()
#     print(result.summary())
#     return result


# CATALOG_TESTS________________________________________________ = 0

# def get_transforms_landolt_r_mags(fits_directory):
# # This as comparison catalog (to check ATLAS refcat2) gives unreasonably high errors,
# #      wheres APASS10 works just fine. Hmm.
#     from photrix.fov import Fov
#     fits_filenames = get_fits_filenames(fits_directory)
#     g_mags, R_mags, i_mags, landolt_r_mags = [], [], [], []
#     for fits_filename in fits_filenames:
#         fits_object = FITS(fits_directory, '', fits_filename)
#         if fits_object.filter == 'R':
#             df_refcat2 = get_refcat2_from_fits_object(fits_object)
#             fov_name = fits_object.object
#             fov_object = Fov(fov_name)
#             for star in fov_object.aavso_stars:

# #################################################################################################
# Keep the following 2 (commented out) "Canopus plots" for full-screen plots as demos for SAS talk.

# # "CANOPUS plot" (comps only, one point per obs:
# #     x=catalog r mag, y=obs InstMag(r) adjusted for extinction and transform):
# ax = axes[0, 0]
# make_labels_9_subplots(ax, 'Adjusted CANOPUS (all images)',
#                        'Catalog Mag (r)', 'Image-adjusted InstMag (r)', zero_line=False)
# df_canopus = df_plot_comp_obs.loc[:,
#              ['SourceID', 'Airmass', 'r', 'i', 'FITSfile', 'JD_fract', 'InstMag']]
# df_canopus['CI'] = df_canopus['r'] - df_canopus['i']
# df_canopus = pd.merge(df_canopus, df_image_effect, how='left', on='FITSfile', sort=False)
# extinction_adjustments = extinction * df_canopus['Airmass']
# transform_adjustments = transform * df_canopus['CI']
# image_adjustments = df_canopus['ImageEffect']
# jd_adjustments = jd_coefficient * df_canopus['JD_fract']
# sum_adjustments = extinction_adjustments + transform_adjustments + image_adjustments + jd_adjustments
# adjusted_instmags = df_canopus['InstMag'] - sum_adjustments
# df_canopus['AdjInstMag'] = adjusted_instmags
# # ax.scatter(x=df_canopus['r'], y=adjusted_instmags, alpha=0.6, color='darkblue')
# ax.scatter(x=df_canopus['r'], y=adjusted_instmags, alpha=0.6, color=comp_color)
# # first_comp_id = df_canopus.iloc[0, 0]
# # df_first_comp = df_canopus.loc[df_canopus['SourceID'] == first_comp_id, :]
# draw_x_line(ax, user_selections['min_r_mag'])
# draw_x_line(ax, user_selections['min_r_mag'])
#
# # "CANOPUS plot" (comps only, one point per obs:
# #     x=catalog r mag adjusted for extinction and transform, y=obs InstMag(r)):
# ax = axes[0, 1]
# make_labels_9_subplots(ax, 'Adjusted CANOPUS DIFF plot (all images)',
#                        'Catalog Mag (r)', 'Adjusted InstMag - r(cat)', zero_line=False)
# # Using data from previous plot:
# ax.scatter(x=df_canopus['r'], y=(adjusted_instmags - df_canopus['r']), alpha=0.6, color=comp_color)
# # ax.scatter(x=df_canopus['r'], y=adjusted_instmags, alpha=0.6, color='darkblue')
# draw_x_line(ax, user_selections['min_r_mag'])
# draw_x_line(ax, user_selections['min_r_mag'])
# #################################################################################################

# For spanning one subplot across more than one subplot tile:
# gs = axes[1, 0].get_gridspec()
# for row in [1, 2]:
#     for col in range(3):
#         axes[row, col].remove()
# axbig = fig.add_subplot(gs[1:, 1:])
# axbig.set_title('MP Target Lightcurve', loc='center', pad=-3)  # pad in points
# ax.set_xlabel(xlabel_jd, labelpad=-29)  # labelpad in points
# ax.set_ylabel('Mag (r)', labelpad=-5)  # "
