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

ARROW_CHARACTER = '\U00002794'

DAYS_PER_YEAR_NOMINAL = 365.25
VALID_FITS_FILE_EXTENSIONS = ['.fits', '.fit', '.fts']
DEGREES_PER_RADIAN = 180.0 / pi

# To assess FITS files in assess():
MP_PHOTOMETRY_FILTER = 'Clear'
MIN_FWHM = 1.5  # in pixels.
MAX_FWHM = 14  # "
FOCUS_LENGTH_MAX_PCT_DEVIATION = 3.0

MARGIN_RA_ZERO = 5  # in degrees, to handle RA ~zero; s/be well larger than images' height or width.

# For color handling:
FILTERS_FOR_MP_COLOR_INDEX = ('R', 'I')
COLOR_INDEX_PASSBANDS = ('r', 'i')
DEFAULT_MP_RI_COLOR = 0.2  # close to known Sloan mean (r-i) for MPs.
MAX_COLOR_COMP_UNCERT = 0.0125  # mag
MAX_COLOR_MAG_UNCERT = 0.10  # mag
COLOR_PLOT_FILENAME = 'Image_ColorIndex.png'

# To screen observations:
MAX_MAG_UNCERT = 0.03  # min signal/noise for COMP obs (as InstMagSigma).
# The next two probably should go in Instrument class, but ok for now.
ADU_SATURATED = 56000  # Max ADU allowable in original (Ur) images.
VIGNETTING = (1846, 0.62)  # (px from center, max fract of ADU_SATURATED allowed) both at corner.

# For this package:
MP_TOP_DIRECTORY = 'C:/Astro/MP Photometry/'
LOG_FILENAME = 'mp_photometry.log'
CONTROL_FILENAME = 'control.txt'
TRANSFORM_CONTROL_FILENAME = 'control_transform.txt'
COLOR_INDEX_CONTROL_FILENAME = 'control_color_index.txt'
DF_OBS_ALL_FILENAME = 'df_obs_all.csv'
DF_IMAGES_ALL_FILENAME = 'df_images_all.csv'
DF_COMPS_ALL_FILENAME = 'df_comps_all.csv'
TRANSFORM_CLEAR_SR_SR_SI = 0.025  # estimate from MP 1074 20191109 (37 images).
DEFAULT_MODEL_OPTIONS = {'fit_transform': False, 'fit_extinction': False,
                         'fit_vignette': True, 'fit_xy': False,
                         'fit_jd': True}  # defaults if not found in control file.

# For selecting comps within Refcat2 object (intentionally wide; can narrow with a control file later):
MIN_R_MAG = 10
MAX_R_MAG = 16
MAX_G_UNCERT = 20  # millimagnitudes
MAX_R_UNCERT = 20  # "
MAX_I_UNCERT = 20  # "
MIN_SLOAN_RI_COLOR = -0.4
MAX_SLOAN_RI_COLOR = 0.8

DO_PHOT_COMP_SELECTION_DEFAULTS = {
    'min_catalog_r_mag': 10,
    'max_catalog_r_mag': 16,
    'max_catalog_dr_mag': 20,
    'min_catalog_ri_color': 0.0,
    'max_catalog_ri_color': 0.4
}

TRANSFORM_COMP_SELECTION_DEFAULTS = {
    'min_catalog_r_mag': 10,
    'max_catalog_r_mag': 16,
    'max_catalog_dr_mag': 20,
    'min_catalog_ri_color': -0.4,
    'max_catalog_ri_color': 0.8
}

COLOR_COMP_SELECTION_DEFAULTS = {
    'min_catalog_r_mag': 10,
    'max_catalog_r_mag': 16,
    'max_catalog_dr_mag': 20,
    'min_catalog_ri_color': -0.4,
    'max_catalog_ri_color': 0.8
}

# For ALCDEF File generation:
DSW_SITE_DATA = {'longitude': -105.6536, 'latitude': +35.3311,
                 'facility': 'Deep Sky West Observatory',
                 'mpccode': 'V28'}
ALCDEF_DATA = {'contactname': 'Eric V. Dose',
               'contactinfo': 'MP@ericdose.com',
               'observers': 'Dose, E.V.',
               'filter': 'C',
               'magband': 'SR'}


_____EARLY_WORKFLOW_ATLAS_BASED________________________________________________ = 0

"""  ***************************************************************************
     WORKFLOW STEPS (example lines):
     * Ensure at least 1 file IN Clear filter.
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


def assess():
    """  First, verify that all required files are in the working directory or otherwise accessible.
         Then, perform checks on FITS files in this directory before performing the photometry proper.
         Modeled after and extended from assess() found in variable-star photometry package 'photrix'.
                                    May be zero for MP color index determination only. [int]
    :return: [None]
    """
    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
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
        print('Next: (1) enter MP pixel positions in', CONTROL_FILENAME,
              'AND SAVE it,\n      (2) make_dfs()')
        log_file.write('assess(): ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.' + '\n')
    else:
        print('\n >>>>> ' + str(n_warnings) + ' warnings (see listing above).')
        print('        Correct these and rerun assess() until no warnings remain.')
        log_file.write('assess(): ' + str(n_warnings) + ' warnings.' + '\n')

    write_control_txt_stub(this_directory, log_file, df)        # if it doesn't already exist.
    write_transform_control_txt_stub(this_directory, log_file)  # "
    write_color_control_txt_stub(this_directory, log_file)      # "
    log_file.close()


def make_dfs():
    """ For one MP on one night: gather images and ATLAS refcat2 catalog data, make df_comps and df_obs.
    :return: [None]
    USAGE: make_dfs()
    """
    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
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

    # Read user's MP *pixel* location in each of 2 images, from control file:
    mp_location_filenames, x_pixels, y_pixels = read_mp_locations()
    if mp_location_filenames is None:
        print(' >>>>> ' + CONTROL_FILENAME + ': something wrong with #MP lines. Stopping.')
        log_file.write(' >>>>> ' + CONTROL_FILENAME + ': something wrong with #MP lines. Stopping.\n')
        exit(1)

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
    print('\nNext: (1) enter comp selection limits and model options in ' + CONTROL_FILENAME,
          '\n      (2) run do_mp_phot()')


_____TRANSFORM_MEASUREMENT_______________________________________________ = 0


# def get_transform(filter='Clear', passband='r'):
#     """ Get transform (filter=Clear, passband=SloanR, color=(SloanR-SloanI) from one directory's df_obs.
#         Must have 2 or more images in chosen filter (usually moot, as we will have the whole night's set).
#         First, user must ensure that current working directory is correct (prob. by running resume().
#         Color index hard-coded as Sloan r - Sloan i.
#     :param filter: filter in which images were taken (to select from df_obs). [string]
#     :param passband: passband (e.g., Johnson-Cousins 'R' or Sloan 'i') to target. [string]
#     :return: dataframe, each row one image, columns=T,
#              dT (transform & its uncertainty). [pandas DataFrame]
#     USAGE: fit = get_transform()
#     """
#     df_obs_all = read_df_obs_all()
#     df_comps = read_df_comps_all()
#     df_images = read_df_images_all()
#     df_merged = pd.merge(left=df_obs_all, right=df_comps,
#                          how='left', left_on='SourceID', right_on='CompID', sort=False)
#     df_merged = pd.merge(left=df_merged, right=df_images,
#                          how='left', on='FITSfile', sort=False).copy()
#     is_comp = pd.Series([t.lower() == 'comp' for t in df_merged['Type']])
#     is_filter = pd.Series([f.lower() == filter.lower() for f in df_merged['Filter']])
#     to_keep = is_comp & is_filter
#     df = df_merged[list(to_keep)].copy()
#
#     df['CI'] = df['r'] - df['i']
#     df['Difference'] = df['InstMag'] - df[passband]
#     n_images = len(df['FITSfile'].drop_duplicates())
#     if n_images < 2:
#         print(' >>>>> ERROR: get_transform() must get more than one image in filter \'' + filter + '\'.')
#         return None
#     else:
#         print('MixedModel (' + str(n_images) + ' images):')
#         # fit = MixedModelFit(data=df, dep_var='Difference',
#         fixed_vars=['CI', 'CI2'], group_var='FITSfile')
#         fit = MixedModelFit(data=df, dep_var='Difference', fixed_vars=['CI'], group_var='FITSfile')
#         print('Transform(' + passband + '->' + filter + ') =',
#               str(fit.df_fixed_effects.loc['CI', 'Value']),
#               'stdev =', str(fit.df_fixed_effects.loc['CI', 'Stdev']))
#     return fit

def calc_transform(f='Clear', pbf='r', pb1='r', pb2='i'):
    """ From one image in filter f, get transform for filter f-->catalog passband pbf,
            subject to color index=cat_pb1 - cat_pb2.
        Must have already run start() or resume() to set working directory.
        Must have already run through make_dfs() for this directory.
        Will use all FITS file images taken in filter
    :param f: name of filter in which FITS file image was taken, and defining the transform. [string]
    :param pbf: name of catalog passband to which to transform magnitudes (associated with filter). [string]
    :param pb1: first catalog passband of color index. [string]
    :param pb2: second catalog passband of color index. [string]
    :return: None. Writes results to console and summary to log file.
    """
    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    state = get_session_state()  # for extinction and transform values.

    # Set up required data:
    df_transform = make_df_all(filters_to_include=f, comps_only=True, require_mp_obs_each_image=False)
    user_selections = read_selection_criteria(TRANSFORM_CONTROL_FILENAME, TRANSFORM_COMP_SELECTION_DEFAULTS)
    apply_calc_transform_selections(df_transform, user_selections)  # adds boolean column 'UseInModel'.
    options_dict = read_regression_options(TRANSFORM_CONTROL_FILENAME)

    # Perform mixed-model regression:
    model = TransformModel(df_transform, f, pbf, pb1, pb2, state, options_dict)

    # Make diagnostic plots:
    make_transform_diagnostic_plots(model, df_transform, f, pbf, pb1, pb2, state, user_selections)

    # Write results to log file and to console:
    print('Transform=', '{0:.4f}'.format(model.mm_fit.df_fixed_effects.loc['CI', 'Value']),
          '  tr_sigma=', '{0:.4f}'.format(model.mm_fit.df_fixed_effects.loc['CI', 'Stdev']),
          '    mag_sigma=', '{0:.1f}'.format(1000.0 * model.mm_fit.sigma), 'mMag.')


class TransformModel:
    def __init__(self, df_transform, f, pbf, pb1, pb2, state, options_dict):
        """  Makes and holds color-transform model via mixed-model regression.
             Requires data from at least 3 images in f.
        :param df_transform: table of data from which to draw data for regression. [pandas Dataframe]
        :param f: name of filter in which FITS file image was taken, and defining the transform. [string]
        :param pbf: name of catalog passband to which to transform mags (associated with filter). [string]
        :param pb1: first catalog passband of color index. [string]
        :param pb2: second catalog passband of color index. [string]
        :param state:
        :param options_dict: holds options for making comp fit. [pandas dict object]
        """
        self.df_used = df_transform.copy().loc[df_transform['UseInModel'], :]  # only obs used in model.
        n_images = len(self.df_used['FITSfile'].drop_duplicates())
        self.enough_images = (n_images >= 3)
        if not self.enough_images:
            return
        self.f = f
        self.pbf, self.pb1, self.pb2 = pbf, pb1, pb2
        self.state = state
        self.fit_extinction = options_dict.get('fit_extinction', False)
        self.fit_vignette = options_dict.get('fit_vignette', True)
        self.fit_xy = options_dict.get('fit_xy', False)
        self.fit_jd = options_dict.get('fit_jd', True)

        self.dep_var_name = 'InstMag_with_offsets'
        self.mm_fit = None      # placeholder for the fit result [photrix MixedModelFit object].
        self.extinction = None  # "
        self.vignette = None    # "
        self.x = None           # "
        self.y = None           # "
        self.jd1 = None         # "
        self.transform = None
        self.transform_sigma = None

        self._prep_and_do_regression()

    def _prep_and_do_regression(self):
        """ Using photrix.util.MixedModelFit class (which wraps statsmodels.MixedLM.from_formula() etc).
            This function uses comp data only (no minor planet data).
        :return: [None] Puts model into self.mm_fit.
        """
        if not self.enough_images:
            return
        self.df_used['CI'] = self.df_used[self.pb1] - self.df_used[self.pb2]

        # Initiate dependent-variable offset, which will aggregate all such offset terms:
        dep_var_offset = self.df_used[self.pbf].copy()  # *copy* CatMag, or it will be damaged

        # Build fixed-effect (x) variable list and construct dep-var offset:
        fixed_effect_var_list = ['CI']
        if self.fit_extinction:
            fixed_effect_var_list.append('Airmass')
        else:
            extinction = self.state['extinction']['Clear']
            dep_var_offset += extinction * self.df_used['Airmass']
            print(' Extinction (Airmass) not fit: value fixed at',
                  '{0:.3f}'.format(extinction))
        if self.fit_vignette:
            fixed_effect_var_list.append('Vignette')
        if self.fit_xy:
            fixed_effect_var_list.extend(['X1024', 'Y1024'])
        if self.fit_jd:
            fixed_effect_var_list.append('JD_fract')

        # Build 'random-effect' variable:
        random_effect_var_name = 'FITSfile'  # cirrus effect is per-image

        # Build dependent (y) variable:
        self.df_used[self.dep_var_name] = self.df_used['InstMag'] - dep_var_offset

        # Execute regression:
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
        self.mm_fit = MixedModelFit(data=self.df_used,
                                    dep_var=self.dep_var_name,
                                    fixed_vars=fixed_effect_var_list,
                                    group_var=random_effect_var_name)
        if not self.mm_fit.converged:
            print(' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.')
            print(self.mm_fit.statsmodels_object.summary())
            print(' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.')
        else:
            # TODO: fix these
            self.transform = 9999.
            self.transform_sigma = 9999.


def make_transform_diagnostic_plots(model, df_model, f, pbf, pb1, pb2, state, user_selections):
    """  Display and write to file several diagnostic plots, to help decide which obs, comps, images
         might need removal by editing control file.
    :param model: mixed model summary object. [photrix.MixedModelFit object]
    :param df_model: dataframe of all data including UseInModel (user selection) column. [pandas DataFrame]
    :param f: name of filter in which observations were made. [string]
    :param pbf: name of target passband (associated with filter) in which mags to be reported. [string]
    :param pb1: name of first of two catalog passbands in color index. [string]
    :param pb2: name of second of two catalog passbands in color index. [string]
    :param state: session state for this observing session [dict]
    :param user_selections: comp selection criteria, used for drawing limits on plots [python dict]
    :return: [None] Writes image files e.g., Transform_Image1_QQ_comps.png
    """
    this_directory, mp_string, an_string = get_context()

    # Delete any previous transform image files from current directory:
    image_filenames = [fn for fn in os.listdir('.')
                       if fn.startswith('Transform_Image') and fn.endswith('.png')]
    for fn in image_filenames:
        os.remove(fn)

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
    # if 'CI' in model.mm_fit.df_fixed_effects.index:
    #     transform = model.mm_fit.df_fixed_effects.loc['CI', 'Value']  # if fit in model
    # else:
    #     transform = TRANSFORM_CLEAR_SR_SR_SI  # default if not fit in model (normal case)
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
    transform_string = f + ARROW_CHARACTER + pbf + ' (' + pb1 + '-' + pb2 + ')'
    print(transform_string)

    # ################ FIGURE 1: Q-Q plot of mean comp effects (one point per comp star used in model),
    #    code heavily adapted from photrix.process.SkyModel.plots():
    window_title = 'Transform Q-Q Plot (by comp):  MP ' + mp_string + '   AN ' + an_string
    page_title = 'MP ' + mp_string + '   AN ' + an_string + \
                 '   ::   Transform Q-Q plot by comp ' + transform_string
    plot_annotation = str(n_comps) + ' comps used in model.\n(tags: comp SourceID)'
    df_y = df_plot_comp_obs.loc[:, ['SourceID', 'Residual']].groupby(['SourceID']).mean()
    df_y = df_y.sort_values(by='Residual')
    y_data = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y.index.values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
                          'Transform_Image1_QQ_comps.png')

    # ################ FIGURE 2: Q-Q plot of comp residuals (one point per comp obs),
    #    code heavily adapted from photrix.process.SkyModel.plots():
    window_title = 'Transform Q-Q Plot (by comp observation):  MP ' + mp_string + '   AN ' + an_string
    page_title = 'MP ' + mp_string + '   AN ' + an_string + '   ::   Transform Q-Q plot by obs ' +\
                 transform_string
    plot_annotation = str(len(df_plot_comp_obs)) + ' observations of ' + \
        str(n_comps) + ' comps used in model.\n (tags: observation Serial numbers)'
    df_y = df_plot_comp_obs.loc[:, ['Serial', 'Residual']]
    df_y = df_y.sort_values(by='Residual')
    y_data = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y['Serial'].values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
                          'Transform_Image2_QQ_obs.png')


_____COLOR_INDEX_PHOTOMETRY________________________________________________ = 0


def do_color(filters=FILTERS_FOR_MP_COLOR_INDEX, passbands=COLOR_INDEX_PASSBANDS):
    """ Use comp-star and MP instrument mags, and comp catalog magnitudes to estimate
        MP color index in Sloan r-i *ONLY* (for now).
    :param filters: specify the two filters for which images are to be used, e.g., ('R', 'I') to
        use Johnson R and I images (the normal case). [2-tuple of strings]
    :param passbands: specify the two passbands defining the color index, e.g., ('r', 'i') to
        define color in Sloan r and i (the normal case). In any case, the two passbands must be
        represented in the ALTAS refcat2 catalog and must be included
        in the local catalog object. [2-tuple of strings]
    :return: None. Writes to screen and to log file, only.
    """
    pa, pb = passbands  # typically Sloan 'r' and 'i'.
    passband_string = pa + '-' + pb

    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n===== do_color()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Load required data:
    df_obs_all = read_df_obs_all()
    df_comps_all = read_df_comps_all()
    df_images_all = read_df_images_all()
    state = get_session_state()  # for extinction and transform values.

    # Identify files with correct filters:
    fa, fb = filters
    filter_string = fa + '-' + fb
    is_fa_image = (df_images_all['Filter'] == fa)
    is_fb_image = (df_images_all['Filter'] == fb)
    both_filters_available = any(is_fa_image) and any(is_fb_image)
    if not both_filters_available:
        print(' >>>>> ERROR: images missing for one or both of filters: \'' + fa + '\', \'' + fb + '\'')
        return None
    fa_filenames = list(df_images_all.loc[is_fa_image, 'FITSfile'])
    fb_filenames = list(df_images_all.loc[is_fb_image, 'FITSfile'])
    ci_filenames = fa_filenames + fb_filenames

    # Keep only images with valid MP observation:
    valid_mp_filenames, mp_filters = [], []
    for filename in ci_filenames:
        is_filename_mp_row = (df_obs_all['FITSfile'] == filename) & (df_obs_all['Type'] == 'MP')
        if sum(is_filename_mp_row) == 1:
            mp_instmag = df_obs_all.loc[is_filename_mp_row, 'InstMag'].iloc[0]
            mp_instmagsigma = df_obs_all.loc[is_filename_mp_row, 'InstMagSigma'].iloc[0]
            if (mp_instmag is not None) and (mp_instmagsigma is not None):
                if (not np.isnan(mp_instmag)) and (not np.isnan(mp_instmagsigma)):
                    if mp_instmagsigma < MAX_COLOR_MAG_UNCERT:
                        valid_mp_filenames.append(filename)
                        if filename in fa_filenames:
                            mp_filters.append(fa)
                        if filename in fb_filenames:
                            mp_filters.append(fb)
    ci_filenames = valid_mp_filenames.copy()
    if not (fa in mp_filters and fb in mp_filters):
        print(' >>>>> ERROR: after screening for MP observations,'
              'images missing for one or both of filters: \'' + fa + '\', \'' + fb + '\'')
        return None

    # Make list of all comps in color-index images, and having low enough catalog uncertainty:
    is_ci_obs = [(f in ci_filenames) for f in df_obs_all['FITSfile']]
    df_ci_obs = df_obs_all[is_ci_obs].copy()  # includes MP observations.
    is_ci_comp_obs = [type == 'Comp' for type in df_ci_obs['Type']]
    ci_comp_ids = df_ci_obs.loc[is_ci_comp_obs, 'SourceID'].drop_duplicates()
    is_high_snr = [(df_comps_all.loc[c, 'dr'] < 1000 * MAX_COLOR_COMP_UNCERT) and
                   (df_comps_all.loc[c, 'di'] < 1000 * MAX_COLOR_COMP_UNCERT)
                   for c in ci_comp_ids]
    ci_comp_ids = ci_comp_ids[is_high_snr].copy()

    # Keep only comps which are represented in every color-index image:
    ci_comp_list = []
    n_ci_images = len(fa_filenames) + len(fb_filenames)
    for this_comp_id in ci_comp_ids:
        n_images_this_comp = sum([id == this_comp_id for id in df_ci_obs['SourceID']])
        if n_images_this_comp == n_ci_images:
            ci_comp_list.append(this_comp_id)
    is_comp_in_list = [(id in ci_comp_list) for id in df_ci_obs['SourceID']]
    df_ci_obs = df_ci_obs[is_comp_in_list].copy()

    # Make one merged dataframe, one row per eligible comp observation, but only selected rows:
    df_fit_obs = pd.merge(left=df_ci_obs.loc[:, ['FITSfile', 'SourceID', 'Type', 'InstMag']],
                          right=df_comps_all.loc[:, ['CompID', pa, pb]],
                          how='left', left_on='SourceID', right_on='CompID', sort=False)
    df_fit_obs = pd.merge(left=df_fit_obs, right=df_images_all.loc[:, ['FITSfile', 'Filter']],
                          how='left', on='FITSfile', sort=False)
    df_fit_obs = df_fit_obs.sort_values(by=['SourceID', 'Filter', 'FITSfile'])  # to ease any debugging.

    # Remove rows for comps eliminated by #COLOR_COMP directive in color control file:
    user_selections = read_selection_criteria(COLOR_INDEX_CONTROL_FILENAME, COLOR_COMP_SELECTION_DEFAULTS)
    is_comp = (df_fit_obs['Type'] == 'Comp')
    deselect_for_comp_id = df_fit_obs['SourceID'].isin(user_selections['color_comps']) & is_comp
    obs_to_keep = [not d for d in deselect_for_comp_id]
    df_fit_obs = df_fit_obs.loc[obs_to_keep, :]

    # Make variables for regression fit:
    fit_dict_list = []
    is_fa_row = (df_fit_obs['Filter'] == fa)
    is_fb_row = (df_fit_obs['Filter'] == fb)
    for this_comp_id in ci_comp_list:
        fit_dict = dict()
        fit_dict['comp_id'] = this_comp_id
        fit_dict['CI'] = df_comps_all.loc[this_comp_id, pa] - df_comps_all.loc[this_comp_id, pb]

        is_comp_id = (df_fit_obs['SourceID'] == this_comp_id)
        is_comp_id_fa = is_comp_id & is_fa_row
        is_comp_id_fb = is_comp_id & is_fb_row
        # TODO probably should include extinction/airmass effects here too, since images may differ (sigh).
        comp_id_mean_fa_instmag = df_fit_obs.loc[is_comp_id_fa, 'InstMag'].mean()  # mean over images.
        comp_id_mean_fb_instmag = df_fit_obs.loc[is_comp_id_fb, 'InstMag'].mean()  # "
        fit_dict['InstMagDiff'] = comp_id_mean_fa_instmag - comp_id_mean_fb_instmag
        fit_dict_list.append(fit_dict)
    df_fit = pd.DataFrame(data=fit_dict_list)
    df_fit.index = list(df_fit['comp_id'])

    # Perform regression fit:
    result = smf.ols(formula='CI ~ InstMagDiff', data=df_fit).fit()

    # Build MP input data for color-index (Sloan r-i) prediction:
    instmag_list_fa, instmag_list_fb, sigma_list_fa, sigma_list_fb = [], [], [], []
    mp_dict = dict()
    for filename in df_fit_obs['FITSfile'].drop_duplicates():
        mp_instmag = df_obs_all.loc[(df_obs_all['FITSfile'] == filename) &
                                    (df_obs_all['Type'] == 'MP'), 'InstMag'].iloc[0]
        mp_sigma = df_obs_all.loc[(df_obs_all['FITSfile'] == filename) &
                                  (df_obs_all['Type'] == 'MP'), 'InstMagSigma'].iloc[0]
        if df_images_all.loc[filename, 'Filter'] == fa:
            instmag_list_fa.append(mp_instmag)
            sigma_list_fa.append(mp_sigma)
        if df_images_all.loc[filename, 'Filter'] == fb:
            instmag_list_fb.append(mp_instmag)
            sigma_list_fb.append(mp_sigma)
    mp_dict['InstMagDiff'] = sum(instmag_list_fa) / len(instmag_list_fa) -\
        sum(instmag_list_fb) / len(instmag_list_fb)
    df_prediction = pd.DataFrame(data=mp_dict, index=range(len(mp_dict)))

    # Predict MP color index and its uncertainty, then return them:
    mp_color = result.predict(df_prediction).iloc[0]
    mag_diff = df_prediction['InstMagDiff'].iloc[0]
    sigma_intercept = result.bse.Intercept
    sigma_slope = result.bse.InstMagDiff
    mean_diff = result.model.data.frame['InstMagDiff'].mean()
    variance_color = sigma_intercept ** 2 + sigma_slope ** 2 * (mag_diff - mean_diff) ** 2 + \
        sum([sigma ** 2 for sigma in sigma_list_fa]) / (len(sigma_list_fa) ** 2) + \
        sum([sigma ** 2 for sigma in sigma_list_fb]) / (len(sigma_list_fb) ** 2)
    sigma_color = sqrt(variance_color)
    n_ci_comps = len(ci_comp_list)

    # Generate plot (may as well do it here rather than in another fn, as all data is already gathered):
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(11, 8.5))  # (width, height) in "inches"
    ax = axes  # not subscripted if just one subplot in Figure
    page_title = 'MP ' + mp_string + '   AN ' + an_string + '   ::   Color Index regression'
    ax.set_title(page_title, color='darkblue', fontsize=20, pad=30)
    plot_annotation = '{0:.3f}'.format(mp_color) +\
        ' ' + u'\u00B1' + ' {0:.3f}'.format(sigma_color) + ' from ' + filter_string +\
        ' (' + str(n_ci_comps) + ' comps used)'
    fig.text(x=0.5, y=0.87, s=plot_annotation,
             verticalalignment='top', horizontalalignment='center', fontsize=12)
    x_values = [x[1] for x in result.model.exog]  # unpack from 2-d array.
    y_values = result.model.endog
    ax.set_xlabel('Diff(instrumental magnitudes): ' + filter_string)
    ax.set_ylabel('Color index: ' + passband_string)
    ax.grid(True, color='lightgray', zorder=-1000)
    ax.scatter(x=x_values, y=y_values, alpha=0.7, color='black', zorder=+1000)
    x_mp = [mag_diff]
    y_mp = [mp_color]
    ax.scatter(x=x_mp, y=y_mp, alpha=1, s=144, marker='X', color='orangered', zorder=+1001)

    # Label with comp ID: (1) outlier residuals, (2) lowest and highest inst. mag diffs:
    is_outlier = [abs(t) > 2.5 for t in result.resid_pearson]
    is_highest = [x == max(x_values) for x in x_values]
    is_lowest = [x == min(x_values) for x in x_values]
    to_label = [(o or h or l) for (o, h, l) in zip (is_outlier, is_highest, is_lowest)]
    labels = df_fit['comp_id']
    for x, y, label, add_label, t in zip(x_values, y_values, labels, to_label, result.resid_pearson):
        if add_label:
            if t > 0:
                ax.annotate(label, xy=(x, y), xytext=(-4, 4),
                            textcoords='offset points', ha='right', va='bottom', rotation=-40)
            else:
                ax.annotate(label, xy=(x, y), xytext=(4, -4),
                            textcoords='offset points', ha='left', va='top', rotation=-40)
    plt.show()
    fig.savefig(COLOR_PLOT_FILENAME)

    # Write final results to screen and to log:
    output = 'MP color index (' + passband_string + ') = ' + '{0:.3f}'.format(mp_color) +\
             ' ' + u'\u00B1' + ' {0:.3f}'.format(sigma_color) + ' from ' + filter_string +\
             '  (' + str(n_ci_comps) + ' comps used)'
    print(output + '\n   Write this to ' + CONTROL_FILENAME + ': #MP_RI_COLOR  ' +
          '{0:.3f}'.format(mp_color))
    log_file.write(output)
    log_file.close()


_____LIGHTCURVE_PHOTOMETRY________________________________________________ = 0


def do_mp_phot():
    """ Primary lightcurve photometry for one session. Takes all data incl. color index, generates:
    (1) canopus-import format results, (2) ALCDEF-format file, and (3) all diagnostic plots.
    Typically iterated, pruning comp-star ranges and outliers, until converged and then simply stop.
    USAGE: do_mp_phot()   [no return value]
    """
    context = get_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n===== do_mp_phot()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Load required data:
    df_all = make_df_all(filters_to_include='Clear', comps_only=False, require_mp_obs_each_image=True)
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
    # df_model['UseInModel'] = True
    # Note: df_model now contains all obs, comp, and image data, for both comp stars and minor planets.

    # Mark df_model with user selections, sync comp and image dfs:
    user_selections = read_selection_criteria(CONTROL_FILENAME, DO_PHOT_COMP_SELECTION_DEFAULTS)
    apply_do_phot_selections(df_model, user_selections)  # modifies in-place.

    # # Sync the comp and image dataframes (esp. user selections); may be needed later for plotting:
    # df_model_comps, df_model_images = sync_comps_and_images(df_model, df_comps_all, df_images_all)
    # print(str(len(df_model_comps)), 'comps retained in model.')

    # # Make data diagnostics for plotting:
    # df_model = make_diagnostics(df_model)
    #
    # # Do data-only (pre-model) plots; use them to update (in-place) 'UseInModel' column in the 3 dfs:
    # do_pre_model_plots(df_model, df_model_comps, df_model_images, user_selections, mp_color_ri)

    # Make photometric model via mixed-model regression, using only selected observations:
    options_dict = read_regression_options(CONTROL_FILENAME)
    model = SessionModel(df_model, mp_color_ri, state, options_dict)

    make_session_diagnostic_plots(model, df_model, mp_color_ri, state, user_selections)

    write_canopus_file(model)
    write_alcdef_file(model, mp_color_ri, source_string_ri)
    model_jds = df_model['JD_mid']
    print(' >>>>> Please add this line to MPfile', mp_string + ':',
          '  #OBS', '{0:.5f}'.format(model_jds.min()), ' {0:.5f}'.format(model_jds.max()),
          ' ;', an_string)


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
            print(' Transform (Color Index) not fit: value fixed at',
                  '{0:.3f}'.format(self.transform_fixed))
        if self.fit_extinction:
            fixed_effect_var_list.append('Airmass')
        else:
            extinction = self.state['extinction']['Clear']
            dep_var_offset += extinction * self.df_used_comps_only['Airmass']
            print(' Extinction (Airmass) not fit: value fixed at',
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
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
        self.mm_fit = MixedModelFit(data=self.df_used_comps_only,
                                    dep_var=self.dep_var_name,
                                    fixed_vars=fixed_effect_var_list,
                                    group_var=random_effect_var_name)
        print(self.mm_fit.statsmodels_object.summary())
        if not self.mm_fit.converged:
            # print(' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.')
            # print(self.mm_fit.statsmodels_object.summary())
            print(' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.')

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
    y_data = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y.index.values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
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
    y_data = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y['Serial'].values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
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
    make_9_subplot(ax, 'Catalog Mag Uncertainty (dr)', 'Catalog Mag (r)', 'mMag', '', False,
                   x_data=df_plot_comp_obs['r'], y_data=df_plot_comp_obs['dr'])

    # Catalog color plot (comps only, one point per comp, x=cat r mag, y=cat color (r-i)):
    ax = axes[0, 1]
    make_9_subplot(ax, 'Catalog Color Index', 'Catalog Mag (r)', 'CI Mag', '', zero_line=False,
                   x_data=df_plot_comp_obs['r'], y_data=(df_plot_comp_obs['r'] - df_plot_comp_obs['i']))
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=len(model.df_mp_mags) * [mp_color_ri],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)

    # Inst Mag plot (comps only, one point per obs, x=cat r mag, y=InstMagSigma):
    ax = axes[0, 2]
    make_9_subplot(ax, 'Instrument Magnitude Uncertainty', 'Catalog Mag (r)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['r'], y_data=df_plot_comp_obs['InstMagSigma'])
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=model.df_mp_mags['InstMagSigma'],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)

    # Cirrus plot (comps only, one point per image, x=JD_fract, y=Image Effect):
    ax = axes[1, 0]
    df_this_plot = pd.merge(df_image_effect, df_plot_comp_obs.loc[:, ['FITSfile', 'JD_fract']],
                            how='left', on='FITSfile', sort=False).drop_duplicates()
    make_9_subplot(ax, 'Image effect (cirrus plot)', xlabel_jd, 'mMag', '', False,
                   x_data=df_this_plot['JD_fract'], y_data=1000.0 * df_this_plot['ImageEffect'],
                   alpha=1.0, jd_locators=True)

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

    # Airmass plot (comps only, one point per obs; x=JD_fract, y=Airmass):
    ax = axes[2, 1]
    make_9_subplot(ax, 'Airmass vs time', xlabel_jd, 'Airmass', '', False,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=df_plot_comp_obs['Airmass'],
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
    make_9_subplot(ax, 'Model residual vs r (catalog)', 'Catalog Mag (r)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['r'], y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=len(model.df_mp_mags) * [0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)
    draw_x_line(ax, user_selections['min_r_mag'])
    draw_x_line(ax, user_selections['max_r_mag'])

    # Comp residual plot (comps only, one point per obs: x=raw Instrument Mag, y=model residual):
    ax = axes[0, 1]
    make_9_subplot(ax, 'Model residual vs raw Instrument Mag', 'Raw instrument mag', 'mMag', '', True,
                   x_data=df_plot_comp_obs['InstMag'], y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter(x=model.df_mp_mags['InstMag'], y=len(model.df_mp_mags) * [0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)

    # Comp residual plot (comps only, one point per obs: x=catalog r-i color, y=model residual):
    ax = axes[0, 2]
    make_9_subplot(ax, 'Model residual vs Color Index (cat)', 'Catalog Color (r-i)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['r'] - df_plot_comp_obs['i'],
                   y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter(x=[mp_color_ri], y=[0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)

    # Comp residual plot (comps only, one point per obs: x=Julian Date fraction, y=model residual):
    ax = axes[1, 0]
    make_9_subplot(ax, 'Model residual vs JD', xlabel_jd, 'mMag', '', True,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=1000.0 * df_plot_comp_obs['Residual'],
                   jd_locators=True)

    # Comp residual plot (comps only, one point per obs: x=Airmass, y=model residual):
    ax = axes[1, 1]
    make_9_subplot(ax, 'Model residual vs Airmass', 'Airmass', 'mMag', '', True,
                   x_data=df_plot_comp_obs['Airmass'], y_data=1000.0 * df_plot_comp_obs['Residual'])

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
                   y_data= 1000.0 * df_plot_comp_obs['Residual'])

    plt.show()
    fig.savefig('Image4_Residuals.png')

    # ################ FIGURE(S) 5: Variability plots:
    # Several comps on a subplot, vs JD, normalized by (minus) the mean of all other comps' responses.
    # Make df_offsets (one row per obs, at first with only raw offsets):
    make_comp_variability_plots(df_plot_comp_obs, xlabel_jd, transform, sigma)


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
    mpdir = os.path.join(MP_TOP_DIRECTORY, 'MP_' + str(mp))
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


def get_context():
    """ This is run at beginning of workflow functions (except start() or resume()) to orient the function.
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


def read_mp_locations():
    """ Reads control file, parses exactly 2 #MP lines, returns filenames, x- and y-pixels.
    :return: mp_location_filenames [list of 2 strings], x_pixels, y_pixels [each list of 2 floats].
    """
    mp_location_key = '#MP_LOCATION'
    mp_location_filenames, x_pixels, y_pixels = [], [], []
    with open(CONTROL_FILENAME, 'r') as cf:
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
    with open(CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines
        for line in lines:
            if line.upper().startswith('#MP_RI_COLOR '):
                tokens = line[len('#MP_RI_COLOR '):].split(maxsplit=1)
                if len(tokens) < 2:
                    print(' >>>>> ERROR: Either color or color source is missing from ' + CONTROL_FILENAME)
                else:
                    ri_string = tokens[0]
                    mp_ri_color_source = tokens[1].strip()
                    try:
                        mp_ri_color = float(ri_string)  # parse color value.
                    except ValueError:
                        print(' >>>>> ERROR: cannot read #MP_RI_COLOR given in ' + CONTROL_FILENAME)
                        mp_ri_color, mp_ri_color_source = None, \
                            '[error in reading control file ' + CONTROL_FILENAME + ']'
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

    color_comp_list = []
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
        for key in option_dict.keys():
            directive_key = '#' + key.upper()
            if content.startswith(directive_key):
                value = content[len(directive_key):].strip().lower()
                if value.startswith('y') or value == 'true':
                    option_dict[key] = True
                elif value.startswith('n') or value == 'false':
                    option_dict[key] = False
                else:
                    print(' >>>>> WARNING:', filename, directive_key, 'value not understood.')
    return option_dict


_____READING_DATA_FILES_____________________________________ = 0


def make_df_all(filters_to_include=MP_PHOTOMETRY_FILTER, comps_only=False, require_mp_obs_each_image=False):
    """ General, reusable function to merge df_obs_all, df_comps_all, and df_images_all. OK 20200702.
    :param filters_to_include: either one filter name, or a list of filters. Only observations in
                that filter or filters will be retained.
                Use None to include all filters [None, or string, or list of strings]
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

    # Keep only obs from images in requested filter(s):
    if filters_to_include is not None:
        image_is_in_filter = [(f in filters_to_include) for f in df_images_all['Filter']]
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


def make_comp_variability_plots(df_plot_comp_obs, xlabel_jd, transform, sigma):
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
        plt.savefig('Image5_Comp Variability_' + '{:02d}'.format(i_figure + 1) + '.png')

    # Verify that all comps were plotted exactly once (debug):
    all_comps_plotted_once = (sorted(comp_ids) == sorted(plotted_comp_ids))
    # print('all comp ids were plotted exactly once = ', str(all_comps_plotted_once))
    if not all_comps_plotted_once:
        print('comp ids plotted more than once',
              [item for item, count in Counter(plotted_comp_ids).items() if count > 1])


def draw_x_line(ax, x_value, color='lightgray'):
    ax.axvline(x=x_value, color=color, linewidth=1, zorder=-100)


_____SUPPORT________________________________________________ = 0


def write_control_txt_stub(this_directory, log_file, df):
    defaults = DO_PHOT_COMP_SELECTION_DEFAULTS
    df['SecondsRelative'] = [(utc - df['UTC_mid'].min()).total_seconds() for utc in df['UTC_mid']]
    i_earliest = df['SecondsRelative'].nsmallest(n=1).index[0]
    i_latest = df['SecondsRelative'].nlargest(n=1).index[0]
    earliest_filename = df.loc[i_earliest, 'Filename']
    latest_filename = df.loc[i_latest, 'Filename']
    lines = [';----- This is ' + CONTROL_FILENAME + ' for directory:\n;      ' + this_directory,
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
             ';----- OPTIONS for regression model, rarely used:',
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
    fullpath = os.path.join(this_directory, CONTROL_FILENAME)
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(lines)
            log_file.write('New ' + CONTROL_FILENAME + ' file written.\n')


def write_transform_control_txt_stub(this_directory, log_file):
    defaults = TRANSFORM_COMP_SELECTION_DEFAULTS
    lines = [';----- This is ' + TRANSFORM_CONTROL_FILENAME + ' for directory:\n;      ' + this_directory,
             ';',
             ';===== SELECTION CRITERIA BLOCK =====================================',
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
             ';----- OPTIONS for regression model, rarely used:',
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
    fullpath = os.path.join(this_directory, TRANSFORM_CONTROL_FILENAME)
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(lines)
            log_file.write('New ' + TRANSFORM_CONTROL_FILENAME + ' file written.\n')


def write_color_control_txt_stub(this_directory, log_file):
    defaults = COLOR_COMP_SELECTION_DEFAULTS
    lines = [';----- This is ' + COLOR_INDEX_CONTROL_FILENAME + ' for directory:\n;      ' + this_directory,
             ';',
             ';===== SELECTION CRITERIA BLOCK =====================================',
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
             ';----- OPTIONS for regression model, rarely used:',
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
    fullpath = os.path.join(this_directory, COLOR_INDEX_CONTROL_FILENAME)
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(lines)
            log_file.write('New ' + COLOR_INDEX_CONTROL_FILENAME + ' file written.\n')


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


def apply_calc_transform_selections(df_transform, user_selections):
    """ Applies user selections to set desired df_transform's 'UseInModel' row(s) to False, in-place.
    :param df_transform: observations dataframe. [pandas DataFrame]
    :param user_selections: dict of lists of items to remove from df_obs before modeling. [dict of lists]
    :return df_obs: [None] (both dataframes are modified in-place.)
    """
    # TODO: are we lucky enough for this to work as is?
    apply_do_phot_selections(df_transform, user_selections)


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


#


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



# def do_ci_plot(result, mag_diff, mp_color, sigma_color, mp_string, filter_string, color_string):
#     """ Construct, display, and save as PNG a plot of the color index regression.
#     :param result: results from OLS regression. [object]
#     :param mag_diff: inst. magnitude difference (typically R-I) for MP. [float]
#     :param mp_color: color index Sloan (r-i) for MP. [float]
#     :param sigma_color: uncertainty in mp_color. [float]
#     :param mp_string: MP designation, e.g., 'MP 2415' or '2415 Ganesa'. [string]
#     :param filter_string: filters used in observations, e.g., 'R-I'. [string]
#     :param color_string: passbands in which color is expressed, e.g., 'Sloan r-i'. [string]
#     :return: None
#     """
#     fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(9, 6))  # (width, height) in "inches"
#     ax = axes  # not subscripted if just one subplot in Figure
#     page_title = 'Color index regression for ' + mp_string
#     ax.set_title(page_title, color='darkblue', fontsize=20, pad=30)
#     x_values = [x[1] for x in result.model.exog]  # unpack from 2-d array.
#     y_values = result.model.endog
#     ax.set_xlabel('Diff(instrumental magnitudes): ' + filter_string)
#     ax.set_ylabel('Color index: ' + color_string)
#     ax.grid(True, color='lightgray', zorder=-1000)
#     ax.scatter(x=x_values, y=y_values, alpha=0.7, color='black', zorder=+1000)
#     x_mp = [mag_diff]
#     y_mp = [mp_color]
#     ax.scatter(x=x_mp, y=y_mp, alpha=1, color='orangered', zorder=+1001)
#
#     plt.show()
#     filename = 'Image_ColorIndex.png'
#     fig.savefig(filename)