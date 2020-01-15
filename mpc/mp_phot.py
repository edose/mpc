# Python core packages:
import os
import sys
from math import cos, sin, sqrt, pi, log10, log, floor
from collections import Counter
from datetime import datetime, timezone

# External packages:
import numpy as np
import pandas as pd
# import requests
# from scipy.interpolate import interp1d
# from statistics import median, mean
import matplotlib.pyplot as plt

# From this (mpc) package:
# from mpc.mpctools import *
from mpc.catalogs import Refcat2, get_bounding_ra_dec

# From external (EVD) package photrix:
from photrix.image import Image, FITS, Aperture
from photrix.util import RaDec, jd_from_datetime_utc, degrees_as_hex, ra_as_hours, MixedModelFit

__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

DAYS_PER_YEAR_NOMINAL = 365.25
VALID_FITS_FILE_EXTENSIONS = ['.fits', '.fit', '.fts']
DEGREES_PER_RADIAN = 180.0 / pi

# To assess FITS files in assess():
MIN_FWHM = 1.5  # in pixels.
MAX_FWHM = 14  # "
REQUIRED_FILES_PER_FILTER = [('Clear', 5)]
FOCUS_LENGTH_MAX_PCT_DEVIATION = 3.0

MARGIN_RA_ZERO = 5  # in degrees, to handle RA ~zero; s/be well larger than images' height or width.

# To screen observations:
MAX_MAG_UNCERT = 0.03  # min signal/noise for comp obs (as InstMagSigma).
ADU_CAL_SATURATED = 60000
ADU_UR_SATURATED = 54000  # This probably should go in Instrument class, but ok for now.

# For this package:
MP_TOP_DIRECTORY = 'J:/Astro/Images/MP Photometry/'
LOG_FILENAME = 'mp_photometry.log'
CONTROL_FILENAME = 'control.txt'
DF_OBS_FILENAME = 'df_obs.csv'
DF_COMPS_FILENAME = 'df_comps.csv'
TRANSFORM_CLEAR_SR_SR_SI = 0.025  # estimate from MP 1047 20191109 (37 images).
DEFAULT_MODEL_OPTIONS = {'fit_transform': False, 'fit_extinction': False,
                         'fit_vignette': True, 'fit_xy': False,
                         'fit_jd': False}  # defaults if not found in file control.txt.
DEFAULT_MP_COLOR_R_I = 0.2  # close to known Sloan mean (r-i) for MPs.

# For selecting comps within Refcat2 object (intentionally wide; can narrow with control.txt later):
MIN_R_MAG = 10
MAX_R_MAG = 16
MAX_G_UNCERT = 20  # millimagnitudes
MAX_R_UNCERT = 20  # "
MAX_I_UNCERT = 20  # "
MIN_BV_COLOR = 0.45
MAX_BV_COLOR = 0.95

# For package photrix:
PHOTRIX_TOP_DIRECTORIES = ['C:/Astro/Borea Photrix/', 'J:/Astro/Images/Borea Photrix Archives/']
FILE_RENAMING_FILENAME = 'File-renaming.txt'
UR_FILE_RENAMING_PATH = 'Photometry/File-renaming.txt'
DF_MASTER_FILENAME = 'df_master.csv'


_____ATLAS_BASED_WORKFLOW________________________________________________ = 0


def start(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Preliminaries to begin MP photometry workflow.
    :param mp_top_directory: path of lowest directory common to all MP photometry FITS, e.g.,
               'J:/Astro/Images/MP Photometry' [string]
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

    # Find correct photrix directory:
    photrix_directory = None
    for p_dir in PHOTRIX_TOP_DIRECTORIES:
        this_directory = os.path.join(p_dir, an_string)
        df_master_fullpath = os.path.join(this_directory, 'Photometry', DF_MASTER_FILENAME)
        if os.path.exists(df_master_fullpath):
            if os.path.isfile(df_master_fullpath):
                print(df_master_fullpath, 'found.')
                photrix_directory = this_directory
                break
    if photrix_directory is None:
        print(' >>>>> start() cannot find photrix directory. Log file unchanged.')
        return

    # Initiate log file and finish:
    log_file = open(LOG_FILENAME, mode='w')  # new file; wipe old one out if it exists.
    log_file.write(mp_directory + '\n')
    log_file.write('MP: ' + mp_string + '\n')
    log_file.write('AN: ' + an_string + '\n')
    log_file.write('Photrix found: ' + photrix_directory + '\n')
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
    # TODO: Test this function.
    if mp_number is None or an_string is None:
        print(' >>>>> Usage: start(top_directory, mp_number, an_string)')
        return

    # Go to proper working directory:
    mp_int = int(mp_number)  # put this in try/catch block?
    mp_string = str(mp_int)
    this_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    os.chdir(this_directory)

    # Verify that proper log file already exists in the working directory:
    log_this_directory, log_mp_string, log_an_string, _ = get_context()
    if log_mp_string.lower() == mp_string.lower() and log_an_string.lower() == an_string.lower():
        print('READY TO GO in', this_directory)
    else:
        print(' >>>>> Can\'t resume in', this_directory)


def assess():
    """  First, verify that all required files are in the working directory or otherwise accessible.
         Then, perform checks on FITS files in this directory before performing the photometry proper.
         Modeled after and extended from assess() found in variable-star photometry package 'photrix'.
    :return: [None]
    """
    this_directory, mp_string, an_string, photrix_directory = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    log_file.write('\n===== access()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    n_warnings = 0

    # Get FITS file names in current directory:
    fits_filenames = get_fits_filenames(this_directory)
    print(str(len(fits_filenames)) + ' FITS files found:')
    log_file.write(str(len(fits_filenames)) + ' FITS files found:' + '\n')

    # Ensure that relevant uncalibrated (Ur) files and cross-reference to them all accessible:
    print(' >>>>>', os.path.join(photrix_directory, 'Photometry', FILE_RENAMING_FILENAME))
    df_ur = pd.read_csv(os.path.join(photrix_directory, 'Photometry', FILE_RENAMING_FILENAME),
                        sep=';', index_col='PhotrixName')
    print('   Verifying presence of', len(fits_filenames), 'Ur (uncalibrated) FITS files...')
    ur_fits_missing = []
    for fits_name in fits_filenames:
        ur_fits_name = df_ur.loc[fits_name, 'UrName']
        ur_fits = FITS(photrix_directory, 'Ur', ur_fits_name)
        if not ur_fits.is_valid:
            ur_fits_missing.append(ur_fits_name)
            print(' >>>>> UR file ' + ur_fits_name + ' not found.')
            log_file.write(' >>>>> UR file ' + ur_fits_name + ' not found.')
    if len(ur_fits_missing) >= 1:
        print(' >>>>> ' + str(len(ur_fits_missing)) + ' Ur FITS files missing.')
        log_file.write(' >>>>> ' + str(len(ur_fits_missing)) + ' Ur FITS files missing.\n')
    else:
        print('   All Ur files found.')
        log_file.write('   All Ur files found.\n')
    n_warnings += len(ur_fits_missing)

    # Verify that all required FITS file types exist within this directory and are valid:
    filter_counter = Counter()
    for filename in fits_filenames:
        fits = FITS(this_directory, '', filename)
        if fits.is_valid:
            filter_counter[fits.filter] += 1
    for filt, required in REQUIRED_FILES_PER_FILTER:
        if filter_counter[filt] >= required:
            print('   ' + str(filter_counter[filt]), 'in filter', filt + '.')
            log_file.write('   ' + str(filter_counter[filt]) + ' in filter ' + filt + '.\n')
        else:
            print(' >>>>> ' + str(filter_counter[filt]) +
                  ' in filter ' + filt + ' found: NOT ENOUGH. ' +
                  str(required) + ' are required.')
            log_file.write(' >>>>> ' + str(filter_counter[filt]) +
                           ' in filter ' + filt + ' found: NOT ENOUGH. ' +
                           str(required) + ' are required.')
            n_warnings += 1

    # Start dataframe for main FITS integrity checks:
    fits_extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in fits_filenames])
    df = pd.DataFrame({'Filename': fits_filenames,
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

    # Non-FITS files: should be none; report and REMOVE THEM from df:
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
        print('Next: (1) enter MP pixel positions in control.txt,\n      (2) make_df_obs()')
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
    lines = [';----- This is control.txt for directory ' + this_directory,
             ';',
             ';===== For make_df_obs() ===========================================',
             ';----- REQUIRED:',
             '#MP  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
                                           'early filename, change if needed',
             '#MP  ' + latest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
                                         ' late filename, change if needed',
             ';',
             ';===== For mp_phot() ===============================================',
             ';----- OPTIONAL for comp selection before model:',
             ';#SERIAL 123,7776 2254   16   ; to omit observations by Serial number (many per line OK)',
             ';#COMP  2245 144,   781       ; to omit comp by comp ID (many per line OK)',
             ';#IMAGE  Obj-0000-V           ; to omit FITS image Obj-0000-V.fts specifically',
             (';#MIN_R_MAG ' + str(MIN_R_MAG)).ljust(30) + '; default=' + str(MIN_R_MAG),
             (';#MAX_R_MAG ' + str(MAX_R_MAG)).ljust(30) + '; default=' + str(MAX_R_MAG),
             (';#MIN_BV_COLOR ' + str(MIN_BV_COLOR)).ljust(30) + '; default=' + str(MIN_BV_COLOR),
             (';#MAX_BV_COLOR ' + str(MAX_BV_COLOR)).ljust(30) + '; default=' + str(MAX_BV_COLOR),
             ';',
             ';----- OPTIONS for defining model:',
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
             ';',
             ';===== GIVE MP COLOR if known ======================================',
             ';#MP_RI_COLOR UseDefault      ; or give a *measured* r-i (ATLAS refcat2) color',
             ';'
             ]
    lines = [line + '\n' for line in lines]
    fullpath = os.path.join(this_directory, 'control.txt')
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(lines)
            log_file.write('New control.txt file written.\n')

    log_file.close()


def make_df_obs():
    """ For one MP on one night: gather images and ATLAS refcat2 catalog data, make df_comps and df_obs.
    :return: [None]
    USAGE: make_df_obs()
    """
    this_directory, mp_string, an_string, photrix_directory = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n===== make_df_obs()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Get all relevant FITS filenames, make lists of FITS and Image objects (per photrix):
    fits_names = get_fits_filenames(this_directory)
    fits_list = [FITS(this_directory, '', fits_name) for fits_name in fits_names]  # FITS objects
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
        if image.fits.filter == 'Clear':
            for i_comp in df_radec.index:
                ra = df_radec.loc[i_comp, 'RA_deg']
                dec = df_radec.loc[i_comp, 'Dec_deg']
                x0, y0 = image.fits.xy_from_radec(RaDec(ra, dec))
                image.add_aperture(str(i_comp), x0, y0)
            print(image.fits.filename + ':', str(len(image.apertures)), 'apertures.')

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

    # Make df_obs, by building from Image apertures (adapted from photrix.process.make_df_master()):
    df_ur = pd.read_csv(os.path.join(photrix_directory, 'Photometry', 'File-renaming.txt'),
                        sep=';', index_col='PhotrixName')
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
        df_apertures = df_apertures.loc[df_apertures['net_flux'] > 0.0, :]  # remove absent sources.
        df_apertures = df_apertures.loc[df_apertures['max_adu'] <= ADU_CAL_SATURATED, :]
        df_apertures['InstMag'] = -2.5 * np.log10(df_apertures['net_flux'])\
            + 2.5 * log10(image.fits.exposure)
        df_apertures['InstMagSigma'] = (2.5 / log(10)) * \
                                       (df_apertures['net_flux_sigma'] / df_apertures['net_flux'])
        snr_ok = [sig < MAX_MAG_UNCERT for sig in df_apertures['InstMagSigma']]
        df_apertures = df_apertures.loc[snr_ok, :]
        df_apertures.drop(['n_disc_pixels', 'n_annulus_pixels', 'max_adu',
                           'net_flux', 'net_flux_sigma'],
                          axis=1, inplace=True)  # delete unneeded columns.

        # Find and add each aperture's max ADU from Ur (pre-calibrated) image (to enforce non-saturation):
        ur_filename = df_ur.loc[image.fits.filename, 'UrName']
        df_apertures['UrFITSfile'] = ur_filename
        ur_image = Image.from_fits_path(photrix_directory, 'Ur', ur_filename)
        max_adu_list, log_adu_list = [], []
        for star_id in df_apertures.index:
            ap = Aperture(ur_image, star_id, df_apertures.loc[star_id, 'Xcentroid'],
                          df_apertures.loc[star_id, 'Ycentroid'], df_punches=None)
            max_adu_list.append(ap.max_adu)
        df_apertures['MaxADU_Ur'] = max_adu_list
        not_saturated = [(max_adu > 0.0) for max_adu in max_adu_list]  # screen for saturation right here.
        df_image_obs = df_apertures[not_saturated].copy()

        # Add FITS-specific columns:
        df_image_obs['FITSfile'] = image.fits.filename
        df_image_obs['JD_start'] = jd_from_datetime_utc(image.fits.utc_start)
        df_image_obs['UTC_start'] = image.fits.utc_start
        df_image_obs['Exposure'] = image.fits.exposure
        df_image_obs['UTC_mid'] = image.fits.utc_mid
        df_image_obs['JD_mid'] = jd_from_datetime_utc(image.fits.utc_mid)
        df_image_obs['Filter'] = image.fits.filter
        df_image_obs['Airmass'] = image.fits.airmass
        df_image_obs['JD_fract'] = np.nan  # placeholder (actual value requires that all JDs be known).

        df_image_obs_list.append(df_image_obs)

    df_obs = pd.DataFrame(pd.concat(df_image_obs_list, ignore_index=True, sort=True))
    df_obs['Type'] = ['MP' if id.startswith('MP_') else 'Comp' for id in df_obs['SourceID']]
    df_obs.sort_values(by=['JD_mid', 'Type', 'SourceID'], inplace=True)
    df_obs.insert(0, 'Serial', range(1, 1 + len(df_obs)))
    df_obs.index = list(df_obs['Serial'])
    df_obs = reorder_df_columns(df_obs, ['Serial', 'FITSfile', 'JD_mid', 'SourceID', 'Type',
                                         'Filter', 'Exposure', 'InstMag', 'InstMagSigma'])
    jd_floor = floor(df_obs['JD_mid'].min())  # requires that all JD_mid values be known.
    df_obs['JD_fract'] = df_obs['JD_mid'] - jd_floor
    print('   ' + str(len(df_obs)) + ' obs retained.')

    # Make df_comps:
    df_comps = refcat2.selected_columns(['RA_deg', 'Dec_deg', 'RP1', 'R1', 'R10',
                                         'g', 'dg', 'r', 'dr', 'i', 'di', 'BminusV', 'R', 'T_eff'])
    comp_ids = [str(i) for i in df_comps.index]
    df_comps.index = comp_ids
    df_comps.insert(0, 'CompID', comp_ids)

    print('   ' + str(len(df_comps)) + ' comps retained.')

    # Write df_obs to CSV file (rather than returning the df):
    fullpath_df_obs = os.path.join(this_directory, DF_OBS_FILENAME)
    df_obs.to_csv(fullpath_df_obs, sep=';', quotechar='"',
                  quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    n_comp_obs = sum([t.lower() == 'comp' for t in df_obs['Type']])
    n_mp_obs = sum([t.lower() == 'mp' for t in df_obs['Type']])
    log_file.write(DF_OBS_FILENAME + ' written: ' + str(n_comp_obs) + ' comp obs & ' +
                   str(n_mp_obs) + ' MP obs.\n')
    print('df_obs written to', fullpath_df_obs)

    # Write df_comps to CSV file (rather than returning the df):
    fullpath_df_comps = os.path.join(this_directory, DF_COMPS_FILENAME)
    df_comps.to_csv(fullpath_df_comps, sep=';', quotechar='"',
                    quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('df_comps written to', fullpath_df_comps)
    log_file.write(DF_COMPS_FILENAME + ' written: ' + str(len(df_comps)) + ' comps.\n')
    log_file.close()


def mp_phot():
    """ Use obs quality and user's control.txt to screen obs for photometry and diagnostic plotting."""
    this_directory, mp_string, an_string, photrix_directory = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n===== mp_phot()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Load required data:
    df_obs_all = read_df_obs()
    df_comps_all = read_df_comps()
    state = get_session_state()  # for extinction and transform values.
    obs_selections = read_obs_selections()
    df_obs = select_df_obs(df_obs_all, obs_selections)  # remove comps, obs, images per control.txt file.

    # Keep obs only for comps present in every Clear image (df_obs):
    n_fitsfiles = len(df_obs['FITSfile'].drop_duplicates())
    comp_id_list = df_obs['SourceID'].copy().drop_duplicates()
    obs_counts = [sum(df_obs['SourceID'] == id) for id in comp_id_list]
    comp_ids_to_remove = [id for (id, n) in zip(comp_id_list, obs_counts) if n != n_fitsfiles]
    rows_to_remove = df_obs['SourceID'].isin(comp_ids_to_remove)
    df_obs = df_obs.loc[~rows_to_remove, :]

    # Keep only comps (in df_comps) still in df_obs:
    comp_id_list = df_obs['SourceID'].copy().drop_duplicates()  # refresh list.
    rows_to_keep = df_comps_all['CompID'].isin(comp_id_list)
    df_comps = df_comps_all.loc[rows_to_keep, :]

    # Make photometric model via mixed-model regression:
    option_dict = read_model_options()
    model = SessionModel(df_obs, df_comps, state, option_dict)

    # Organize data, write model_summary.txt, write canopus.txt.

    # Call do_plots():
    do_plots(model, df_obs_all, df_comps_all, mp_string, an_string)

    # Write log file lines:


def do_plots(model, df_obs_all, df_comps_all, mp_string, an_string):
    """ Make all diagnostic plots. User will use the plots to decide whether to keep solution, or to
        adjust control.txt and run mp_phot() again.
        This is top-level fn & outside of SessionModel class, so can access all data incl
            mag & colorlimits, original (pre-selection) obs data, etc.
    :param model: Results of running model on comp and mp data. [SessionModel object]
    :param df_obs_all: all observations from make_df_obs(), retained for model or not. [pandas DataFrame]
    :param df_comps_all: all comp stars from make_df_obs(), retained for model or not. [pandas DataFrame]
    :param mp_string:
    :param an_string:
    :return: [None]
    """
    # Preliminary data prep:
    is_mp_obs = (model.df_obs['Type'] == 'MP')
    is_comp_obs = (model.df_obs['Type'] == 'Comp')
    n_mp_obs = sum(is_mp_obs)
    n_comp_obs = sum(is_comp_obs)
    df_mp_obs = model.df_obs.loc[is_mp_obs, :]
    df_comp_obs = model.df_obs.loc[is_comp_obs, :]
    model_serials = df_comp_obs['Serial']

    # FIGURE 1 (Q-Q plot), one entire page:
    # Heavily adapted from photrix.process.SkyModel.plots().
    from scipy.stats import norm
    df_y = pd.DataFrame({'Residual': model.mm_fit.df_observations['Residual'] * 1000.0,
                         'Serial': model_serials})
    df_y = df_y.sort_values(by='Residual')
    n = len(df_y)
    t_values = [norm.ppf((k-0.5)/n) for k in range(1, n+1)]

    # Construct Q-Q plot:
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 8))  # (width, height) in "inches"
    ax.grid(True, color='lightgray', zorder=-1000)
    ax.set_title('Q-Q plot of Residuals: ' + mp_string + '   ' + an_string,
                 color='darkblue', fontsize=20, weight='bold')
    ax.set_xlabel('t (sigma.residuals = ' + str(round(1000.0 * model.mm_fit.sigma, 1)) + ' mMag)')
    ax.set_ylabel('Residual (mMag)')
    ax.scatter(x=t_values, y=df_y['Residual'], alpha=0.6, color='darkgreen', zorder=+1000)

    # Label potential outliers:
    mean_y = df_y['Residual'].mean()
    std_y = df_y['Residual'].std()
    z_score_y = (df_y['Residual'] - mean_y) / std_y
    df_y['T'] = t_values
    df_to_label = df_y[abs(z_score_y) >= 2.0]
    for x, y, label in zip(df_to_label['T'], df_to_label['Residual'], df_to_label['Serial']):
        ax.annotate(label, xy=(x, y), xytext=(4, -4),
                    textcoords='offset points', ha='left', va='top', rotation=-40)

    # Add reference line:
    x_low = 1.10 * min(df_y['T'])
    x_high = 1.10 * max(df_y['T'])
    y_low = x_low * std_y
    y_high = x_high * std_y
    ax.plot([x_low, x_high], [y_low, y_high], color='gray', zorder=-100, linewidth=1)

    # Add annotation: number of observations:
    fig.text(x=0.5, y=0.87,
             s=str(n_comp_obs) + ' comp-star observations in model.',
             verticalalignment='top', horizontalalignment='center',
             fontsize=12)
    fig.canvas.set_window_title('Q-Q Plot:  MP ' + mp_string + '   AN ' + an_string)
    plt.show()


class SessionModel:
    def __init__(self, df_obs, df_comps, state, option_dict, do_plots=True):
        """  Makes and holds photometric model via mixed-model regression. Affords prediction for MP mags.
        :param df_obs:
        :param df_comps:
        :param state:
        :param option_dict: holds options for making comp fit; its elements are:
            fit_transform: True iff transform is to be fit; never True in actual photometry model,
                   True set only rarely, to extract transform from images of one field of view. [boolean]
            fit_extinction: True iff extinction is to be fit (uncommon; usually get known value from
                   Site object). [boolean]
            fit_vignette: True iff centered, parabolic term to be included in model. [boolean]
            fit_xy: True iff linear x and y terms to be included in model. [boolean]
            fit_jd: True iff linear time term (zero-point creep in time seen in plot of "cirrus"
                   random-effect term). [boolean]
        :param do_plots: True iff plots desired; always true for actual photometry model. [boolean]
        """
        self.df_obs = df_obs.copy()
        self.df_comps = df_comps.copy()
        self.state = state
        defaults = DEFAULT_MODEL_OPTIONS
        self.fit_transform = option_dict.get('fit_transform', defaults['fit_transform'])
        self.fit_extinction = option_dict.get('fit_extinction', defaults['fit_extinction'])
        self.fit_vignette = option_dict.get('fit_vignette', defaults['fit_vignette'])
        self.fit_xy = option_dict.get('fit_xy', defaults['fit_xy'])
        self.fit_jd = option_dict.get('fit_jd', defaults['fit_jd'])
        self.do_plots = do_plots

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
        self._calc_mp_mags()
        self._build_output()
        if do_plots:
            self.make_plots()

    def _prep_and_do_regression(self):
        """ Using photrix.util.MixedModelFit class (which wraps statsmodels.MixedLM.from_formula() etc).
            Use ONLY comp data in the model. Later use model's .predict() to calculate best MP mags.
            This is adapted from photrix package SkyModel.
        :return: [None]
        """
        # First, extend comp-only obs data with appropriate data from df_comps:
        df_comp_obs_subset = self.df_obs.loc[(self.df_obs['Type'] == 'Comp'),
                                             ['Serial', 'FITSfile', 'SourceID', 'Type', 'InstMag',
                                              'Airmass', 'JD_fract', 'Vignette', 'X1024', 'Y1024']].copy()
        df_model = pd.merge(left=df_comp_obs_subset, right=self.df_comps,
                            how='left', left_on='SourceID', right_on='CompID', sort=False)
        df_model['CI'] = df_model['r'] - df_model['i']

        # Initiate dependent-variable offset, which will aggregate all such offset terms:
        dep_var_offset = df_model['r'].copy()  # *copy* CatMag, or it will be damaged

        # Build fixed-effect (x) variable list and construct dep-var offset:
        fixed_effect_var_list = []
        if self.fit_transform:
            fixed_effect_var_list.append('CI')
        else:
            self.transform_fixed = TRANSFORM_CLEAR_SR_SR_SI  # TODO: measure this, then get from self.state.
            print(' >>>>> Transform (Color Index) not fit: value fixed at',
                  '{0:.3f}'.format(self.transform_fixed))
        if self.fit_extinction:
            fixed_effect_var_list.append('Airmass')
        else:
            extinction = self.state['extinction']['Clear']
            dep_var_offset += extinction * df_model['Airmass']
            print(' >>>>> Extinction (Airmass) not fit: value fixed at',
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
        df_model[self.dep_var_name] = df_model['InstMag'] - dep_var_offset

        # Execute regression:
        self.mm_fit = MixedModelFit(data=df_model, dep_var=self.dep_var_name,
                                    fixed_vars=fixed_effect_var_list,
                                    group_var=random_effect_var_name)
        if self.mm_fit.statsmodels_object.scale != 0.0 and \
                self.mm_fit.statsmodels_object.nobs == len(df_model):
            print(self.mm_fit.statsmodels_object.summary())

    def _calc_mp_mags(self):
        df_mp_obs = self.df_obs.loc[(self.df_obs['Type'] == 'MP'),
                                    ['Serial', 'FITSfile', 'SourceID', 'Type', 'InstMag',
                                     'Airmass', 'JD_fract', 'Vignette', 'X1024', 'Y1024']].copy()
        bogus_cat_mag = 0.0  # we'll need this later, to correct raw predictions.
        df_mp_obs['CatMag'] = bogus_cat_mag  # totally bogus local value, corrected for later.
        df_mp_obs['CI'] = DEFAULT_MP_COLOR_R_I  # best we can do without directly measuring it.
        raw_predictions = self.mm_fit.predict(df_mp_obs, include_random_effect=False)

        # Compute dependent-variable offsets for MP:
        dep_var_offsets = pd.Series(len(df_mp_obs) * [0.0], index=raw_predictions.index)
        if self.fit_transform is False:
            dep_var_offsets += self.transform_fixed * df_mp_obs['CI']
        if self.fit_extinction is False:
            dep_var_offsets += self.state['extinction']['Clear'] * df_mp_obs['Airmass']

        # Extract best MP mag (in synthetic Sloan r):
        best_mp_mags = df_mp_obs['InstMag'] - dep_var_offsets - raw_predictions + bogus_cat_mag
        return best_mp_mags

    def _build_output(self):
        pass

    def make_plots(self):
        pass






_____SUPPORT________________________________________________ = 0


def get_context():
    """ This is run at beginning of workflow functions (except start()) to orient the function.
    :return: 4-tuple: (this_directory, mp_string, an_string, photrix_directory) [3 strings]
    """
    this_directory = os.getcwd()
    if not os.path.isfile(LOG_FILENAME):
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
    photrix_directory = lines[3][len('Photrix found:'):].strip()
    return this_directory, mp_string, an_string, photrix_directory


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


def read_df_obs():
    """  Simple utility to read df_obs.csv file and return the original DataFrame.
    :return: df_obs from make_df_obs() [pandas Dataframe]
    """
    this_directory, _, _, _ = get_context()
    fullpath = os.path.join(this_directory, DF_OBS_FILENAME)
    df_obs = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_obs


def read_df_comps():
    """  Simple utility to read df_comps.csv file and return the original DataFrame.
    :return: df_comps from make_df_obs() [pandas Dataframe]
    """
    this_directory, _, _, _ = get_context()
    fullpath = os.path.join(this_directory, DF_COMPS_FILENAME)
    df_comps = pd.read_csv(fullpath, sep=';', index_col=0)
    comp_ids = [str(id) for id in df_comps['CompID']]
    df_comps.loc[:, 'CompID'] = comp_ids
    df_comps.index = comp_ids
    return df_comps


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
    refcat2.select_bv_color(MIN_BV_COLOR, MAX_BV_COLOR)
    lines.append('Refcat2: BV color screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_dgaia()
    lines.append('Refcat2: dgaia screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.remove_overlapping()
    lines.append('Refcat2: overlaps removed to ' + str(len(refcat2.df_selected)) + ' stars.')
    return lines

def read_obs_selections():
    """ Reads observation selection lines from control.txt, compiles lists of observations to remove
            (for use by select_df_obs()).
    :return: dict of lists of items to remove from df_obs before use in model. [dict of lists]
    """
    # Parse control.txt:
    with open(CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines

    serial_list, comp_list, image_list = [], [], []
    for line in lines:
        content = line.strip().split(';')[0].strip()  # upper case, comments removed.
        content_upper = content.upper()
        if content_upper.startswith('#SERIAL'):
            values = content[len('#SERIAL'):].strip().replace(',',' ').split()
            serial_list.extend([int(v) for v in values])
        if content_upper.startswith('#COMP'):
            values = content[len('#COMP'):].strip().replace(',', ' ').split()
            comp_list.extend([int(v) for v in values])
        if content_upper.startswith('#IMAGE'):
            image_filename = content[len('#IMAGE'):].strip()
            image_list.append(image_filename)
    return {'serials': serial_list, 'comps': comp_list, 'images': image_list}


def select_df_obs(df_obs, obs_selections):
    """ Reads file control.txt and applies selections, returns curated df_obs.
    :param df_obs: observation dataframe from make_df_obs(). [pandas DataFrame]
    :param obs_selections: dict of lists of items to remove from df_obs before use in model. [dict of lists]
    :return df_obs: curated observation dataframe for use in mp_phot(). [pandas DataFrame]
    """
    # Apply directives:
    remove_for_serials = df_obs['Serial'].isin(obs_selections['serials'])
    remove_for_comps = df_obs['SourceID'].isin(obs_selections['comps'])
    remove_for_image = df_obs['FITSfile'].isin(obs_selections['images'])
    to_remove = remove_for_serials | remove_for_comps | remove_for_image
    df_obs = df_obs.loc[~to_remove, :].copy()
    return df_obs


def read_model_options():
    """ Returns dict of options for SessionModel (mixed-model fit of comp data).
        Reads from control.txt, uses default for any option not read in from there."""
    # Parse control.txt:
    with open(CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines
    option_dict = DEFAULT_MODEL_OPTIONS
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


_____ANCILLARY_CODE________________________________________________ = 0


def get_transform():
    """ Get transform (filter=Clear, passband=SloanR, color=(SloanR-SloanI) from one directory's df_obs.
        First, user must ensure that current working directory is correct (prob. by running resume().
    :return: dataframe, each row one image, columns=T, dT (transform & its uncertainty). [pandas DataFrame]
    USAGE: fit = get_transform()
    """
    df_obs = read_df_obs()
    is_comp = pd.Series([t.lower() == 'comp' for t in df_obs['Type']])
    is_clear = pd.Series([f.lower() == 'clear' for f in df_obs['Filter']])
    to_keep = is_comp & is_clear
    df_comp_obs = df_obs[list(to_keep)]
    df_comps = read_df_comps()
    df = pd.merge(left=df_comp_obs, right=df_comps,
                      how='left', left_on='SourceID', right_on='CompID', sort=False)
    df['CI'] = df['r'] - df['i']
    df['CI2'] = df['CI'] * df['CI']
    df['Difference'] = df['InstMag'] - df['r']
    # fit = MixedModelFit(data=df, dep_var='Difference', fixed_vars=['CI', 'CI2'], group_var='FITSfile')
    fit = MixedModelFit(data=df, dep_var='Difference', fixed_vars=['CI'], group_var='FITSfile')
    print('Transform(CI) =', str(fit.df_fixed_effects.loc['CI', 'Value']),
          'stdev =', str(fit.df_fixed_effects.loc['CI', 'Stdev']))
    # print('Transform(CI2) =', str(fit.df_fixed_effects.loc['CI2', 'Value']),
    #       'stdev =', str(fit.df_fixed_effects.loc['CI2', 'Stdev']))
    return fit


def mp_phot_test():
    from random import random, gauss, uniform, seed
    _, _, _, _ = get_context()   # forces exit if working directory is invalid.
    state = get_session_state()  # for extinction and transform values.
    seed(3423)
    n_comps = 40
    df_comps = read_df_comps()[0:n_comps]
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
    iiii = 4





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







