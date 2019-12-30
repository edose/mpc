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
# import matplotlib.pyplot as plt

# From this (mpc) package:
# from mpc.mpctools import *
from mpc.catalogs import Refcat2, get_bounding_ra_dec

# From external (EVD) package photrix:
from photrix.image import Image, FITS, Aperture
from photrix.util import RaDec, jd_from_datetime_utc, degrees_as_hex, ra_as_hours

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
# DF_QUALIFIED_OBS_FILENAME = 'df_qual_obs.csv'
# DF_QUALIFIED_COMPS_AND_MPS_FILENAME = 'df_qual_comps_and_mps.csv'

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
        print('Ready to go in', this_directory)
    else:
        print(' >>>>> Problems resuming in', this_directory)


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
        print('Next: get_obs()')
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
             ';----- REQUIRED: 2 #MP lines, for get_obs():',
             '#MP  ' + earliest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ; '
                                           'earliest filename, change if needed',
             '#MP  ' + latest_filename + '  [mp_x_pixel]  [mp_y_pixel]   ;'
                                         ' latest filename, change if needed',
             ';',
             ';----- OPTIONAL directive line examples (usually for do_phot()):',
             ';#SERIAL 123,7776 2254   16   ; to omit observations by Serial number (many per line OK)',
             ';#IMAGE  Obj-0000-V           ; to omit FITS image Obj-0000-V.fts specifically',
             ';',
             ';----- Add your directive lines below:',
             ';'
             ]
    lines = [line + '\n' for line in lines]
    fullpath = os.path.join(this_directory, 'control.txt')
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(lines)
            log_file.write('New control.txt file written.\n')

    log_file.close()


def get_obs():
    """ For one MP on one night: gather images and ATLAS refcat2 catalog data, make df_comps and df_obs."""
    this_directory, mp_string, an_string, photrix_directory = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n===== get_obs()  ' +
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
    lines = refcat2.select_for_photometry()  # apply numerous screens for quality, color, etc.
    refcat2.update_epoch(mid_session_utc)    # apply proper motions to update star positions.
    print('\n'.join(lines), '\n')
    log_file.write('\n'.join(lines) + '\n')

    # Add all comp apertures to all Image objects:
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
    mp_position_filenames, x_pixels, y_pixels = [], [], []
    with open(CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines
        for line in lines:
            if len(mp_position_filenames) < 2:
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
    if len(mp_position_filenames) != 2:
        print(' >>>>> ' + CONTROL_FILENAME + ': something wrong with #MP lines.')
        log_file.write(' >>>>> ' + CONTROL_FILENAME + ': something wrong with #MP lines.')
        sys.exit(1)

    # Extract MP *RA,Dec* positions in the 2 images, used to interpolate all MP positions:
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

    # Add all MP apertures to Image objects:
    utc0 = mp_datetime[0]
    ra0 = mp_ra_deg[0]
    dec0 = mp_dec_deg[0]
    span_seconds = (mp_datetime[1] - utc0).total_seconds()
    ra_per_second = (mp_ra_deg[1] - ra0) / span_seconds
    dec_per_second = (mp_dec_deg[1] - dec0) / span_seconds
    mp_id = 'MP_' + mp_string
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
        df_apertures['InstMag'] = -2.5 * np.log10(df_apertures['net_flux']) +\
                                  2.5 * log10(image.fits.exposure)
        df_apertures['InstMagSigma'] = (2.5 / log(10)) * \
                                       (df_apertures['net_flux_sigma'] /
                                        df_apertures['net_flux'])
        snr_ok = [sig < MAX_MAG_UNCERT for sig in df_apertures['InstMagSigma']]
        print(image.fits.filename + ': of', str(len(df_apertures)), 'apertures,',
              sum(snr_ok), 'kept for high SNR.')
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
        print(image.fits.filename + ': of', str(len(df_apertures)), 'apertures,',
              sum(not_saturated), 'kept for lack of Ur-saturation.')
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

    # Make df_comps:
    df_comps = refcat2.selected_columns(['RA_deg', 'Dec_deg', 'RP1', 'R1', 'R10',
                                         'g', 'dg', 'r', 'dr', 'i', 'di', 'BminusV', 'SloanR', 'T_eff'])

    # Write df_obs to CSV file (rather than returning the df):
    fullpath_df_obs = os.path.join(this_directory, DF_OBS_FILENAME)
    df_obs.to_csv(fullpath_df_obs, sep=';', quotechar='"',
                  quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    n_comp_obs = sum([t.lower() == 'comp' for t in df_obs['Type']])
    n_mp_obs = sum([t.lower() == 'mp' for t in df_obs['Type']])
    log_file.write(DF_OBS_FILENAME + ' written: ' + str(n_comp_obs) + ' comp obs & ' +
                   str(n_mp_obs) + ' MP obs.\n')
    print('df_mp_master written to', fullpath_df_obs)

    # Write df_comps to CSV file (rather than returning the df):
    fullpath_df_comps = os.path.join(this_directory, DF_COMPS_FILENAME)
    df_comps.to_csv(fullpath_df_comps, sep=';', quotechar='"',
                    quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    n_comps = sum([t.lower() == 'comp' for t in df_comps['Type']])
    print('df_comps_and_mp written to', fullpath_df_comps)
    log_file.write(DF_COMPS_FILENAME + ' written: ' + str(n_comps) + ' comps.\n')

    log_file.close()


def do_phot():
    """ Use obs quality and user's control.txt to screen obs for photometry and diagnostic plotting."""
    this_directory, mp_string, an_string, photrix_directory = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n===== do_phot()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Load required data:
    df_obs = get_df_obs()
    df_comps = get_df_comps()
    state = get_session_state()  # for extinction and transform values.
    df_obs = apply_user_selections(df_obs)  # remove comps, obs, images, as directed in control.txt file.

    # Keep obs only for comps present in every Clear image (df_obs):
    n_fitsfiles = len(df_obs['Filename'].drop_duplicates())
    comp_id_list = df_obs['CompID'].copy().drop_duplicates()
    obs_counts = [sum(df_obs['CompID'] == id) for id in comp_id_list]
    comp_ids_to_remove = [id for (id, n) in zip(comp_id_list, obs_counts) if n != n_fitsfiles]
    rows_to_remove = df_obs['CompID'].isin(comp_ids_to_remove)
    df_obs = df_obs.loc[~rows_to_remove, :]

    # Keep only comps (in df_comps) still in df_obs:
    comp_id_list = df_obs['CompID'].copy().drop_duplicates()  # refresh list.
    rows_to_keep = df_comps['CompID'].isin(comp_id_list)
    df_comps = df_comps.loc[rows_to_keep, :]

    # Perform photometry and add results to df_comps:



    # Write df_obs_qualified, df_comps_qualified, and df_mp to CSV files:

    # Call do_plots():

    # Write log file lines:








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
    photrix_directory = lines[3][8:].strip()
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


def get_df_obs():
    """  Simple utility to read df_obs.csv file and return the original DataFrame.
    :return: df_obs from get_obs() [pandas Dataframe]
    """
    this_directory, _, _, _ = get_context()
    fullpath = os.path.join(this_directory, DF_OBS_FILENAME)
    df_mp_master = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_mp_master


def get_df_comps():
    """  Simple utility to read df_comps.csv file and return the original DataFrame.
    :return: df_comps from get_obs() [pandas Dataframe]
    """
    this_directory, _, _, _ = get_context()
    fullpath = os.path.join(this_directory, DF_COMPS_FILENAME)
    df_comps = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_comps


def apply_user_selections(df_obs):
    """ Reads file control.txt and applies selections, returns curated df_obs.
    :param df_obs: observation dataframe from get_obs(). [pandas DataFrame]
    :return df_obs: curated observation dataframe for use in do_phot(). [pandas DataFrame]
    """
    # Parse control.txt:
    with open(CONTROL_FILENAME, 'r') as cf:
        lines = cf.readlines()
        lines = [line.split(";")[0] for line in lines]  # remove all comments
        lines = [line.strip() for line in lines]  # remove lead/trail blanks
        lines = [line for line in lines if line != '']  # remove empty lines

    serial_list, comp_list, image_list = [], [], []
    for line in lines:
        content = line.upper().strip().split(';')[0].strip()  # upper case, comments removed.
        if content.startswith('#SERIAL'):
            values = content[len('#SERIAL'):].strip().replace(',',' ').split()
            serial_list.extend([int(v) for v in values])
        if content.startswith('#COMP'):
            values = content[len('#COMP'):].strip().replace(',', ' ').split()
            comp_list.extend([int(v) for v in values])
        if content.startswith('#IMAGE'):
            image_filename = content[len('IMAGE'):].strip()
            image_list.append(image_filename)

    # Apply directives:
    remove_for_serials = df_obs['Serial'].isin(serial_list)
    remove_for_comps = df_obs['CompID'].isin(comp_list)
    remove_for_image = df_obs['FITSfilename'].isin(image_list)
    to_remove = remove_for_serials | remove_for_comps | remove_for_image
    df_obs = df_obs.loc[to_remove, :].copy()
    return df_obs





# PRELIMINARY_TESTS________________________________________________ = 0

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







