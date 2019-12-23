# Python core packages:
import os
from io import StringIO
from datetime import datetime
from math import cos, sin, sqrt, pi, log10, floor
from collections import Counter

# External packages:
import numpy as np
import statsmodels.api as sm
import pandas as pd
import requests
from scipy.interpolate import interp1d
from statistics import median, mean
import matplotlib.pyplot as plt

# From this (mpc) package:
from mpc.mpctools import *

# From external (EVD) package photrix:
from photrix.image import Image, FITS, Aperture
from photrix.util import RaDec, jd_from_datetime_utc, degrees_as_hex, ra_as_hours
from photrix.fov import FOV_DIRECTORY, Fov


__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

ATLAS_REFCAT2_DIRECTORY = 'J:/Astro/Catalogs/ATLAS-refcat2/mag-0-16/'
RP1_LIMIT = 10  # arcseconds; closeness limit for flux = 0.1 * star flux
R1_LIMIT = 15   # arcseconds; closeness limit for flux = 1 * star flux
R10_LIMIT = 20  # arcseconds; closeness limit for flux = 10 * star flux
ATLAS_REFCAT2_EPOCH_UTC = datetime(2015, 1, 1) + \
                          (datetime(2016, 1, 1) - datetime(2015, 1, 1)) / 2.0  # 2015.5
ATLAS_REFCAT2_MAX_QUERY_DEFSQ = 10
DAYS_PER_YEAR_NOMINAL = 365.25
VALID_FITS_FILE_EXTENSIONS = ['.fits', '.fit', '.fts']
DEGREES_PER_RADIAN = 180.0 / pi

# To assess FITS files in assess():
MIN_FWHM = 1.5  # in pixels.
MAX_FWHM = 14  # "
REQUIRED_FILES_PER_FILTER = [('Clear', 5)]
FOCUS_LENGTH_MAX_PCT_DEVIATION = 3.0

# To screen obs in df_mp_master:
MAX_MAG_UNCERT = 0.02  # min signal/noise for use comp obs (SNR defined here as InstMag / InstMagSigma).
ADU_UR_SATURATED = 54000  # This probably should go in Instrument class, but ok for now.

MATCH_TOLERANCE_ARCSEC = 4  # to match stars across catalogs

PHOTRIX_TOP_DIRECTORIES = ['C:/Astro/Borea Photrix/', 'J:/Astro/Images/Borea Photrix Archives/']
FILE_RENAMING_FILENAME = 'File-renaming.txt'
UR_FILE_RENAMING_PATH = 'Photometry/File-renaming.txt'
DF_MASTER_FILENAME = 'df_master.csv'

MP_TOP_DIRECTORY = 'J:/Astro/Images/MP Photometry/'
LOG_FILENAME = 'mp_photometry.log'
DF_MP_MASTER_FILENAME = 'df_mp_master.csv'
DF_COMPS_AND_MPS_FILENAME = 'df_comps_and_mps.csv'
DF_QUALIFIED_OBS_FILENAME = 'df_qual_obs.csv'
DF_QUALIFIED_COMPS_AND_MPS_FILENAME = 'df_qual_comps_and_mps.csv'


ATLAS_BASED_WORKFLOW________________________________________________ = 0


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

    # Find photrix directory:
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

    log_file = open(LOG_FILENAME, mode='w')  # new file; wipe old one out if it exists.
    log_file.write(mp_directory + '\n')
    log_file.write('MP: ' + mp_string + '\n')
    log_file.write('AN: ' + an_string + '\n')
    log_file.write('Photrix: ' + photrix_directory + '\n')
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
    log_file.write('\n ===== access()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    n_warnings = 0

    # # Ensure that df_master is accessible:
    #     # df_master = get_df_master(os.path.join(photrix_directory, 'Photometry'))
    #     # df_master_exists = (len(df_master) > 100) and (len(df_master.columns) > 20)
    #     # if not df_master_exists:
    #     #     print(' >>>>> df_master.csv not found.')
    #     #     log_file.write(' >>>>> df_master.csv not found.\n')
    #     #     n_warnings += 1

    # Get FITS file names in current directory:
    fits_filenames = get_fits_filenames(this_directory)
    print(str(len(fits_filenames)) + ' FITS files found:')
    log_file.write(str(len(fits_filenames)) + ' FITS files found:' + '\n')

    # Ensure that relevant uncalibrated (Ur) files and cross-reference to them all accessible:
    df_ur = pd.read_csv(os.path.join(photrix_directory, 'Photometry', FILE_RENAMING_FILENAME),
                        sep=';', index_col='PhotrixName')
    print('Verifying presence of', len(fits_filenames), 'Ur (uncalibrated) FITS files...')
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
    n_warnings += len(ur_fits_missing)

    # Verify that all required FITS file types exist within this directory and are valid:
    filter_counter = Counter()
    for filename in fits_filenames:
        fits = FITS(this_directory, '', filename)
        if fits.is_valid:
            filter_counter[fits.filter] += 1
    for filt, required in REQUIRED_FILES_PER_FILTER:
        if filter_counter[filt] >= required:
            print('   ' + str(filter_counter[filt]), 'in filter', filt + ': OK.')
            log_file.write('   ' + str(filter_counter[filt]) + ' in filter ' + filt + ': OK.\n')
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
        print('Next: make_df_mp_master()')
        log_file.write('assess(): ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.' + '\n')
    else:
        print('\n >>>>> ' + str(n_warnings) + ' warnings (see listing above).')
        print('        Correct these and rerun assess() until no warnings remain.')
        log_file.write('assess(): ' + str(n_warnings) + ' warnings.' + '\n')
    log_file.close()


SUPPORT________________________________________________ = 0


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


# def get_df_master(directory):
#     """  Simple utility to read df_master.csv file and return the DataFrame.
#          Adapted from photrix.process.get_df_master() 20191209.
#     :return: pandas DataFrame with all comp, check, and target star raw photometric data
#          (which dataframe is generated and csv-written by R, as of May 2017).
#          The DataFrame index is set to equal column Serial (which was already a kind of index).
#     """
#     fullpath = os.path.join(directory, DF_MASTER_FILENAME)
#     if not os.path.exists(directory):
#         print(' >>>>> Cannot load', fullpath)
#         return None
#     df_master = pd.read_csv(fullpath, sep=';')
#     df_master.index = df_master['Serial']
#     return df_master


def get_fits_filenames(directory):
    all_filenames = pd.Series([e.name for e in os.scandir(directory) if e.is_file()])
    extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in all_filenames])
    is_fits = [ext.lower() in VALID_FITS_FILE_EXTENSIONS for ext in extensions]
    fits_filenames = all_filenames[is_fits]
    return fits_filenames


def get_refcat2_from_fits_file(fits_directory, fits_filename):
    fits_object = FITS(fits_directory, '', fits_filename)
    return get_refcat2_from_fits_object(fits_object)


def get_refcat2_from_fits_object(fits_object):
    image = Image(fits_object)
    deg_ra = fits_object.ra
    deg_dec = fits_object.dec
    ps = fits_object.plate_solution  # a pandas Series
    ra_list, dec_list = [], []
    for xfract in [-0.5, 0.5]:
        dx = xfract * image.xsize
        for yfract in [-0.5, 0.5]:
            dy = yfract * image.ysize
            d_ra = 1.03 * (dx * ps['CD1_1'] + dy * ps['CD1_2'])   # in degrees
            d_dec = 1.03 * (dx * ps['CD2_1'] + dy * ps['CD2_2'])  # "
            ra_list.append(deg_ra + d_ra)
            dec_list.append(deg_dec + d_dec)
    ra_deg_min = min(ra_list) % 360.0
    ra_deg_max = max(ra_list) % 360.0
    dec_deg_min = min(dec_list)
    dec_deg_max = max(dec_list)
    return get_refcat2(ATLAS_REFCAT2_DIRECTORY, ra_deg_min, ra_deg_max, dec_deg_min, dec_deg_max, True)


def get_refcat2(directory=ATLAS_REFCAT2_DIRECTORY,
                ra_deg_min=None, ra_deg_max=None, dec_deg_min=None, dec_deg_max=None, sort_ra=True):
    ra_spec_first = int(floor(ra_deg_min)) % 360  # will always be in [0, 360).
    ra_spec_last = int(floor(ra_deg_max)) % 360   # "
    if ra_spec_last < ra_spec_first:
        ra_spec_last += 360
    dec_spec_first = int(floor(dec_deg_min))
    dec_spec_first = max(dec_spec_first, -90)
    dec_spec_last = int(floor(dec_deg_max))
    dec_spec_last = min(dec_spec_last, 89)
    # print('RA: ', str(ra_spec_first), str((ra_spec_last % 360) + 1))
    # print('Dec:', str(dec_spec_first), str(dec_spec_last))
    df_list = []
    n_degsq = (ra_spec_last + 1 - ra_spec_first) * (dec_spec_last + 1 - dec_spec_first)
    if n_degsq > ATLAS_REFCAT2_MAX_QUERY_DEFSQ:
        print(' >>>>> Too many defsq (' + str(n_degsq) + ') requested. Stopping.')
    for ra_spec in range(ra_spec_first, ra_spec_last + 1):
        for dec_spec in range(dec_spec_first, dec_spec_last + 1):
            df_degsq = read_one_refcat2_sqdeg(directory, ra_spec % 360, dec_spec)
            # print('From:', str(ra_spec % 360), str(dec_spec), ' -> ', str(len(df_degsq)), 'rows.')
            df_list.append(df_degsq)
    df = pd.DataFrame(pd.concat(df_list, ignore_index=True))  # new index of unique integers

    # Trim dataframe based on user's actual limits on RA and Dec:
    ra_too_low = (df['RA_deg'] < ra_deg_min) & (df['RA_deg'] >= ra_spec_first)
    ra_too_high = (df['RA_deg'] > ra_deg_max) & (df['RA_deg'] <= (ra_spec_last % 360) + 1)
    dec_too_low = df['Dec_deg'] < dec_deg_min
    dec_too_high = df['Dec_deg'] > dec_deg_max
    # print(str(sum(ra_too_low)), str(sum(ra_too_high)), str(sum(dec_too_low)), str(sum(dec_too_high)))
    radec_outside_requested = ra_too_low | ra_too_high | dec_too_low | dec_too_high
    df = df[~radec_outside_requested]
    if sort_ra is True:
        df = df.copy().sort_values(by='RA_deg')
    return df


def read_one_refcat2_sqdeg(directory=ATLAS_REFCAT2_DIRECTORY, ra_deg_min=None, dec_deg_min=None):
    ra_deg_int = int(ra_deg_min)
    dec_deg_int = int(dec_deg_min)
    filename = '{:03d}'.format(ra_deg_int) + '{:+03d}'.format(dec_deg_int) + '.rc2'
    fullpath = os.path.join(directory, filename)
    df = pd.read_csv(fullpath, sep=',', engine='python', header=None,
                     skip_blank_lines=True, error_bad_lines=False,
                     usecols=[0, 1, 4, 5, 6, 7, 18, 19, 20, 21, 22, 25, 26, 29, 30], prefix='X')
    df.columns = ['RA_deg', 'Dec_deg', 'PM_ra', 'dPM_ra', 'PM_dec', 'dPM_dec',
                  'RP1', 'R1', 'R10', 'G', 'dG', 'R', 'dR', 'I', 'dI']
    df['RA_deg'] *= 0.00000001
    df['Dec_deg'] *= 0.00000001
    df['PM_ra'] *= 0.00001    # proper motion in arcsec/year
    df['dPM_ra'] *= 0.00001   # uncert in PM, arcsec/year
    df['PM_dec'] *= 0.00001   # proper motion in arcsec/year
    df['dPM_dec'] *= 0.00001  # uncert in PM, arcsec/year
    df['RP1'] = [None if rp1 == 999 else rp1 / 10.0 for rp1 in df['RP1']]  # radius in arcseconds
    df['R1'] = [None if r1 == 999 else r1 / 10.0 for r1 in df['R1']]       # "
    df['R10'] = [None if r10 == 999 else r10 / 10.0 for r10 in df['R10']]  # "
    df['G'] *= 0.001  # in magnitudes; dG remains in millimagnitudes
    df['R'] *= 0.001  # in magnitudes; dR remains in millimagnitudes
    df['I'] *= 0.001  # in magnitudes; dI remains in millimagnitudes
    return df


def remove_overlapping_comps(df_comps_atlas):
    rp1_too_close = pd.Series([False if pd.isnull(rp1) else (rp1 < RP1_LIMIT)
                               for rp1 in df_comps_atlas['RP1']])
    r1_too_close = pd.Series([False if pd.isnull(r1) else (r1 < R1_LIMIT)
                              for r1 in df_comps_atlas['R1']])
    r10_too_close = pd.Series([False if pd.isnull(r10) else (r10 < R10_LIMIT)
                               for r10 in df_comps_atlas['R10']])
    is_overlapping = rp1_too_close | r1_too_close | r10_too_close
    return df_comps_atlas.loc[list(~is_overlapping)]


def update_radec_to_date(df_comps_atlas, date_utc):
    d_years = (date_utc - ATLAS_REFCAT2_EPOCH_UTC).total_seconds() / (DAYS_PER_YEAR_NOMINAL * 24 * 3600)
    ra_date = [ra_epoch + d_years * pm_ra / 3600.0
               for (ra_epoch, pm_ra) in zip(df_comps_atlas['RA_deg'], df_comps_atlas['PM_ra'])]
    dec_date = [dec_epoch + d_years * pm_dec / 3600.0
                for (dec_epoch, pm_dec) in zip(df_comps_atlas['Dec_deg'], df_comps_atlas['PM_dec'])]
    df_comps_date = df_comps_atlas.copy()
    df_comps_date.loc[:, 'RA_deg'] = ra_date
    df_comps_date.loc[:, 'Dec_deg'] = dec_date
    columns_to_drop = ['PM_ra', 'dPM_ra', 'PM_dec', 'dPM_dec']  # best to remove, as they no longer be used.
    df_comps_date.drop(columns_to_drop, inplace=True, axis=1)
    return df_comps_date


def find_matching_comp(df_comps, ra_deg, dec_deg):
    # tol_deg = MATCH_TOLERANCE_ARCSEC / 3600.0
    tol_deg = 25 * MATCH_TOLERANCE_ARCSEC / 3600.0
    ra_tol = abs(tol_deg / cos(dec_deg * DEGREES_PER_RADIAN))
    dec_tol = tol_deg
    within_ra = (abs(df_comps['RA_deg'] - ra_deg) < ra_tol) |\
                (abs((df_comps['RA_deg'] + 360.0) - ra_deg) < ra_tol) |\
                (abs(df_comps['RA_deg'] - (ra_deg + 360.0)) < ra_tol)
    within_dec = abs(df_comps['Dec_deg'] - dec_deg) < dec_tol
    within_box = within_ra & within_dec
    if sum(within_box) == 0:
        return None
    elif sum(within_box) == 1:
        return (df_comps.index[within_box])[0]
    else:
        # TODO: Here, we have to choose the *closest* df_comps comp and return its index:
        df_sub = df_comps.loc[list(within_box), ['RA_deg', 'Dec_deg']]
        cos2 = cos(dec_deg) ** 2
        dist2_ra_1 = ((df_sub['RA_deg'] - ra_deg) ** 2) / cos2
        dist2_ra_2 = (((df_sub['RA_deg'] + 360.0) - ra_deg) ** 2) / cos2
        dist2_ra_3 = ((df_sub['RA_deg'] - (ra_deg + 360.0)) ** 2) / cos2
        dist2_ra = [min(d1, d2, d3) for (d1, d2, d3) in zip(dist2_ra_1, dist2_ra_2, dist2_ra_3)]
        dist2_dec = (df_sub['Dec_deg'] - dec_deg) ** 2
        dist2 = [ra2 + dec2 for (ra2, dec2) in zip(dist2_ra, dist2_dec)]
        i = dist2.index(min(dist2))
        return df_sub.index.values[i]


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


def get_df_mp_master():
    """  Simple utility to read df_mp_master.csv file and return the original DataFrame.
    :return: df_mp_master from mp_phot() [pandas Dataframe]
    """
    this_directory, _, _, _ = get_context()
    fullpath = os.path.join(this_directory, DF_MP_MASTER_FILENAME)
    df_mp_master = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_mp_master


PRELIMINARY_TESTS________________________________________________ = 0


def regress_landolt_r_mags():
    """ Regress Landolt R magnitudes against matching ATLAS refcat2 g, r, and i magnitudes."""
    # First, collect all data into dataframe df_mags:
    fov_list = get_landolt_fovs(FOV_DIRECTORY)
    mag_dict_list = []
    for fov in fov_list:
        if fov.target_type.lower() == 'standard':
            # Get ATLAS refcat2 stars within vicinity of FOV stars:
            ra_degs = [star.ra for star in fov.aavso_stars]
            ra_deg_min = min(ra_degs)
            ra_deg_max = max(ra_degs)
            dec_degs = [star.dec for star in fov.aavso_stars]
            dec_deg_min = min(dec_degs)
            dec_deg_max = max(dec_degs)
            df_refcat2 = get_refcat2(ra_deg_min=ra_deg_min, ra_deg_max=ra_deg_max,
                                     dec_deg_min=dec_deg_min, dec_deg_max=dec_deg_max)
            df_refcat2 = remove_overlapping_comps(df_refcat2)

            for fov_star in fov.aavso_stars:
                refcat2_matching = find_matching_comp(df_refcat2, fov_star.ra, fov_star.dec)
                if refcat2_matching is not None:
                    g_mag = df_refcat2.loc[refcat2_matching, 'G']
                    r_mag = df_refcat2.loc[refcat2_matching, 'R']
                    i_mag = df_refcat2.loc[refcat2_matching, 'I']
                    if g_mag is not None and r_mag is not None and i_mag is not None:
                        try:
                            mag_dict = {'fov': fov.fov_name, 'fov_star': fov_star.star_id,
                                        'Landolt_R': fov_star.mags['R'][0],
                                        'g': g_mag, 'r': r_mag, 'i': i_mag}
                            mag_dict_list.append(mag_dict)
                        except KeyError:
                            print(' >>>>> Caution:', fov.fov_name, fov_star.star_id, 'is missing R mag.')
    this_index = [mag_dict['fov'] + '_' + mag_dict['fov_star'] for mag_dict in mag_dict_list]
    df_mags = pd.DataFrame(data=mag_dict_list, index=this_index)

    # Perform regression; Landolt R is indep var, and dep variables are matching refcat2 g, r, and i mags:
    # return df_mags  # for now.
    df_y = df_mags[['Landolt_R']]
    df_x = df_mags[['r']]
    # df_x = df_mags[['g', 'r', 'i']]
    df_x.loc[:, 'intercept'] = 1.0
    weights = len(df_mags) * [1]
    result = sm.WLS(df_y, df_x, weights).fit()  # see bulletin2.util
    print(result.summary())
    return result


def get_landolt_fovs(fov_directory=FOV_DIRECTORY):
    """ Return list of FOV objects, one for each Landolt standard field."""
    all_filenames = pd.Series([e.name for e in os.scandir(fov_directory) if e.is_file()])
    fov_list = []
    for filename in all_filenames:
        fov_name = filename.split('.')[0]
        if not fov_name.startswith(('$', 'Std_NGC')):  # these are not Landolt FOVs
            fov_object = Fov(fov_name)
            if fov_object.is_valid:
                fov_list.append(fov_object)
    return fov_list


# def get_transforms_landolt_r_mags(fits_directory):
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







