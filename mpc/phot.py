# Python core packages:
from math import cos, sin, sqrt, pi, log10, floor
from collections import Counter

# External packages:
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d
from statistics import median, mean
import matplotlib.pyplot as plt

# From this (mpc) package:
from mpc.mpctools import *
from mpc.catalogs import *

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

# To screen obs in df_mp_master:
MAX_MAG_UNCERT = 0.02  # min signal/noise for use comp obs (SNR defined here as InstMag / InstMagSigma).
ADU_UR_SATURATED = 54000  # This probably should go in Instrument class, but ok for now.

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


def gather_data():
    """ For one MP on one night:
            gather images and ATLAS refcat2 catalog data, make df_comps and df_obs."""
    this_directory, mp_string, an_string, photrix_directory = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n ===== gather_data()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Get all relevant FITS filenames, make lists of FITS and Image objects (per photrix):
    fits_names = get_fits_filenames(this_directory)
    fits_list = [FITS(this_directory, '', fits_name) for fits_name in fits_names]  # FITS objects
    image_list = [Image(fits_object) for fits_object in fits_list]  # Image objects

    # Get outermost bounding RA, Dec for all images:
    ra_deg_min_list, ra_deg_max_list, dec_deg_min_list, dec_deg_max_list = [], [], [], []
    for fits in fits_list:
        ra_deg_min, ra_deg_max, dec_deg_min, dec_deg_max = get_bounding_ra_dec(fits)
        ra_deg_min_list.append(ra_deg_min)
        ra_deg_max_list.append(ra_deg_max)
        dec_deg_min_list.append(dec_deg_min)
        dec_deg_max_list.append(dec_deg_max)





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







