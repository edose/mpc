__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
from datetime import datetime
from math import cos, sin, sqrt, pi, log10, floor
from collections import Counter

# External packages:
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statistics import median, mean
import matplotlib.pyplot as plt

# From this (mpc) package:
from mpc.mp_astrometry import *

# From external (EVD) package photrix:
from photrix.image import Image, FITS, Aperture
from photrix.util import RaDec, jd_from_datetime_utc, degrees_as_hex, ra_as_hours

# To assess FITS files in assess():
MIN_FWHM = 1.5  # in pixels.
MAX_FWHM = 14  # "
REQUIRED_FILES_PER_FILTER = [('V', 1), ('R', 1), ('I', 1), ('Clear', 5)]
FOCUS_LENGTH_MAX_PCT_DEVIATION = 3.0

# To screen obs in df_mp_master:
COMPS_MIN_SNR = 25  # min signal/noise for use comp obs (SNR defined here as InstMag / InstMagSigma).
ADU_UR_SATURATED = 54000  # This probably should go in Instrument class, but ok for now.

# Defaults for final screening of obs in qualify_comps():
COMPS_MIN_R_MAG = 10.5    # a guess
COMPS_MAX_R_MAG = 15.5  # a guess; probably needs to be customized per-MP, even per-session.
COMPS_MIN_COLOR_VI = 0.2
COMPS_MAX_COLOR_VI = 1.0

VALID_FITS_FILE_EXTENSIONS = ['.fits', '.fit', '.fts']
DEGREES_PER_RADIAN = 180.0 / pi

PHOTRIX_TOP_DIRECTORIES = ['C:/Astro/Borea Photrix/', 'C:/Astro/Borea Photrix Archives/']
FILE_RENAMING_FILENAME = 'File-renaming.txt'
UR_FILE_RENAMING_PATH = 'Photometry/File-renaming.txt'
DF_MASTER_FILENAME = 'df_master.csv'

MP_TOP_DIRECTORY = 'C:/Astro/MP Photometry/'
LOG_FILENAME = 'mp_photometry.log'
DF_MP_MASTER_FILENAME = 'df_mp_master.csv'
DF_COMPS_AND_MPS_FILENAME = 'df_comps_and_mps.csv'
DF_QUALIFIED_OBS_FILENAME = 'df_qual_obs.csv'
DF_QUALIFIED_COMPS_AND_MPS_FILENAME = 'df_qual_comps_and_mps.csv'



NEW_WORKFLOW________________________________________________ = 0


# ==========================================================================
#  phot_apass.py: comprehensive photometry for minor planets.
#  Uses : (1) df_master.csv from night's photrix run, to get Landolt standard (V,R,I) data.
#         (2) MP-field images (one each in V,R,I), to get R-mags and V-I color on comp stars, and
#         (3) MP-field images (as many as possible, in Clear), from which we get lightcurve in R.#
#
#
#  The PROPOSED WORKFLOW...below are elements that can later be strung together:
#  OK: From photrix: df_master, extract stds data as df_stds.
#  OK: From df_stds, construct Z(V), Z(R), and Z(I), time-dependent (interpolated).
#  OK: Get comp candidates from APASS.
#  OK: Interpolate RA,Dec by JD for MP location in every field image (start with X, Y pixels & make RaDecs).
#  From first field image(s), extract V, R, and I untransformed magnitudes for MP and all comps.
#  From V & I untransformed magnitudes, get color indices for MP and all comps. Screen comps on CI.
#  From R magnitudes and color indices, get best transformed R magnitudes. Screen comps on best R mags.
#  For each field image and all its best comp R mags, calculate delta(R).
#  For each field image and its delta(R), get best R mag for MP.
#  Make diagnostic plots, write MP lightcurve in canopus-ready format (
# ***********************


# def start(mp_top_directory=MP_PHOT_TOP_DIRECTORY, mp_number=None, an_string=None):
#     """  Preliminaries to begin MP photometry workflow.
#     :param mp_top_directory: path of lowest directory common to all MP photometry FITS, e.g.,
#                'C:/Astro/MP Photometry' [string]
#     :param mp_number: number of target MP, e.g., 1602 for Indiana. [integer or string].
#     :param an_string: Astronight string representation, e.g., '20191106' [string].
#     :return: [None]
#     """
#     if mp_number is None or an_string is None:
#         print(' >>>>> Usage: start(top_directory, mp_number, an_string)')
#         return
#
#     # Construct directory path and make it the working directory:
#     mp_int = int(mp_number)  # put this in try/catch block?
#     mp_string = str(mp_int)
#     mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
#     os.chdir(mp_directory)
#     print('Working directory set to:', mp_directory)
#
#     # Find photrix directory:
#     photrix_directory = None
#     for p_dir in PHOTRIX_TOP_DIRECTORIES:
#         this_directory = os.path.join(p_dir, an_string)
#         df_master_fullpath = os.path.join(this_directory, 'Photometry', DF_MASTER_FILENAME)
#         if os.path.exists(df_master_fullpath):
#             if os.path.isfile(df_master_fullpath):
#                 print(df_master_fullpath, 'found.')
#                 photrix_directory = this_directory
#                 break
#     if photrix_directory is None:
#         print(' >>>>> start() cannot find photrix directory. Log file unchanged.')
#         return
#
#     log_file = open(LOG_FILENAME, mode='w')  # new file; wipe old one out if it exists.
#     log_file.write(mp_directory + '\n')
#     log_file.write('MP: ' + mp_string + '\n')
#     log_file.write('AN: ' + an_string + '\n')
#     log_file.write('Photrix: ' + photrix_directory + '\n')
#     log_file.write('This log started: ' +
#                    '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
#     log_file.close()
#     print('Log file started.')
#     print('Next: assess()')
#
#
# def resume(mp_top_directory=MP_PHOT_TOP_DIRECTORY, mp_number=None, an_string=None):
#     """  Restart a workflow in its correct working directory,
#          but keep the previous log file--DO NOT overwrite it.
#     parameters as for start().
#     :return: [None]
#     """
#     if mp_number is None or an_string is None:
#         print(' >>>>> Usage: start(top_directory, mp_number, an_string)')
#         return
#
#     # Go to proper working directory:
#     mp_int = int(mp_number)  # put this in try/catch block?
#     mp_string = str(mp_int)
#     this_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
#     os.chdir(this_directory)
#
#     # Verify that proper log file already exists in the working directory:
#     log_this_directory, log_mp_string, log_an_string, _ = get_context()
#     if log_mp_string.lower() == mp_string.lower() and log_an_string.lower() == an_string.lower():
#         print('Ready to go in', this_directory)
#     else:
#         print(' >>>>> Problems resuming in', this_directory)
#
#
# def assess():
#     """  First, verify that all required files are in the working directory or otherwise accessible.
#          Then, perform checks on FITS files in this directory before performing the photometry proper.
#          Modeled after and extended from assess() found in variable-star photometry package 'photrix'.
#     :return: [None]
#     """
#     this_directory, mp_string, an_string, photrix_directory = get_context()
#     log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
#     log_file.write('\n ===== access()  ' +
#                    '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
#     n_warnings = 0
#
#     # Ensure that df_master is accessible:
#     df_master = get_df_master(os.path.join(photrix_directory, 'Photometry'))
#     df_master_exists = (len(df_master) > 100) and (len(df_master.columns) > 20)
#     if not df_master_exists:
#         print(' >>>>> df_master.csv not found.')
#         log_file.write(' >>>>> df_master.csv not found.\n')
#         n_warnings += 1
#
#     # Get FITS file names in current directory:
#     fits_filenames = get_fits_filenames(this_directory)
#     print(str(len(fits_filenames)) + ' FITS files found:')
#     log_file.write(str(len(fits_filenames)) + ' FITS files found:' + '\n')
#
#     # Ensure that relevant uncalibrated (Ur) files and cross-reference to them all accessible:
#     df_ur = pd.read_csv(os.path.join(photrix_directory, 'Photometry', FILE_RENAMING_FILENAME),
#                         sep=';', index_col='PhotrixName')
#     print('Verifying presence of', len(fits_filenames), 'Ur (uncalibrated) FITS files...')
#     ur_fits_missing = []
#     for fits_name in fits_filenames:
#         ur_fits_name = df_ur.loc[fits_name, 'UrName']
#         ur_fits = FITS(photrix_directory, 'Ur', ur_fits_name)
#         if not ur_fits.is_valid:
#             ur_fits_missing.append(ur_fits_name)
#             print(' >>>>> UR file ' + ur_fits_name + ' not found.')
#             log_file.write(' >>>>> UR file ' + ur_fits_name + ' not found.')
#     if len(ur_fits_missing) >= 1:
#         print(' >>>>> ' + str(len(ur_fits_missing)) + ' Ur FITS files missing.')
#         log_file.write(' >>>>> ' + str(len(ur_fits_missing)) + ' Ur FITS files missing.\n')
#     n_warnings += len(ur_fits_missing)
#
#     # Verify that all required FITS file types exist within this directory and are valid:
#     filter_counter = Counter()
#     for filename in fits_filenames:
#         fits = FITS(this_directory, '', filename)
#         if fits.is_valid:
#             filter_counter[fits.filter] += 1
#     for filt, required in REQUIRED_FILES_PER_FILTER:
#         if filter_counter[filt] >= required:
#             print('   ' + str(filter_counter[filt]), 'in filter', filt + ': OK.')
#             log_file.write('   ' + str(filter_counter[filt]) + ' in filter ' + filt + ': OK.\n')
#         else:
#             print(' >>>>> ' + str(filter_counter[filt]) +
#                   ' in filter ' + filt + ' found: NOT ENOUGH. ' +
#                   str(required) + ' are required.')
#             log_file.write(' >>>>> ' + str(filter_counter[filt]) +
#                            ' in filter ' + filt + ' found: NOT ENOUGH. ' +
#                            str(required) + ' are required.')
#             n_warnings += 1
#
#     # Start dataframe for main FITS integrity checks:
#     fits_extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in fits_filenames])
#     df = pd.DataFrame({'Filename': fits_filenames,
#                        'Extension': fits_extensions.values}).sort_values(by=['Filename'])
#     df = df.set_index('Filename', drop=False)
#     df['Valid'] = False
#     df['PlateSolved'] = False
#     df['Calibrated'] = True
#     df['FWHM'] = np.nan
#     df['FocalLength'] = np.nan
#
#     # Try to open all fits filenames as FITS, collect all info relevant to errors and warnings:
#     for filename in df['Filename']:
#         fits = FITS(this_directory, '', filename)
#         df.loc[filename, 'Valid'] = fits.is_valid
#         if fits.is_valid:
#             df.loc[filename, 'PlateSolved'] = fits.is_plate_solved
#             df.loc[filename, 'Calibrated'] = fits.is_calibrated
#             df.loc[filename, 'FWHM'] = fits.fwhm
#             df.loc[filename, 'FocalLength'] = fits.focal_length
#
#     # Non-FITS files: should be none; report and REMOVE THEM from df:
#     invalid_fits = df.loc[~ df['Valid'], 'Filename']
#     if len(invalid_fits) >= 1:
#         print('\nINVALID FITS files:')
#         for f in invalid_fits:
#             print('    ' + f)
#         print('\n')
#         df = df.loc[df['Valid'], :]  # keep only rows for valid FITS files.
#         del df['Valid']  # all rows in df now refer to valid FITS files.
#     n_warnings += len(invalid_fits)
#
#     # Now assess all FITS, and report errors & warnings:
#     not_platesolved = df.loc[~ df['PlateSolved'], 'Filename']
#     if len(not_platesolved) >= 1:
#         print('NO PLATE SOLUTION:')
#         for f in not_platesolved:
#             print('    ' + f)
#         print('\n')
#     else:
#         print('All platesolved.')
#     n_warnings += len(not_platesolved)
#
#     not_calibrated = df.loc[~ df['Calibrated'], 'Filename']
#     if len(not_calibrated) >= 1:
#         print('\nNOT CALIBRATED:')
#         for f in not_calibrated:
#             print('    ' + f)
#         print('\n')
#     else:
#         print('All calibrated.')
#     n_warnings += len(not_calibrated)
#
#     odd_fwhm_list = []
#     for f in df['Filename']:
#         fwhm = df.loc[f, 'FWHM']
#         if fwhm < MIN_FWHM or fwhm > MAX_FWHM:  # too small or large:
#             odd_fwhm_list.append((f, fwhm))
#     if len(odd_fwhm_list) >= 1:
#         print('\nUnusual FWHM (in pixels):')
#         for f, fwhm in odd_fwhm_list:
#             print('    ' + f + ' has unusual FWHM of ' + '{0:.2f}'.format(fwhm) + ' pixels.')
#         print('\n')
#     else:
#         print('All FWHM values seem OK.')
#     n_warnings += len(odd_fwhm_list)
#
#     odd_fl_list = []
#     mean_fl = df['FocalLength'].mean()
#     for f in df['Filename']:
#         fl = df.loc[f, 'FocalLength']
#         focus_length_pct_deviation = 100.0 * abs((fl - mean_fl)) / mean_fl
#         if focus_length_pct_deviation > FOCUS_LENGTH_MAX_PCT_DEVIATION:
#             odd_fl_list.append((f, fl))
#     if len(odd_fl_list) >= 1:
#         print('\nUnusual FocalLength (vs mean of ' + '{0:.1f}'.format(mean_fl) + ' mm:')
#         for f, fl in odd_fl_list:
#             print('    ' + f + ' has unusual Focal length of ' + str(fl))
#         print('\n')
#     else:
#         print('All Focal Lengths seem OK.')
#     n_warnings += len(odd_fl_list)
#
#     # Summarize and write instructions for next steps:
#     if n_warnings == 0:
#         print('\n >>>>> ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.')
#         print('Next: make_df_mp_master()')
#         log_file.write('assess(): ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.' + '\n')
#     else:
#         print('\n >>>>> ' + str(n_warnings) + ' warnings (see listing above).')
#         print('        Correct these and rerun assess() until no warnings remain.')
#         log_file.write('assess(): ' + str(n_warnings) + ' warnings.' + '\n')
#     log_file.close()


def make_df_mp_master():
    """  Main first stage in new MP photometry workflow. Mostly calls other subfunctions.
         Check files, extract standards data from photrix df_mp_master, do raw photometry on MP & comps
             to make df_mp_master.
    :return: df_mp_master: master MP raw photometry data [pandas Dataframe].
    """
    this_directory, mp_string, an_string, photrix_directory = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file.write('\n ===== make_df_mp_master()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Get all relevant FITS filenames, make lists of FITS and Image objects (per photrix):
    fits_names = get_fits_filenames(this_directory)
    fits_list = [FITS(this_directory, '', fits_name) for fits_name in fits_names]
    image_list = [Image(fits_object) for fits_object in fits_list]

    # From first FITS object, get coordinates for comp acquisition:
    deg_ra = fits_list[0].ra
    deg_dec = fits_list[0].dec

    # From first image object, get radius for comp acquisition:
    ps = fits_list[0].plate_solution  # a pandas Series
    # TODO: Is the next line right? [why is CD2_1 repeated?]
    plate_scale = (sqrt(ps['CD1_1']**2 + ps['CD2_1']**2) +
                   sqrt(ps['CD2_1']**2 + ps['CD2_2']**2)) / 2.0  # in degrees per pixel
    pixels_to_corner = sqrt(image_list[0].xsize**2 + image_list[0].ysize**2) / 2.0  # from center to corner
    degrees_to_corner = plate_scale * pixels_to_corner

    # Get all APASS 10 comps that might fall within images:
    comp_search_radius = 1.05*degrees_to_corner
    df_comps = get_apass10_comps(deg_ra, deg_dec, comp_search_radius,
                                 r_min=R_ESTIMATE_MIN, r_max=R_ESTIMATE_MAX, mp_color_only=True)
    df_comps['Type'] = 'Comp'
    print(str(len(df_comps)), 'qualifying APASS10 comps found within',
          '{:.3f}'.format(comp_search_radius) + u'\N{DEGREE SIGN}' + ' of',
          ra_as_hours(deg_ra) + 'h', degrees_as_hex(deg_dec) + u'\N{DEGREE SIGN}')
    df_comps['ID'] = [str(id) for id in df_comps['ID']]

    # Make df_comps_and_mp by adding MP row to df_comps:
    # THIS is where we enforce only one MP for this df_mp_master (and for everything derived from it).
    dict_mp_row = dict()
    for colname in df_comps.columns:
        dict_mp_row[colname] = [None]
    dict_mp_row['ID'] = ['MP_' + mp_string]
    dict_mp_row['Type'] = ['MP']
    df_mp_row = pd.DataFrame(dict_mp_row, index=dict_mp_row['ID'])
    df_comps_and_mp = pd.concat([df_mp_row, df_comps])
    del df_comps, df_mp_row  # as they are now obsolete.

    # Move some of df_comps_and_mp's columns to its left, leave the rest in original order:
    left_columns = ['ID', 'Type', 'degRA', 'degDec', 'R_estimate', 'e_R_estimate']
    df_comps_and_mp = reorder_df_columns(df_comps_and_mp, left_column_list=left_columns)

    # Add all comp apertures to every image:
    print(str(len(image_list)), 'images:')
    for image in image_list:
        for comp_id, type, ra, dec in zip(df_comps_and_mp['ID'], df_comps_and_mp['Type'],
                                          df_comps_and_mp['degRA'], df_comps_and_mp['degDec']):
            if type == 'Comp':
                x0, y0 = image.fits.xy_from_radec(RaDec(ra, dec))
                image.add_aperture(comp_id, x0, y0)
        print(image.fits.filename + ':', str(len(image.apertures)), 'apertures.')

    # Get time range of all MP images:
    utc_mids = [i.fits.utc_mid for i in image_list]
    min_session_utc = min(utc_mids)
    max_session_utc = max(utc_mids)
    mid_session_utc = min_session_utc + (max_session_utc - min_session_utc) / 2

    # Get MP ra,dec and motion from MPC page:
    print("Get and parse MPC page for", mp_string, "on", an_string, '...')
    # TODO: could instead get utc0, ra0, and dec0 from early-image and late-image RA,Dec supplied by user.
    utc_string = '{0:04d}{1:02d}{2:02d}'.format(mid_session_utc.year,
                                                mid_session_utc.month, mid_session_utc.day)
    x = get_one_html_from_list([mp_int], utc_string)
    mp_dict = extract_mp_data(x, chop_html(x)[0], True)
    # mp_dict = convert_mp_dict_to_numerics(mp_dict)
    utc_split = mp_dict['utc'].split(' ')
    utc0 = datetime(year=int(utc_split[0]), month=int(utc_split[1]), day=int(utc_split[2]),
                    hour=int(utc_split[3][0:2]), minute=int(utc_split[3][2:4]),
                    second=int(utc_split[3][4:6])).replace(tzinfo=timezone.utc)
    ra0 = ra_as_degrees(mp_dict['ra'])
    dec0 = dec_as_degrees(mp_dict['dec'])
    motion = float(mp_dict['motion']) / 60.0 / 3600.0  # in degrees per clock second
    motion_pa = float(mp_dict['motion_pa'])  # in degrees from north starting east
    ra_per_second = motion * sin(motion_pa / DEGREES_PER_RADIAN) / cos(dec0 / DEGREES_PER_RADIAN)
    dec_per_second = motion * cos(motion_pa / DEGREES_PER_RADIAN)

    # Locate and add MP aperture to each image:
    #   NB: the MP's (RA, Dec) sky location changes from image to image, unlike the comps.
    print("Add MP aperture...")
    mp_id = 'MP_' + mp_string
    mp_radec_dict = dict()
    for image in image_list:
        dt = (image.fits.utc_mid - utc0).total_seconds()
        ra = ra0 + dt * ra_per_second
        dec = dec0 + dt * dec_per_second
        x0, y0 = image.fits.xy_from_radec(RaDec(ra, dec))
        image.add_aperture(mp_id, x0, y0)
        mp_radec_dict[image.fits.filename] = (ra, dec)  # in degrees; later inserted into df_mp_master.

    # Build df_master_list (to make df_mp_master):
    print("Build df_master_list...")
    df_ur = pd.read_csv(os.path.join(photrix_directory, 'Photometry', FILE_RENAMING_FILENAME),
                        sep=';', index_col='PhotrixName')
    df_one_image_list = []
    for image in image_list:
        fits_name = image.fits.filename
        # Make df_apertures:
        ap_list = []
        ap_names = [k for k in image.apertures.keys()]
        for ap_name in ap_names:
            ap_list.append(dict(image.results_from_aperture(ap_name)))
        df_apertures = pd.DataFrame(ap_list, index=ap_names)  # constructor: list of dicts
        df_apertures['ID'] = df_apertures.index
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
        n_apertures_raw = len(df_apertures)
        df_apertures = df_apertures.loc[df_apertures['net_flux'] > 0.0, :]
        n_apertures_kept = len(df_apertures)
        print(fits_name + ':', n_apertures_kept, 'of', n_apertures_raw, 'obs kept.')
        df_apertures['InstMag'] = -2.5 * np.log10(df_apertures['net_flux']) +\
            2.5 * log10(image.fits.exposure)
        df_apertures['InstMagSigma'] = (2.5 / log(10)) * \
                                       (df_apertures['net_flux_sigma'] /
                                        df_apertures['net_flux'])  # math verified 20170726.

        # Delete unneeded columns (NB: max_adu was Calibrated ADUs, whereas we will need Ur ADUs):
        df_apertures.drop(['n_disc_pixels', 'n_annulus_pixels', 'max_adu',
                           'net_flux', 'net_flux_sigma'],
                          axis=1, inplace=True)  # delete columns (m

        # For each aperture, add its max ADU from the original ("Ur", uncalibrated) FITS file:
        ur_filename = df_ur.loc[fits_name, 'UrName']
        df_apertures['UrFITSfile'] = ur_filename
        ur_image = Image.from_fits_path(photrix_directory, 'Ur', ur_filename)
        df_apertures['MaxADU_Ur'] = np.nan
        for star_id in df_apertures.index:
            ap = Aperture(ur_image, star_id,
                          df_apertures.loc[star_id, 'Xcentroid'],
                          df_apertures.loc[star_id, 'Ycentroid'],
                          df_punches=None)
            df_apertures.loc[star_id, 'MaxADU_Ur'] = ap.max_adu

        # Get and merge in catalog star data from df_comps:
        df_one_image = pd.merge(df_apertures, df_comps_and_mp, how='left', on='ID')
        df_one_image.index = df_one_image['ID']  # just to be sure.

        # Add image-specific data to all rows of df_one_image:
        df_one_image['Filename'] = image.fits.filename
        df_one_image['Exposure'] = image.fits.exposure
        df_one_image['UTC_mid'] = image.fits.utc_mid
        df_one_image['JD_mid'] = jd_from_datetime_utc(image.fits.utc_mid)
        df_one_image['Filter'] = image.fits.filter
        df_one_image['Airmass'] = image.fits.airmass

        # Write MP's expected RA, Dec for this image into MP row:
        df_one_image.loc[mp_id, 'degRA'] = mp_radec_dict[image.fits.filename][0]
        df_one_image.loc[mp_id, 'degDec'] = mp_radec_dict[image.fits.filename][1]

        # Append this image's dataframe to a list, write line to console:
        df_one_image_list.append(df_one_image)

    print("Construct df_mp_master from lists...")
    df_mp_master = pd.DataFrame(pd.concat(df_one_image_list, ignore_index=True))
    df_mp_master.sort_values(['JD_mid', 'ID'], inplace=True)
    df_mp_master.insert(0, 'Serial', range(1, 1 + len(df_mp_master)))  # inserts in place
    obs_id_list = [f + '_' + id for f, id in zip(df_mp_master['Filename'], df_mp_master['ID'])]
    df_mp_master.insert(0, 'ObsID', obs_id_list)
    df_mp_master.index = list(df_mp_master['ObsID'])
    df_mp_master['Type'] = ['MP' if id == mp_id else 'Comp' for id in df_mp_master['ID']]

    # Fill in the JD_fract column:
    jd_floor = floor(df_mp_master['JD_mid'].min())  # requires that all JD_mid values be known.
    df_mp_master['JD_fract'] = df_mp_master['JD_mid'] - jd_floor

    # Move some of the columns to the Dataframe's left, leave the rest in original order:
    left_columns = ['ObsID', 'ID', 'Type', 'R_estimate', 'Serial', 'JD_mid',
                    'InstMag', 'InstMagSigma', 'Exposure']
    df_mp_master = reorder_df_columns(df_mp_master, left_column_list=left_columns)

    # Identify all comp and all MP observations, and report counts:
    is_comp = pd.Series([type == 'Comp' for type in df_mp_master['Type']])
    is_mp = ~is_comp
    print('\nStarting with', str(sum(is_comp)), 'comp obs,', str(sum(is_mp)), 'MP obs:')
    log_file.write('Starting with ' + str(sum(is_comp)) + ' comp obs, ' + str(sum(is_mp)) + ' MP obs:\n')

    # Identify obviously bad observations (dataframe rows), comps and MPs together:
    is_saturated = pd.Series([adu > ADU_UR_SATURATED for adu in df_mp_master['MaxADU_Ur']])
    is_low_snr = pd.Series([ims > (1.0 / COMPS_MIN_SNR) for ims in df_mp_master['InstMagSigma']])

    # Identify comp subset of bad observations, and report counts:
    is_saturated_comp_obs = is_saturated & is_comp
    is_low_snr_comp_obs = is_low_snr & is_comp
    is_bad_comp_obs = is_saturated_comp_obs | is_low_snr_comp_obs
    print('   Comp obs: removing',
          str(sum(is_saturated_comp_obs)), 'saturated, ',
          str(sum(is_low_snr_comp_obs)), 'low SNR.')
    log_file.write('   Comp obs: removing ' +
                   str(sum(is_saturated_comp_obs)) + ' saturated, ' +
                   str(sum(is_low_snr_comp_obs)) + ' low SNR.\n')

    # Identify MP subset of bad observations, and report counts:
    is_saturated_mp_obs = is_saturated & is_mp
    is_low_snr_mp_obs = is_low_snr & is_mp
    is_bad_mp_obs = is_saturated_mp_obs | is_low_snr_mp_obs
    print('     MP obs: removing',
          str(sum(is_saturated_mp_obs)), 'saturated, ',
          str(sum(is_low_snr_mp_obs)), 'low SNR.')
    log_file.write('     MP obs: removing ' +
                   str(sum(is_saturated_mp_obs)) + ' saturated, ' +
                   str(sum(is_low_snr_mp_obs)) + ' low SNR.\n')

    # Remove all identified bad observations:
    to_remove = is_bad_comp_obs | is_bad_mp_obs
    to_keep = list(~to_remove)
    df_mp_master = df_mp_master.loc[to_keep, :].copy()

    # Write df_comps_and_mp to file (rather than returning the df):
    fullpath_comps_and_mp = os.path.join(this_directory, DF_COMPS_AND_MPS_FILENAME)
    df_comps_and_mp.to_csv(fullpath_comps_and_mp, sep=';', quotechar='"',
                           quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('df_comps_and_mp written to', fullpath_comps_and_mp)

    # Write df_master to file (rather than returning the df):
    fullpath_master = os.path.join(this_directory, DF_MP_MASTER_FILENAME)
    df_mp_master.to_csv(fullpath_master, sep=';', quotechar='"',
                        quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('df_mp_master written to', fullpath_master)

    # Write initial comp_limits.txt (implying defaults), without overwriting file if it already exists:
    fullpath = os.path.join(this_directory, 'comp_limits.txt')
    if not os.path.exists(fullpath):
        lines = [';----- This is comp_limits.txt for directory ' + this_directory,
                 ';----- Use to limit comp stars used in qualify_comps() ' +
                 'and its downstream workflow steps.',
                 ';',
                 '#R   None None ; min_R_mag max_R_mag',
                 '#V-I None None ; min_color max_color',
                 '#E(R)MAX  None ; maximum allowed uncertainty in observed comp R magnitude',
                 ';',
                 '#OMIT   ; ids of comps to omit, space- or comma-separated, as many lines as needed']
        lines = [line + '\n' for line in lines]
        with open(fullpath, 'w') as f:
            f.writelines(lines)
            print('New comp_limits.txt written.')

    # Write log file lines:
    n_comps = sum([t.lower() == 'comp' for t in df_comps_and_mp['Type']])
    n_mps = sum([t.lower() == 'mp' for t in df_comps_and_mp['Type']])
    n_comp_obs = sum([t.lower() == 'comp' for t in df_mp_master['Type']])
    n_mp_obs = sum([t.lower() == 'mp' for t in df_mp_master['Type']])
    log_file.write('Keeping ' + str(n_comps) + ' comps, ' + str(n_mps) + ' mps.\n')
    log_file.write('Keeping ' + str(n_comp_obs) + ' comp obs, ' + str(n_mp_obs) + ' MP obs.\n')
    log_file.write(fullpath_comps_and_mp + ' written.\n')
    log_file.write(fullpath_master + ' written.\n')
    log_file.close()


def qualify_comps():
    """  Prepare all data for comp stars, using instrumental magnitudes and other comp data.
         Keep only qualified comp stars, esp. qualifying comps as being observed in every MP image.
    From make_df_master(), use:df_mp_master (one row per observation) and
                               df_comps_and_mp (one row per comp and minor planet).
    :return: [None] Write df_qual_comps_and_mp (updated df_comps_and_mp) to file,
                    write df_qual_obs (updated df_mp_master) to file.
    """
    this_directory, mp_string, an_string, photrix_directory = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    log_file.write('\n ===== qualify_comps()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Load data:
    df_comps_and_mp = get_df_comps_and_mp()
    df_mp_master = get_df_mp_master()
    state = get_session_state()  # for extinction and transform values.
    comp_selection_dict = get_comp_selection()

    # Local (nested) function:
    def write_counts(tag, df_comps_and_mps_input, df_obs_input):
        n_comps_input = sum(df_comps_and_mps_input['Type'] == 'Comp')
        n_comp_obs_input = sum(df_obs_input['Type'] == 'Comp')
        n_mps_input = sum(df_comps_and_mps_input['Type'] == 'MP')
        n_mp_obs_input = sum(df_obs_input['Type'] == 'MP')
        print(tag + ':')
        print('   Comps: ' + str(n_comps_input) + ' with ' + str(n_comp_obs_input) + ' obs.')
        print('   MPs  : ' + str(n_mps_input) + ' with ' + str(n_mp_obs_input) + ' obs.')
        log_file.write(tag + ':\n')
        log_file.write('   Comps: ' + str(n_comps_input) + ' with ' + str(n_comp_obs_input) + ' obs.\n')
        log_file.write('   MPs  : ' + str(n_mps_input) + ' with ' + str(n_mp_obs_input) + ' obs.\n')

    # Write out initial counts:
    write_counts('INPUT', df_comps_and_mp, df_mp_master)

    # Keep only comps, MPs, and their obs for IDs with obs in every image:
    df_semiqual_comps_and_mp, df_semiqual_obs = keep_omnipresent_comps(df_comps_and_mp, df_mp_master)
    write_counts('SEMIQUALIFIED', df_semiqual_comps_and_mp, df_semiqual_obs)
    del df_comps_and_mp, df_mp_master  # Now obsolete; thus raise error if access attempted.

    # Make df_vri (local to this function), with one row per MP-field image in V, R, or I.
    is_vri = list([filter in ['V', 'R', 'I'] for filter in df_semiqual_obs['Filter']])
    vri_filenames = df_semiqual_obs.loc[is_vri, 'Filename'].copy().drop_duplicates()
    vri_obs_ids = list(vri_filenames.index)
    df_vri = pd.DataFrame({'Serial': vri_obs_ids}, index=vri_obs_ids)
    df_vri['Filename'] = vri_filenames
    df_vri['JD_mid'] = list(df_semiqual_obs.loc[vri_obs_ids, 'JD_mid'])
    df_vri['UTC_mid'] = list(df_semiqual_obs.loc[vri_obs_ids, 'UTC_mid'])
    df_vri['Filter'] = list(df_semiqual_obs.loc[vri_obs_ids, 'Filter'])
    df_vri['Airmass'] = list(df_semiqual_obs.loc[vri_obs_ids, 'Airmass'])
    print(str(len(df_vri)), 'VRI images.')

    # Calculate zero-point "Z" value for each VRI image, add to df_vri:
    df_ext_stds = make_df_ext_stds()  # stds data read from photrix:df_master.
    z_curves = make_z_curves(df_ext_stds)  # each dict entry is 'filter': ZCurve object.
    z_values = [z_curves[filter].at_jd(jd) for filter, jd in zip(df_vri['Filter'], df_vri['JD_mid'])]
    df_vri['Z'] = z_values  # new column.

    # From df_vri (VRI images), calc best untransformed mag for each MP and comp, write to df_comps_and_mp:
    # Columns are named 'Best_untr_mag_V', etc.
    id_list = df_semiqual_obs['ID'].copy().drop_duplicates()
    for filter in ['V', 'R', 'I']:
        new_column_name = 'Best_untr_mag_' + filter  # for new column in df_comps_and_mp.
        df_semiqual_comps_and_mp[new_column_name] = None
        extinction = state['extinction'][filter]
        untransformed_mag_dict = dict()
        for id in id_list:
            untransformed_mag_dict[id] = []
        # Get each VRI image's untransformed mag in this filter:
        for idx in df_vri.index:
            if df_vri.loc[idx, 'Filter'] == filter:
                zero_point = df_vri.loc[idx, 'Z']
                airmass = df_vri.loc[idx, 'Airmass']
                for id in id_list:
                    filename = df_vri.loc[idx, 'Filename']
                    obs_id = filename + '_' + id
                    instrumental_mag = df_semiqual_obs.loc[obs_id, 'InstMag']
                    untransformed_mag = instrumental_mag - extinction * airmass - zero_point
                    untransformed_mag_dict[id].append(untransformed_mag)
        # Get averaged untransformed mags in this filter:
        for id in id_list:
            best_untransformed_mag = sum(untransformed_mag_dict[id]) / len(untransformed_mag_dict[id])
            df_semiqual_comps_and_mp.loc[id, new_column_name] = best_untransformed_mag

    # Solve for best (transformed) V-I color index for each comp and MP, write to dataframe:
    df_semiqual_comps_and_mp['Best_CI'] = None
    v_transform = state['transform']['V']
    i_transform = state['transform']['I']
    for id in id_list:
        best_color_index = (df_semiqual_comps_and_mp.loc[id, 'Best_untr_mag_V'] -
                            df_semiqual_comps_and_mp.loc[id, 'Best_untr_mag_I']) /\
                           (1.0 + v_transform - i_transform)
        df_semiqual_comps_and_mp.loc[id, 'Best_CI'] = best_color_index

    # Generate best *transformed* R magnitude for every comp and MP:
    df_semiqual_comps_and_mp['Best_R_mag'] = None
    r_transform = state['transform']['R']
    for id in id_list:
        best_r_mag = df_semiqual_comps_and_mp.loc[id, 'Best_untr_mag_R'] - \
                     r_transform * df_semiqual_comps_and_mp.loc[id, 'Best_CI']
        df_semiqual_comps_and_mp.loc[id, 'Best_R_mag'] = best_r_mag

    # Screen comp obs with user's comp-screening criteria:
    is_comp = df_semiqual_obs['Type'] == 'Comp'
    comp_r_above_min = df_semiqual_obs['Best_R_mag'] >= comp_selection_dict['min_r']
    comp_r_below_max = df_semiqual_obs['Best_R_mag'] <= comp_selection_dict['max_r']
    comp_color_above_min = df_semiqual_obs['Best_CI'] >= comp_selection_dict['min_vi']
    comp_color_below_max = df_semiqual_obs['Best_CI'] <= comp_selection_dict['max_vi']
    comp_r_uncertainty_ok = df_semiqual_obs['InstrMagSigma'] <= comp_selection_dict['max_er']
    comp_omitted = df_semiqual_obs['ID'].isin(comp_selection_dict['ids_to_omit'])
    comp_is_ok = comp_r_above_min & comp_r_below_max & \
        comp_color_above_min & comp_color_below_max & \
        comp_r_uncertainty_ok & (~comp_omitted)
    to_keep = ~is_comp | comp_is_ok  # we keep only: all mps + qualified comps.
    df_screened_obs = df_semiqual_obs[to_keep].copy()
    write_counts('SCREENED', df_semiqual_comps_and_mp, df_screened_obs)
    del df_semiqual_obs  # Now obsolete; thus, raise error if access attempted.

    # Finally, keep only comp and MP IDs that still have qualified data in *every* image:
    df_qual_comps_and_mp, df_qual_obs = keep_omnipresent_comps(df_semiqual_comps_and_mp, df_screened_obs)
    write_counts('QUALIFIED (final)', df_qual_comps_and_mp, df_qual_obs)
    del df_semiqual_comps_and_mp, df_screened_obs  # Now obsolete; thus, raise error if access attempted.

    # Reorder columns in df_qual_comps_and_mp, then write it file (ratner than returning it):
    df_qual_comps_and_mp = reorder_df_columns(df_qual_comps_and_mp,
                                              left_column_list=['ID', 'Type', 'Best_R_mag', 'Best_CI'])
    fullpath_qual_comps_and_mps = os.path.join(this_directory, DF_QUALIFIED_COMPS_AND_MPS_FILENAME)
    df_qual_comps_and_mp.to_csv(fullpath_qual_comps_and_mps, sep=';', quotechar='"',
                                quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('New df_qual_comps_and_mp written to', fullpath_qual_comps_and_mps)

    # Reorder columns in df_qual_obs, then write it to file (rather than returning it):
    df_qual_obs = reorder_df_columns(df_qual_obs, left_column_list=['ObsID', 'ID', 'Type'])
    fullpath_qual_obs = os.path.join(this_directory, DF_QUALIFIED_OBS_FILENAME)
    df_qual_obs.to_csv(fullpath_qual_obs, sep=';', quotechar='"',
                       quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('New df_qual_obs written to', fullpath_qual_obs)

    log_file.close()
    plot_comps()  # plot exactly the comp set produced by qualify_comps()):


def plot_comps():
    """ [Typically not run directly by user; rather typically run from within (at end of) qualify_comps()]
        Make plots supporting user's final selection of comp stars for this MP session.
    :return: [None]
    """
    this_directory, mp_string, an_string, photrix_directory = get_context()
    df_qual_comps_and_mp = get_df_qual_comps_and_mp()
    df_qual_obs = get_df_qual_obs()
    df_qual_comps_only = df_qual_comps_and_mp[df_qual_comps_and_mp['Type'] == 'Comp']
    df_mp_only = df_qual_comps_and_mp[df_qual_comps_and_mp['Type'] == 'MP']
    mp_color = 'darkred'
    comp_color = 'black'

    # FIGURE 1: (multiplot): One point per comp star, plots in a grid within one figure (page).
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(14, 9))  # (width, height) in "inches"

    # Local support function for *this* grid of plots:
    def make_labels(ax, title, xlabel, ylabel, zero_line=False):
        ax.set_title(title, y=0.91)
        ax.set_xlabel(xlabel, labelpad=0)
        ax.set_ylabel(ylabel, labelpad=0)
        if zero_line is True:
            ax.axhline(y=0, color='lightgray', linewidth=1, zorder=-100)

    # "Canopus comp plot" (R InstMag from VRI image *vs* R mag estimate from catalog):
    # Would ideally be a straight line; inspect for outliers.
    # Data from df_mp_master, comp stars only.
    ax = axes[0, 0]  # at upper left.
    make_labels(ax, 'Canopus (CSS) comp plot', 'R_estimate from catalog B & V', 'InstMag(R) from VRI')
    is_instmag_r = df_qual_obs['Filter'] == 'R'
    is_comp = df_qual_obs['Type'] == 'Comp'
    use_in_plot = is_instmag_r & is_comp
    x = df_qual_obs.loc[use_in_plot, 'R_estimate']
    y = df_qual_obs.loc[use_in_plot, 'InstMag']
    ax.scatter(x=x, y=y, alpha=0.5, color=comp_color)
    margin = 0.08
    x_range = max(x) - min(x)
    # y_range = max(y) - min(y)
    x_low, x_high = min(x) - margin * x_range, max(x) + margin * x_range
    intercept = sum(y - x) / len(x)
    y_low, y_high = x_low + intercept, x_high + intercept
    # y_low, y_high = min(y) - margin * y_range, max(y) + margin * y_range
    ax.plot([x_low, x_high], [y_low, y_high], color='black', zorder=-100, linewidth=1)

    # "R-shift plot" (Best R mag - R estimate from catalog vs R estimate from catalog):
    # A large shift suggests comp-star non-blackbody spectrum.
    # Data from df_qual_comps_only, thus comp stars (no MP).
    ax = axes[0, 1]  # next plot to the right.
    make_labels(ax, 'R-shift comp plot', 'R_estimate from catalog B & V',
                'Best R mag - first R_estimate', zero_line=True)
    ax.scatter(x=df_qual_comps_only['R_estimate'],
               y=df_qual_comps_only['Best_R_mag'] - df_qual_comps_only['R_estimate'],
               alpha=0.5, color=comp_color)

    # Observed Error Plot (R mag):
    # Inspect for pattern and for large uncertainties (should be capped at ERR_R_ESTIMATE_MAX):
    # Comp stars only.
    ax = axes[0, 2]
    make_labels(ax, 'Instrumental Sigma', 'Best R mag', 'Sigma(Rmag)')
    is_r_obs = df_qual_obs['Filter'] == 'R'
    df_plot = df_qual_obs.loc[is_r_obs, ['ID', 'InstMagSigma']].copy()
    df_plot = pd.merge(df_plot, df_qual_comps_and_mp[['ID', 'Best_R_mag']],
                       how='left', on='ID', sort=False)
    ax.scatter(x=df_plot['Best_R_mag'],
               y=df_plot['InstMagSigma'],
               alpha=0.5, color=comp_color)

    # Color index plot (suitability of comps relative to MP):
    # Try to choose comp stars near or to left of MP.
    # Comp stars and MP together.
    ax = axes[1, 0]
    make_labels(ax, 'Color Index plot', 'Best R mag', 'Best CI (V-I)')
    ax.scatter(x=df_qual_comps_only['Best_R_mag'],
               y=df_qual_comps_only['Best_CI'],
               alpha=0.5, color=comp_color)
    ax.scatter(x=df_mp_only['Best_R_mag'],
               y=df_mp_only['Best_CI'],
               alpha=1.0, color=mp_color, marker='s', s=96)
    # ax.axhline(y=COLOR_VI_VARIABLE_RISK, color='red', linewidth=1, zorder=-100)

    # Color index plot:
    # Inspect for pattern and for large uncertainties (should be capped at ERR_R_ESTIMATE_MAX):
    # Comp stars only.
    ax = axes[1, 1]
    make_labels(ax, 'R-estimate error', 'Best R mag', 'uncert(R estimate)')
    ax.scatter(x=df_qual_comps_only['Best_R_mag'],
               y=df_qual_comps_only['e_R_estimate'],
               alpha=0.5, color=comp_color)

    # Finish the figure, and show the entire plot:
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.25)
    fig.suptitle('Comp Star Diagnostics    ::     MP_' + mp_string + '   AN' + an_string,
                 color='darkblue', fontsize=20, weight='bold')
    fig.canvas.set_window_title('Comp star diagnostic plots')
    plt.show()

    # FIGURE 2: Multi-plot pages, one plot of each comp's deviations from aggregate of other comps.
    # TODO: Compute and plot each comp's deviations in behavior from median of other comps' (how?).




    # Write diagnostics to console:
    # (or, these data may better go into the log file)



def finish():
    """  From df_qual_obs, calc best MP R mags using comp R mags,
         then make 'Custom' text file suitable for import to Canopus 10.
    :return: [None] Write results into formatted canopus.txt.
    """
    this_directory, mp_string, an_string, _ = get_context()
    mp_id = 'MP_' + mp_string
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    log_file.write('\n ===== finish()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # Get data:
    df_qual_obs = get_df_qual_obs()
    df_clear_obs = df_qual_obs.loc[df_qual_obs['Filter'] == 'Clear', :]  # omit VRI images.
    state = get_session_state()  # for extinction and transform values.
    transform = state['transform']['Clear']

    # For each MP observation, compute best MP R mag, write into a dictionary:
    output_dict_list = []
    image_filenames = df_clear_obs['Filename'].copy().drop_duplicates()
    for image_filename in image_filenames:
        mp_obs_id = image_filename + '_' + mp_id
        of_this_image = df_clear_obs['Filename'] == image_filename
        df_image = df_clear_obs.loc[of_this_image, :]
        is_comp_obs = df_image['Type'] == 'Comp'

        # Compute MP's best R magnitude for this image:
        mean_comp_best_r = mean(df_image.loc[is_comp_obs, 'Best_R_mag'])
        mean_comp_best_vi = mean(df_image.loc[is_comp_obs, 'Best_VI'])
        mean_comp_instmag = mean(df_image.loc[is_comp_obs, 'InstMag'])  # Clear filter.
        mp_best_vi = df_image.loc['MP_' + mp_string, 'Best_VI']
        mp_instmag = df_image.loc['MP_' + mp_string, 'InstMag']  # Clear filter.
        mp_best_r = mp_instmag - mean_comp_instmag + mean_comp_best_r + \
            transform * (mean_comp_best_vi - mp_best_vi)

        # Construct output_dict entry:
        output_dict = dict()
        output_dict['JD'] = df_image.loc[mp_obs_id, 'JD_mid']
        output_dict['Best_R_mag'] = mp_best_r
        output_dict_list.append(output_dict)

    df_output = pd.DataFrame(data=output_dict_list, index=image_filenames).sort_values(by='JD')

    # Write output file:
    lines = []
    for image_filename in image_filenames:
        jd_string = '{:15.6f}'.format(df_output.loc[image_filename, 'JD'])
        r_mag = '{:8.3f}'.format(df_output.loc[image_filename, 'Best_R_mag'])
        lines.append(jd_string + '\t' + r_mag + '\n')
    filename = mp_id + '_' + an_string + '.txt'
    fullpath = os.path.join(this_directory, filename)
    with open(fullpath, 'w') as f:
        f.writelines(lines)
    print('Output file ' + fullpath + '  written.')
    log_file.write('Output file ' + fullpath + '  written.\n')
    log_file.close()


SUPPORT_______________________________________________________ = 0


# def get_context():
#     """ This is run at beginning of workflow functions (except start()) to orient the function.
#     :return: 4-tuple: (this_directory, mp_string, an_string, photrix_directory) [3 strings]
#     """
#     this_directory = os.getcwd()
#     if not os.path.isfile(LOG_FILENAME):
#         return None
#     log_file = open(LOG_FILENAME, mode='r')  # for read only
#     lines = log_file.readlines()
#     log_file.close()
#     if len(lines) < 3:
#         return None
#     if lines[0].strip().lower().replace('\\', '/').replace('//', '/') != \
#             this_directory.strip().lower().replace('\\', '/').replace('//', '/'):
#         print('Working directory does not match directory at top of log file.')
#         return None
#     mp_string = lines[1][3:].strip().upper()
#     an_string = lines[2][3:].strip()
#     photrix_directory = lines[3][8:].strip()
#     return this_directory, mp_string, an_string, photrix_directory
#
#
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
#
#
# def get_fits_filenames(directory):
#     all_filenames = pd.Series([e.name for e in os.scandir(directory) if e.is_file()])
#     extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in all_filenames])
#     is_fits = [ext.lower() in VALID_FITS_FILE_EXTENSIONS for ext in extensions]
#     fits_filenames = all_filenames[is_fits]
#     return fits_filenames


def make_df_ext_stds():
    """ Renders a dataframe with all external standards info, taken from prev made photrix df_master.
        Pre-filters the standards to color indices not too red (for safety from variable stars).
        These external standard serve only to set zero-point (curves) for later processing.
    :return: dataframe of external standards data [pandas Dataframe].
    """
    _, _, _, photrix_directory = get_context()
    df = get_df_master(os.path.join(photrix_directory, 'Photometry'))
    is_std = [id.lower().startswith('std_') for id in df['ModelStarID']]
    df_ext_stds = df[is_std]
    not_too_red = (df_ext_stds['CI'] <= 1.4)
    df_ext_stds = df_ext_stds[not_too_red]
    return df_ext_stds


def make_z_curves(df_ext_stds):
    """  Make and return dict of ZCurve objects, one per filter in external standards (i.e., V, R, and I).
    :param df_ext_stds: all standard-field df_master rows from df_master (photrix), from make_df_ext_stds().
    :return: dict of ZCurve objects, one per filter [dict of ZCurve objects].
    Access a filter's ZCurve object via, e.g.: z_curve_v = z_curves['V']
    """
    std_filters = df_ext_stds['Filter'].copy().drop_duplicates()
    state_dict = get_session_state()
    z_curves = dict()
    for filter in std_filters:
        transform = state_dict['transform'][filter]
        extinction = state_dict['extinction'][filter]
        z_curves[filter] = ZCurve(filter, df_ext_stds, transform, extinction)
    return z_curves


# def get_df_mp_master():
#     """  Simple utility to read df_mp_master.csv file and return the original DataFrame.
#     :return: df_mp_master from mp_phot() [pandas Dataframe]
#     """
#     this_directory, _, _, _ = get_context()
#     fullpath = os.path.join(this_directory, DF_MP_MASTER_FILENAME)
#     df_mp_master = pd.read_csv(fullpath, sep=';', index_col=0)
#     return df_mp_master


def get_df_comps_and_mp():
    """  Simple utility to read df_comps.csv file and return the original DataFrame.
    :return: df_comps_and_mp from mp_phot() [pandas Dataframe]
    """
    this_directory, _, _, _ = get_context()
    fullpath = os.path.join(this_directory, DF_COMPS_AND_MPS_FILENAME)
    df_comps_and_mp = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_comps_and_mp


def get_df_qual_obs():
    """  Simple utility to read df_qual_obs.csv file and return the original DataFrame.
    :return: df_qual_obs from mp_phot() [pandas Dataframe]
    """
    this_directory, _, _, _ = get_context()
    fullpath = os.path.join(this_directory, DF_QUALIFIED_OBS_FILENAME)
    df_qual_obs = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_qual_obs


def get_df_qual_comps_and_mp():
    """  Simple utility to read df_qual_comps_and_mp.csv file and return the original DataFrame.
    :return: df_qual_comps_and_mp from mp_phot() [pandas Dataframe]
    """
    this_directory, _, _, _ = get_context()
    fullpath = os.path.join(this_directory, DF_QUALIFIED_COMPS_AND_MPS_FILENAME)
    df_qual_comps_and_mp = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_qual_comps_and_mp


class ZCurve:
    def __init__(self, filter, df_ext_stds, transform, extinction):
        is_in_filter = (df_ext_stds['Filter'] == filter)
        df_filter = df_ext_stds[is_in_filter].sort_values(by='JD_mid')
        self.jds = df_filter['JD_mid'].copy().drop_duplicates().sort_values().values  # all unique JDs.
        self.n = len(self.jds)
        # print(self.n, self.jds)

        # Calculate Z at each JD where it's needed (to parallel list self.jds):
        self.z_values_jd = []
        for jd in self.jds:
            is_jd = df_filter['JD_mid'] == jd
            df_jd = df_filter[is_jd]
            z_values = []
            for i in df_jd.index:
                airmass = df_jd.loc[i, 'Airmass']
                color_index = df_jd.loc[i, 'CI']
                corrections = extinction * airmass + transform * color_index
                instmag = df_jd.loc[i, 'InstMag']
                catalog_mag = df_jd.loc[i, 'CatMag']
                this_z_value = instmag - catalog_mag - corrections
                z_values.append(this_z_value)
            self.z_values_jd.append(median(z_values))
        # print(self.z_values_jd)

        # Set up appropriate interpolation for future calls to .at_jd():
        if self.n <= 0:
            pass
        elif self.n == 1:
            self.constant = self.z_values_jd[0]
        elif self.n in [2, 3]:
            # Linear interpolation:
            x = self.jds
            y = self.z_values_jd
            this_linear_fit = np.polyfit(x, y, 1)
            self.linear_function = np.poly1d(this_linear_fit)
        else:
            # Spline for 4 or more points:
            x = self.jds
            y = self.z_values_jd
            # weights = len(x) * [1.0]
            # smoothness = len(x) * 0.02 ** 2  # i.e, N * sigma**2
            # self.spline_function = UnivariateSpline(x=x, y=y, w=weights, s=smoothness, ext=3)
            self.spline_function = interp1d(x, y, kind='linear')

    def at_jd(self, jd):
        if self.n <= 0:
            return None
        # If JD is outside standards' JD range, return Z-value from nearest standard.
        if jd < self.jds[0]:
            return self.z_values_jd[0]
        elif jd > self.jds[-1]:
            return self.z_values_jd[-1]
        # Else interpolate in JD:
        if self.n == 1:
            return self.constant
        elif self.n in [2, 3]:
            return self.linear_function(jd)
        else:
            return self.spline_function(jd)


# def get_session_state(site_name='DSW', instrument_name='Borea'):
#     """ Return dict containing extinction & transforms (etc?).
#     :param site_name: name of site for class Site, e.g., 'DSW' [string].
#     :param instrument_name: name of instrument for class Instrument, e.g., 'Borea' [string].
#     :return: Session state data (dict of dicts)
#     Access an extinction via: state['extinction'][filter_string].
#     Access a filter=passband transform (color index V-I) via: state['transform'][filter_string].
#     """
#     from photrix.user import Site, Instrument
#     state = dict()
#     site = Site(site_name)
#     state['extinction'] = site.extinction
#     inst = Instrument(instrument_name)
#     transform_dict = dict()
#     for this_filter in ['V', 'R', 'I']:
#         transform_dict[this_filter] = inst.transform(this_filter, 'V-I')
#     state['transform'] = transform_dict
#     return state


def get_comp_selection():
    """ Reads file comp_selection.txt if exists, uses defaults if not.
    Returns dict of comp star limits on R magnitude, color index, errors, etc. & comps to omit.
    :return comp_selection_dict: comp limits and omissions [dict].
    """
    this_directory, _, _, _ = get_context()
    fullpath = os.path.join(this_directory, 'comp_limits.txt')
    file = open(fullpath, mode='r')  # for read only
    lines = file.readlines()
    file.close()
    comp_selection_dict = dict(min_r=COMPS_MIN_R_MAG, max_r=COMPS_MAX_R_MAG,
                               min_vi=COMPS_MIN_COLOR_VI, max_vi=COMPS_MAX_COLOR_VI,
                               max_er=1 / COMPS_MIN_SNR)
    for line in lines:
        content = line.upper().strip().split(';')[0].strip()  # upper case, comments removed.
        if content.startswith('#R'):
            values = content[len('#R'):].strip()
            min_r_string = values.strip().split()[0]
            if min_r_string != 'NONE':
                comp_selection_dict['min_r'] = float(min_r_string)
            max_r_string = values.strip().split()[1]
            if max_r_string != 'NONE':
                comp_selection_dict['max_r'] = float(max_r_string)
        if content.startswith('#V-I'):
            values = content[len('#V-I'):].strip()
            min_vi_string = values.strip().split()[0]
            if min_vi_string != 'NONE':
                comp_selection_dict['min_vi'] = float(min_vi_string)
            max_vi_string = values.strip().split()[1]
            if max_vi_string != 'NONE':
                comp_selection_dict['max_vi'] = float(max_vi_string)
        if content.startswith('#E(R)MAX'):
            values = content[len('#E(R)MAX'):].strip()
            max_er_string = values.strip().split()[0]
            if max_er_string != 'NONE':
                comp_selection_dict['max_er'] = float(max_er_string)
        if content.startswith('#OMIT'):
            values = content[len('#OMIT'):].strip().replace(',', ' ')
            comp_selection_dict['ids_to_omit'] = [id.strip() for id in values.split() if id.strip() != '']
    return comp_selection_dict


def keep_omnipresent_comps(df_comps_mps_input, df_obs_input):
    """  Take dataframes, keep only rows in each for (all MPs and) only comp IDs represented in every image.
    :param df_comps_mps_input: dataframe of sources, must have column 'ID'. [pandas Dataframe]
    :param df_obs_input: dataframe of observations, must have column 'ID'. [pandas Dataframe]
    :return: tuple (df_qualified_comps_mp, df_qualified_obs) [2-tuple of pandas Dataframes].
    """
    # Make list of comps that are MISSING an obs from at least one image in df_mp_master:
    is_comp = df_obs_input['Type'] == 'Comp'
    comp_ids = df_obs_input.loc[is_comp, 'ID']
    comp_id_list = comp_ids.copy().drop_duplicates()
    n_fitsfiles = len(df_obs_input['Filename'].drop_duplicates())
    incomplete_comp_id_list = []
    for this_id in comp_id_list:
        n_obs_this_id = sum([id == this_id for id in df_obs_input['ID']])
        if n_obs_this_id != n_fitsfiles:
            incomplete_comp_id_list.append(this_id)

    # Make and return new subset dataframes:
    obs_to_remove = df_obs_input['ID'].isin(incomplete_comp_id_list)
    obs_to_keep = ~obs_to_remove
    df_obs_output = df_obs_input[obs_to_keep].copy()
    comps_to_remove = df_comps_mps_input['ID'].isin(incomplete_comp_id_list)
    comps_mps_to_keep = ~comps_to_remove
    df_comps_mps_output = df_comps_mps_input[comps_mps_to_keep].copy()
    return df_comps_mps_output, df_obs_output





def make_packed_mp_string(mp_string):
    n = int(mp_string)
    if n <= 99999:
        return'{0:05d}'.format(n)
    div, mod = divmod(n, 10000)
    if 100000 <= n <= 359999:
        return chr(div - 9 + (ord('A') - 1)) + '{0:04d}'.format(mod)
    if 260000 <= n <= 619999:
        return chr(div - 9 - 26 + (ord('a') - 1)) + '{0:04d}'.format(mod)
    else:
        return 'XXXXX'
