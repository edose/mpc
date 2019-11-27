__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
from io import StringIO
import shutil
from datetime import datetime, timezone, timedelta
from math import cos, sin, sqrt, pi, log10, floor
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import UnivariateSpline, interp1d
from statistics import median
from mpc.mpctools import *
from photrix.image import Image, FITS, Aperture
from photrix.util import RaDec, jd_from_datetime_utc, degrees_as_hex, ra_as_hours

MAX_FWHM = 14  # in pixels.
MP_COMP_RADEC_ERROR_MAX = 1.0  # in arcseconds
# MP_COLOR_BV_MIN = 0.68  # defines color range for comps (a bit wider than normal MP color range).
MP_COLOR_BV_MIN = 0.2  # a different guess (more inclusive)
MP_COLOR_BV_MAX = 1.2  # "
COMPS_MIN_R_MAG = 10.5    # a guess
COMPS_MAX_R_MAG = 15.5  # a guess; probably needs to be customized per-MP, even per-session.
R_ESTIMATE_V_COEFF = 0.975  # from regression on Best_R_mag
R_ESTIMATE_BV_COLOR_COEFF = -0.419  # "
R_ESTIMATE_INTERCEPT = 0  # "
ERR_R_ESTIMATE_MAX = 0.25  # this can be fairly high (inclusive), as there will be more screens, later.
COMPS_MIN_SNR = 25  # min signal/noise for use comp obs (SNR defined here as InstMag / InstMagSigma).
ADU_UR_SATURATED = 54000  # This probably should go in Instrument class, but ok for now.
COLOR_VI_VARIABLE_RISK = 1.5  # greater VI values risk being a variable star, thus unsuitable as comp star.

DEGREES_PER_RADIAN = 180.0 / pi

PHOTRIX_TOP_DIRECTORY = 'C:/Astro/Borea Photrix/'
UR_LIST_PATH = 'Photometry/File-renaming.txt'

MP_TOP_DIRECTORY = 'J:/Astro/Images/MP Photometry/'
LOG_FILENAME = 'mp_photometry.log'
DF_MP_MASTER_FILENAME = 'df_mp_master.csv'
DF_COMPS_AND_MP_FILENAME = 'df_comps_and_mp.csv'
DF_QUALIFIED_OBS_FILENAME = 'df_qual_obs.csv'
DF_QUALIFIED_COMPS_AND_MP_FILENAME = 'df_qual_comps_and_mp.csv'


def canopus(mp_top_directory=MP_TOP_DIRECTORY, rel_directory=None):
    """ Read all FITS in mp_directory, rotate right, bin 2x2, invalidating plate solution.
    Intended for making images suitable (North Up, smaller) for Canopus 10.
    Tests OK ~20191101.
    :param mp_top_directory: top path for FITS files [string]
    :param rel_directory: rest of path to FITS files, e.g., 'MP_768/AN20191020' [string]
    : return: None
    """
    this_directory = os.path.join(mp_top_directory, rel_directory)
    # clean_subdirectory(this_directory, 'Canopus')
    # output_directory = os.path.join(this_directory, 'Canopus')
    output_directory = this_directory
    import win32com.client
    app = win32com.client.Dispatch('MaxIm.Application')
    count = 0
    for entry in os.scandir(this_directory):
        if entry.is_file():
            fullpath = os.path.join(this_directory, entry.name)
            doc = win32com.client.Dispatch('MaxIm.Document')
            doc.Openfile(fullpath)
            doc.RotateRight()  # Canopus requires North-up.
            doc.Bin(2)  # to fit into Canopus.
            doc.StretchMode = 2  # = High, the better to see MP.
            output_filename, output_ext = os.path.splitext(entry.name)
            output_fullpath = os.path.join(output_directory, output_filename + '_Canopus' + output_ext)
            doc.SaveFile(output_fullpath, 3, False, 3, False)  # FITS, no stretch, floats, no compression.
            doc.Close  # no parentheses is actually correct. (weird MaxIm API)
            count += 1
            print('*', end='', flush=True)
    print('\n' + str(count), 'converted FITS now in', output_directory)



NEW_WORKFLOW________________________________________________ = 0
# ***********************
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


def start(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Preliminaries to begin MP photometry workflow.
    :param mp_top_directory: path of lowest directory common to all MP photometry FITS, e.g.,
               'J:/Astro/Images/MP Photometry' [string]
    :param mp_number: number of target MP, e.g., 1602 for Indiana. [integer or string].
    :param an_string: Astronight string representation, e.g., '20191106' [string].
    :return: [None]
    """
    mp_int = int(mp_number)  # put this in try/catch block.
    mp_string = str(mp_int)
    mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    os.chdir(mp_directory)
    print('Working directory set to:', mp_directory)
    log_file = open(LOG_FILENAME, mode='w')  # new file; wipe old one out if it exists.
    log_file.write(mp_directory + '\n')
    log_file.write('MP: ' + mp_string + '\n')
    log_file.write('AN: ' + an_string + '\n')
    log_file.write('This log started:' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    print('Log file started.')
    print(' >>>>> Next:  assess().')
    log_file.close()


def assess():
    """  Perform checks on FITS files in this directory before performing the photometry proper.
    Modeled after assess() in variable-star photometry package 'photrix'.
    :return: [None]
    """
    this_directory, mp_string, an_string, log_file = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.

    # Get list of FITS file names & start dataframe df:
    all_filenames = pd.Series([e.name for e in os.scandir(this_directory) if e.is_file()])
    extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in all_filenames])
    is_fits = [ext.startswith('.f') for ext in extensions]
    # fits_filenames = all_filenames[is_fits]
    fits_extensions = extensions[is_fits]

    fits_filenames = [f for (f, ext) in zip(all_filenames, extensions) if ext.startswith('.f')]
    df = pd.DataFrame({'Filename': fits_filenames,
                       'Extension': fits_extensions}).sort_values(by=['Filename'])
    df = df.set_index('Filename', drop=False)
    df['Valid'] = False
    df['PlateSolved'] = False
    df['Calibrated'] = True
    df['FWHM'] = np.nan
    df['FocalLength'] = np.nan
    log_file.write('assess(): ' + str(len(fits_filenames)) + ' FITS files found.' + '\n')

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

    # Now assess all FITS, and report errors & warnings:
    not_platesolved = df.loc[~ df['PlateSolved'], 'Filename']
    if len(not_platesolved) >= 1:
        print('NO PLATE SOLUTION:')
        for f in not_platesolved:
            print('    ' + f)
        print('\n')
    else:
        print('All platesolved.')

    not_calibrated = df.loc[~ df['Calibrated'], 'Filename']
    if len(not_calibrated) >= 1:
        print('\nNOT CALIBRATED:')
        for f in not_calibrated:
            print('    ' + f)
        print('\n')
    else:
        print('All calibrated.')

    odd_fwhm_list = []
    for f in df['Filename']:
        fwhm = df.loc[f, 'FWHM']
        if fwhm < 1.5 or fwhm > MAX_FWHM:  # too small or large:
            odd_fwhm_list.append((f, fwhm))
    if len(odd_fwhm_list) >= 1:
        print('\nUnusual FWHM (in pixels):')
        for f, fwhm in odd_fwhm_list:
            print('    ' + f + ' has unusual FWHM of ' + '{0:.2f}'.format(fwhm) + ' pixels.')
        print('\n')
    else:
        print('All FWHM values seem OK.')

    odd_fl_list = []
    mean_fl = df['FocalLength'].mean()
    for f in df['Filename']:
        fl = df.loc[f, 'FocalLength']
        if abs((fl - mean_fl)) / mean_fl > 0.03:
            odd_fl_list.append((f, fl))
    if len(odd_fl_list) >= 1:
        print('\nUnusual FocalLength (vs mean of ' + '{0:.1f}'.format(mean_fl) + ' mm:')
        for f, fl in odd_fl_list:
            print('    ' + f + ' has unusual Focal length of ' + str(fl))
        print('\n')
    else:
        print('All Focal Lengths seem OK.')

    # Summarize and write instructions for next steps:
    n_warnings = len(not_calibrated) + len(not_platesolved) +\
                 len(odd_fwhm_list) + len(odd_fl_list) + len(invalid_fits)
    if n_warnings == 0:
        print('\n >>>>> ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.')
        print('Next: run make_df_mp_master().')
        log_file.write('assess(): ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.' + '\n')
    else:
        print('\n >>>>> ' + str(n_warnings) + ' warnings (see listing above).')
        print('        Correct these and rerun assess() until no warnings remain.')
        log_file.write('assess(): ' + str(n_warnings) + ' warnings.' + '\n')
    log_file.close()


def make_df_mp_master(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Main first stage in new MP photometry workflow. Mostly calls other subfunctions.
         Check files, extract standards data from photrix df_mp_master, do raw photometry on MP & comps
             to make df_mp_master.
    :param mp_top_directory: path of lowest directory common to all MP photometry FITS, e.g.,
               'J:/Astro/Images/MP Photometry' [string]
    :param mp_number: number of target MP, e.g., 1602 for Indiana. [integer or string].
    :param an_string: Astronight string representation, e.g., '20191106' [string].
    :return: df_mp_master: master MP raw photometry data [pandas Dataframe].
    """
    this_directory, mp_string, an_string, log_file = get_context()
    log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
    mp_int = int(mp_number)  # put this in try/catch block.
    mp_string = str(mp_int)

    # Get all relevant FITS filenames, make FITS and Image objects (photrix):
    mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    all_names = [entry.name for entry in os.scandir(mp_directory) if entry.is_file()]
    fits_names = [a for a in all_names if a.split('.')[1][0].lower() == 'f']
    fits_list = []
    image_list = []
    for fits_name in fits_names:
        # construct FITS and Image objects:
        fits_object = FITS(mp_top_directory, os.path.join('MP_' + mp_string, 'AN' + an_string), fits_name)
        image = Image(fits_object)
        fits_list.append(fits_object)
        image_list.append(image)

    # From first FITS object, get coordinates for comp acquisition:
    degRA = fits_list[0].ra
    degDec = fits_list[0].dec

    # From first image object, get radius for comp acquisition:
    ps = fits_list[0].plate_solution  # a pandas Series
    plate_scale = (sqrt(ps['CD1_1']**2 + ps['CD2_1']**2) +
                   sqrt(ps['CD2_1']**2 + ps['CD2_2']**2)) / 2.0  # in degrees per pixel
    pixels_to_corner = sqrt(image_list[0].xsize**2 + image_list[0].ysize**2) / 2.0  # from center to corner
    degrees_to_corner = plate_scale * pixels_to_corner

    # Get all APASS 10 comps that might fall within images:
    comp_search_radius = 1.05*degrees_to_corner
    df_comps = get_apass10_comps(degRA, degDec, comp_search_radius,
                                 r_min=COMPS_MIN_R_MAG, r_max=COMPS_MAX_R_MAG, mp_color_only=True)
    df_comps['Type'] = 'Comp'
    print(str(len(df_comps)), 'qualifying APASS10 comps found within',
          '{:.3f}'.format(comp_search_radius) + u'\N{DEGREE SIGN}' + ' of',
          ra_as_hours(degRA) + 'h', degrees_as_hex(degDec) + u'\N{DEGREE SIGN}')
    df_comps['ID'] = [str(id) for id in df_comps['ID']]

    # Make df_comps_and_mp by adding MP row to df_comps:
    dict_mp_row = dict()
    for colname in df_comps.columns:
        dict_mp_row[colname] = [None]
    dict_mp_row['ID'] = ['MP_' + mp_string]
    dict_mp_row['Type'] = ['MP']
    df_mp_row = pd.DataFrame(dict_mp_row, index=[dict_mp_row['ID']])
    df_comps_and_mp = pd.concat([df_mp_row, df_comps])
    del df_comps, df_mp_row  # obsolete.

    # Move some of df_comps_and_mp's columns to its left, leave the rest in original order:
    left_columns = ['ID', 'Type', 'degRA', 'degDec', 'R_estimate', 'e_R_estimate']
    df_comps_and_mp = reorder_df_columns(df_comps_and_mp, left_columns)

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
    df_ur = pd.read_csv(os.path.join(PHOTRIX_TOP_DIRECTORY, an_string, UR_LIST_PATH),
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
        ur_image = Image.from_fits_path(PHOTRIX_TOP_DIRECTORY, os.path.join(an_string, 'Ur'), ur_filename)
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
    # df_mp_master.index = list(df_mp_master['Serial'])
    n_comp_obs = sum([type == 'Comp' for type in df_mp_master['Type']])
    n_mp_obs = len(df_mp_master) - n_comp_obs

    # Fill in the JD_fract and JD_fract2 columns:
    jd_floor = floor(df_mp_master['JD_mid'].min())  # requires that all JD_mid values be known.
    df_mp_master['JD_fract'] = df_mp_master['JD_mid'] - jd_floor

    # Move some of the columns to the Dataframe's left, leave the rest in original order:
    left_columns = ['ObsID', 'ID', 'Type', 'R_estimate', 'Serial', 'JD_mid',
                    'InstMag', 'InstMagSigma', 'Exposure']
    df_mp_master = reorder_df_columns(df_mp_master, left_columns, [])

    # Remove obviously bad comp and MP observations (dataframe rows) and summarize for user::
    is_comp = pd.Series([type == 'Comp' for type in df_mp_master['Type']])
    is_mp = ~is_comp
    is_saturated = pd.Series([adu > ADU_UR_SATURATED for adu in df_mp_master['MaxADU_Ur']])
    is_low_snr = pd.Series([ims > (1.0 / COMPS_MIN_SNR) for ims in df_mp_master['InstMagSigma']])
    print('\nStarting with', str(sum(is_comp)), 'comp obs,', str(sum(is_mp)), 'MP obs:')
    # Identify bad comp observations:
    is_saturated_comp_obs = is_saturated & is_comp
    is_low_snr_comp_obs = is_low_snr & is_comp
    is_bad_comp_obs = is_saturated_comp_obs | is_low_snr_comp_obs
    print('   Comp obs: removing',
          str(sum(is_saturated_comp_obs)), 'saturated, ',
          str(sum(is_low_snr_comp_obs)), 'low SNR.')
    # Identify bad MP observations:
    is_saturated_mp_obs = is_saturated & is_mp
    is_low_snr_mp_obs = is_low_snr & is_mp
    is_bad_mp_obs = is_saturated_mp_obs | is_low_snr_mp_obs
    print('     MP obs: removing',
          str(sum(is_saturated_mp_obs)), 'saturated, ',
          str(sum(is_low_snr_mp_obs)), 'low SNR.')
    # Perform the removal:
    to_remove = is_bad_comp_obs | is_bad_mp_obs
    to_keep = list(~to_remove)
    df_mp_master = df_mp_master.loc[to_keep, :]
    is_comp = pd.Series([type == 'Comp' for type in df_mp_master['Type']])  # remake; rows have changed
    is_mp = ~is_comp  # remake; rows have changed
    print('Keeping', str(sum(is_comp)), 'comp obs,', str(sum(is_mp)), 'MP obs.\n')

    # Write df_comps_and_mp to file:
    fullpath = os.path.join(mp_directory, DF_COMPS_AND_MP_FILENAME)
    df_comps_and_mp.to_csv(fullpath, sep=';', quotechar='"',
                           quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('df_comps_and_mp written to', fullpath)
    # TODO: Not sure the next line's quantities are the relevant ones.
    log_file.write(fullpath + ': ' + str(sum(is_comp)) + ' comp obs, ' + str(sum(is_mp)) + 'MP obs.')
    # do not return df_comps_and_mp...it is stored in file.

    # Write df_master to file:
    fullpath = os.path.join(mp_directory, DF_MP_MASTER_FILENAME)
    df_mp_master.to_csv(fullpath, sep=';', quotechar='"',
                        quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    # TODO: Not sure the next line's quantities are the relevant ones.
    print('df_mp_master written to', fullpath)
    log_file.write(fullpath + ': ' + str(sum(is_comp)) + ' comp obs, ' + str(sum(is_mp)) + 'MP obs.')
    # do not return df_mp_master...it is stored in file.


def prep_comps(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Prepare all data for comp stars, using instrumental magnitudes and other comp data.
         Make preliminary comp-star selections, esp. qualifying comps as being observed in every MP image.
         Make plots to assist user in further screening comp stars.
    From make_df_master(), use:df_mp_master (one row per observation) and
                               df_comps_and_mp (one row per comp and minor planet).
    :param mp_top_directory: path of lowest directory common to all MP photometry FITS, e.g.,
               'J:/Astro/Images/MP Photometry' [string]
    :param mp_number: number of target MP, e.g., 1602 for Indiana. [integer or string].
    :param an_string: Astronight string representation, e.g., '20191106' [string].
    :return: [None] Write updated df_mp_master and df_comps_and_mp to files.
    """
    mp_string = str(mp_number)
    mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)

    # ================ Start TESTING CODE block. ====================
    # Copy backup files (direct from make_df_mp_master()) to files by mp_phot().
    # For testing only:
    # print('***** Using SAVED copies of df_comps_and_mp.csv and df_mp_master.csv.')
    # old_path = os.path.join(mp_directory, 'df_comps_and_mp - Copy.csv')
    # new_path = os.path.join(mp_directory, 'df_comps_and_mp.csv')
    # os.remove(new_path)
    # shutil.copy2(old_path, new_path)
    # old_path = os.path.join(mp_directory, 'df_mp_master - Copy.csv')
    # new_path = os.path.join(mp_directory, 'df_mp_master.csv')
    # os.remove(new_path)
    # shutil.copy2(old_path, new_path)
    # ================= End of TESTING CODE block. ===================

    # Load data:
    df_mp_master = get_df_mp_master(mp_top_directory, mp_string, an_string)
    df_comps_and_mp = get_df_comps_and_mp(mp_top_directory, mp_string, an_string)
    state = get_session_state()  # for extinction and transform values.

    # Make df_VRI, local to this function, one row per MP-field image in V, R, or I.
    is_VRI = list([filter in ['V', 'R', 'I'] for filter in df_mp_master['Filter']])
    VRI_filenames = df_mp_master.loc[is_VRI, 'Filename'].copy().drop_duplicates()  # index is correct too.
    VRI_obs_ids = list(VRI_filenames.index)
    df_VRI = pd.DataFrame({'Serial': VRI_obs_ids}, index=VRI_obs_ids)
    df_VRI['Filename'] = VRI_filenames
    df_VRI['JD_mid'] = list(df_mp_master.loc[VRI_obs_ids, 'JD_mid'])
    df_VRI['UTC_mid'] = list(df_mp_master.loc[VRI_obs_ids, 'UTC_mid'])
    df_VRI['Filter'] = list(df_mp_master.loc[VRI_obs_ids, 'Filter'])
    df_VRI['Airmass'] = list(df_mp_master.loc[VRI_obs_ids, 'Airmass'])
    print(str(len(df_VRI)), 'VRI images.')

    # Calculate zero-point "Z" value for each VRI image, add to df_VRI:
    df_ext_stds = make_df_ext_stds(an_string)  # stds data come from photrix:df_master.
    z_curves = make_z_curves(df_ext_stds)  # each dict entry is 'filter': ZCurve object.
    z_values = [z_curves[filter].at_jd(jd) for filter, jd in zip(df_VRI['Filter'], df_VRI['JD_mid'])]
    df_VRI['Z'] = z_values  # new column.

    # Make list of only comp and MP IDs that have qualified data in *every* image in df_mp_master:
    id_list = df_mp_master['ID'].copy().drop_duplicates()
    n_filenames = len(df_mp_master['Filename'].copy().drop_duplicates())
    qualified_id_list = []
    for this_id in id_list:
        n_this_id = sum([id == this_id for id in df_mp_master['ID']])
        if n_this_id == n_filenames:
            qualified_id_list.append(this_id)
    print('IDs: ', str(len(qualified_id_list)), 'qualified from',
          str(len(id_list)), 'in df_comps_and_mp.')

    # Make new df_qual_obs (subset of df_mp_master):
    obs_to_keep = [id in qualified_id_list for id in df_mp_master['ID']]
    df_qual_obs = df_mp_master.loc[obs_to_keep, :]  # new df; all images have same comps and MPs.
    print('obs: ', str(len(df_qual_obs)), 'kept in df_qual_obs from',
          str(len(df_mp_master)), 'in df_mp_master.')
    del df_mp_master  # trigger error if used below.

    # Make new df_qual_comps_and_mp (subset of df_comps_and_mp):
    rows_to_keep = [id in qualified_id_list for id in df_comps_and_mp['ID']]
    df_qual_comps_and_mp = df_comps_and_mp[rows_to_keep]
    del df_comps_and_mp  # trigger error if used below.

    # Add column 'InAllImages' to df_comps_and_mp and to df_mp_master:
    # qualified_comps_and_mp = [id in qualified_id_list for id in df_comps_and_mp['ID']]
    # df_comps_and_mp['InAllImages'] = qualified_comps_and_mp
    # df_mp_master = pd.merge(df_mp_master, df_comps_and_mp[['ID', 'InAllImages']].copy(),
    #                         how='left', on='ID').set_index('ObsID', drop=False)

    # From VRI images, calculate best untransformed mag for each MP and comp, write to df_comps_and_mp:
    # Columns are named 'Best_untr_mag_V', etc.
    for filter in ['V', 'R', 'I']:
        new_column_name = 'Best_untr_mag_' + filter  # for new column in df_comps_and_mp.
        df_qual_comps_and_mp[new_column_name] = None
        extinction = state['extinction'][filter]
        untransformed_mag_dict = dict()
        for id in qualified_id_list:
            untransformed_mag_dict[id] = []
        for idx in df_VRI.index:
            if df_VRI.loc[idx, 'Filter'] == filter:
                zero_point = df_VRI.loc[idx, 'Z']
                airmass = df_VRI.loc[idx, 'Airmass']
                for id in qualified_id_list:
                    filename = df_VRI.loc[idx, 'Filename']
                    obs_id = filename + '_' + id
                    instrumental_mag = df_qual_obs.loc[obs_id, 'InstMag']
                    untransformed_mag = instrumental_mag - extinction * airmass - zero_point
                    untransformed_mag_dict[id].append(untransformed_mag)
        for id in qualified_id_list:
            best_untransformed_mag = sum(untransformed_mag_dict[id]) / len(untransformed_mag_dict[id])
            df_qual_comps_and_mp.loc[id, new_column_name] = best_untransformed_mag

    # Solve for best (transformed) V-I color index for each comp and MP, write to df_comps_and_mp:
    df_qual_comps_and_mp['Best_CI'] = None
    v_transform = state['transform']['V']
    i_transform = state['transform']['I']
    for id in qualified_id_list:
        best_color_index = (df_qual_comps_and_mp.loc[id, 'Best_untr_mag_V'] -
                            df_qual_comps_and_mp.loc[id, 'Best_untr_mag_I']) /\
                           (1.0 + v_transform - i_transform)
        df_qual_comps_and_mp.loc[id, 'Best_CI'] = best_color_index

    # Generate best *transformed* R magnitude for every comp (not necessary for MP):
    df_qual_comps_and_mp['Best_R_mag'] = None
    r_transform = state['transform']['R']
    for id in qualified_id_list:
        best_r_mag = df_qual_comps_and_mp.loc[id, 'Best_untr_mag_R'] - \
                     r_transform * df_qual_comps_and_mp.loc[id, 'Best_CI']
        df_qual_comps_and_mp.loc[id, 'Best_R_mag'] = best_r_mag

    # Reorder columns in df_qual_comps_and_mp, then write it file:
    df_qual_comps_and_mp = reorder_df_columns(df_qual_comps_and_mp,
                                              ['ID', 'Type', 'Best_R_mag', 'Best_CI'])
    fullpath = os.path.join(mp_directory, DF_QUALIFIED_COMPS_AND_MP_FILENAME)
    df_qual_comps_and_mp.to_csv(fullpath, sep=';', quotechar='"',
                           quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('New df_qual_comps_and_mp written to', fullpath)

    # Reorder columns in df_qual_obs, then write it to file:
    df_qual_obs = reorder_df_columns(df_qual_obs, ['ObsID', 'ID', 'Type'])
    fullpath = os.path.join(mp_directory, DF_QUALIFIED_OBS_FILENAME)
    df_qual_obs.to_csv(fullpath, sep=';', quotechar='"',
                        quoting=2, index=True)  # quoting=2-->quotes around non-numerics.
    print('New df_qual_obs written to', fullpath)


def plot_comps(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Make plots supporting user's final selection of comp stars for this MP session.
    :param mp_top_directory:
    :param mp_number:
    :param an_string:
    :return: [None]
    """
    import matplotlib.pyplot as plt

    mp_string = str(mp_number)
    df_qual_comps_and_mp = get_df_qual_comps_and_mp(mp_top_directory, mp_string, an_string)
    df_qual_obs = get_df_qual_obs(mp_top_directory, mp_string, an_string)

    df_qual_comps_only = df_qual_comps_and_mp[df_qual_comps_and_mp['Type'] == 'Comp']
    df_mp_only = df_qual_comps_and_mp[df_qual_comps_and_mp['Type'] == 'MP']
    mp_color = 'darkred'
    comp_color = 'black'

    # FIGURE 1: (multiplot): One point per comp star, plots in a grid within one figure (page).
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(14, 9))  # (width, height) in "inches"

    # Local support function for *this* grid of plots:
    def make_labels(ax, title, xlabel, ylabel, zero_line=False):
        ax.set_title(title, y=0.91)
        ax.set_xlabel(xlabel, labelpad=0)
        ax.set_ylabel(ylabel, labelpad=0)
        if zero_line is True:
            ax.axhline(y=0, color='lightgray', linewidth=1, zorder=-100)

    # "Canopus comp plot" (R InstMag from VRI image *vs* R mag estimate from catalog):
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
    y_range = max(y) - min(y)
    x_low, x_high = min(x) - margin * x_range, max(x) + margin * x_range
    intercept = sum(y - x) / len(x)
    y_low, y_high = x_low + intercept, x_high + intercept
    # y_low, y_high = min(y) - margin * y_range, max(y) + margin * y_range
    ax.plot([x_low, x_high], [y_low, y_high], color='black', zorder=-100, linewidth=1)

    # "R-shift plot" (Best R mag - R estimate from catalog vs R estimate from catalog):
    # Data from df_qual_comps_only, thus comp stars (no MP).
    ax = axes[0, 1]  # next plot to the right.
    make_labels(ax, 'R-shift comp plot', 'R_estimate from catalog B & V',
                'Best R mag - first R_estimate', zero_line=True)
    ax.scatter(x=df_qual_comps_only['R_estimate'],
               y=df_qual_comps_only['Best_R_mag'] - df_qual_comps_only['R_estimate'],
               alpha=0.5, color=comp_color)

    # Color index plot:
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

    # Finish the figure, and show the entire plot:
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.25)
    fig.suptitle('Comp Star Diagnostics    ::     MP_' + mp_string + '   AN' + an_string,
                 color='darkblue', fontsize=20, weight='bold')
    fig.canvas.set_window_title('Comp star diagnostic plots')
    plt.show()


    # Write diagnostics to console:
    # (these data may better go into a file, later)



    # Write select_comps.txt stub to mp_directory (using statistics--e.g. from above plots?):










SUPPORT_______________________________________________________ = 0


def get_context():
    """ This is run at beginning of workflow functions (except start()) to orient the function.
    :return: 4-tuple: (this_directory, mp_string, an_string, log_file) [3 strings and a file handle]
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
    return this_directory, mp_string, an_string, log_file


def make_df_ext_stds(an_string=None):
    """ Renders a dataframe with all external standards info, taken from prev made photrix df_master.
        Pre-filters the standards to color indices not too red (for safety from variable stars).
        These external standard serve only to set zero-point (curves) for later processing.
    :param an_string: e.g., '20191105', to find the correct df_master [string].
    :return: dataframe of external standards data [pandas Dataframe].
    """
    from photrix.process import get_df_master
    df = get_df_master(an_rel_directory=an_string)
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


def get_df_mp_master(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Simple utility to read df_mp_master.csv file and return the original DataFrame.
    :param mp_top_directory: path to top directory [string].
    :param mp_number: number of target Minor Planet [string or int].
    :param an_string: Astronight (not UTC) date designation, e.g., '20191105' [string].
    :return: df_mp_master from mp_phot() [pandas Dataframe]
    """
    mp_string = str(mp_number)
    mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    fullpath = os.path.join(mp_directory, DF_MP_MASTER_FILENAME)
    df_mp_master = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_mp_master


def get_df_comps_and_mp(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Simple utility to read df_comps.csv file and return the original DataFrame.
    :param mp_top_directory: path to top directory [string].
    :param mp_number: number of target Minor Planet [string or int].
    :param an_string: Astronight (not UTC) date designation, e.g., '20191105' [string].
    :return: df_comps_and_mp from mp_phot() [pandas Dataframe]
    """
    mp_string = str(mp_number)
    mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    fullpath = os.path.join(mp_directory, DF_COMPS_AND_MP_FILENAME)
    df_comps_and_mp = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_comps_and_mp


def get_df_qual_obs(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Simple utility to read df_qual_obs.csv file and return the original DataFrame.
    :param mp_top_directory: path to top directory [string].
    :param mp_number: number of target Minor Planet [string or int].
    :param an_string: Astronight (not UTC) date designation, e.g., '20191105' [string].
    :return: df_qual_obs from mp_phot() [pandas Dataframe]
    """
    mp_string = str(mp_number)
    mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    fullpath = os.path.join(mp_directory, DF_QUALIFIED_OBS_FILENAME)
    df_qual_obs = pd.read_csv(fullpath, sep=';', index_col=0)
    return df_qual_obs


def get_df_qual_comps_and_mp(mp_top_directory=MP_TOP_DIRECTORY, mp_number=None, an_string=None):
    """  Simple utility to read df_qual_comps_and_mp.csv file and return the original DataFrame.
    :param mp_top_directory: path to top directory [string].
    :param mp_number: number of target Minor Planet [string or int].
    :param an_string: Astronight (not UTC) date designation, e.g., '20191105' [string].
    :return: df_qual_comps_and_mp from mp_phot() [pandas Dataframe]
    """
    mp_string = str(mp_number)
    mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    fullpath = os.path.join(mp_directory, DF_QUALIFIED_COMPS_AND_MP_FILENAME)
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


def get_apass10_comps(ra, dec, radius, r_min=None, r_max=None, mp_color_only=True, add_R_estimate=True):
    """  Renders a dataframe with all needed comp info, including estimated R mags.
    Tests OK ~ 20191106.
    :param ra: center Right Ascension for comp search, in degrees only [float].
    :param dec: center Declination for comp search, in degrees only [float].
    :param radius: radius of circular search area, in degrees [float].
    :param r_min: minimum R magnitude (brightest limit) [float]. None = no limit.
    :param r_max: maximum R magnitude (faintest limit) [float]. None = no limit.
    :param mp_color_only: True means keep only comps close to typical MP color [boolean].
    :param add_R_estimate: True means add R_estimate column and its error [boolean].
    :return: dataframe of comp data [pandas Dataframe].
    Columns = degRA, e_RA, degDec, e_Dec, Vmag, e_Vmag, Bmag, e_Bmag, R_estimate, e_R_estimate.
    """
    url = 'https://www.aavso.org/cgi-bin/apass_dr10_download.pl'
    data = {'ra': ra, 'dec': dec, 'radius': radius, 'outtype': 1}
    result = requests.post(url=url, data=data)
    # convert this text to pandas Dataframe (see photrix get_df_master() for how).
    df = pd.read_csv(StringIO(result.text), sep=',')
    df = df.rename(columns={'radeg': 'degRA', 'raerr(")': 'e_RA', 'decdeg': 'degDec', 'decerr(")': 'e_Dec',
                            'Johnson_V (V)': 'Vmag', 'Verr': 'e_Vmag',
                            'Johnson_B (B)': 'Bmag', 'Berr': 'e_Bmag'})
    df['ID'] = df.index
    columns_to_keep = ['ID', 'degRA', 'e_RA', 'degDec', 'e_Dec', 'Vmag', 'e_Vmag', 'Vnobs', 'Bmag',
                       'e_Bmag', 'Bnobs']
    df = df[columns_to_keep]
    df = df[df['e_RA'] < MP_COMP_RADEC_ERROR_MAX]
    df = df[df['e_Dec'] < MP_COMP_RADEC_ERROR_MAX]
    df = df[~pd.isnull(df['Vmag'])]
    df = df[~pd.isnull(df['e_Vmag'])]
    df = df[~pd.isnull(df['Bmag'])]
    df = df[~pd.isnull(df['e_Bmag'])]
    df['B-V'] = df['Bmag'] - df['Vmag']
    df['e_B-V'] = np.sqrt(df['e_Vmag'] ** 2 + df['e_Bmag'] ** 2)
    df_comps = df
    if add_R_estimate:
        df_comps['R_estimate'] = R_ESTIMATE_V_COEFF * df['Vmag'] +\
                                 R_ESTIMATE_BV_COLOR_COEFF * df['B-V'] +\
                                 R_ESTIMATE_INTERCEPT
        df_comps['e_R_estimate'] = np.sqrt(R_ESTIMATE_V_COEFF**2 * df_comps['e_Vmag']**2 +
                                           R_ESTIMATE_BV_COLOR_COEFF**2 * df_comps['e_B-V']**2)
        if r_min is not None:
            df = df[df['R_estimate'] >= r_min]
        if r_max is not None:
            df = df[df['R_estimate'] <= r_max]
        df_comps = df[df['e_R_estimate'] <= ERR_R_ESTIMATE_MAX]
    if mp_color_only is True:
        above_min = MP_COLOR_BV_MIN <= df_comps['B-V']
        below_max = df_comps['B-V'] <= MP_COLOR_BV_MAX
        color_ok = list(above_min & below_max)
        df_comps = df_comps[color_ok]
    return df_comps


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


def reorder_df_columns(df, left_column_list=[], right_column_list=[]):
    new_column_order = left_column_list +\
                       [col_name for col_name in df.columns
                        if col_name not in (left_column_list + right_column_list)] +\
                       right_column_list
    df = df[new_column_order]
    return df


def write_to_log(    text=''):
    pass

# def do_cwd(mp_top_directory=MP_TOP_DIRECTORY + '/Test', mp_number=3460, an_string='20191118'):
#     mp_int = int(mp_number)  # put this in try/catch block.
#     mp_string = str(mp_int)
#     mp_directory = os.path.join(mp_top_directory, 'MP_' + mp_string, 'AN' + an_string)
#     save_cwd = os.getcwd()
#     print('Starting wd:', os.getcwd())
#     print('Going to:', mp_directory, '...')
#     os.chdir(mp_directory)
#     print('wd is now:', os.getcwd())
#     print('going back...')
#     os.chdir(save_cwd)
#     print('finall, wd is now:', os.getcwd())


def try_reg():
    """  This was used to get R_estimate coeffs, using catalog data to predict experimental Best_R_mag.
    :return: [None]
    """
    df_comps_and_mp = get_df_comps_and_mp(mp_top_directory='J:/Astro/Images/MP Photometry/Test',
                                          mp_number=1074, an_string='20191109')
    dfc = df_comps_and_mp[df_comps_and_mp['Type'] == 'Comp']
    dfc = dfc[dfc['InAllImages'] == True]

    # from sklearn.linear_model import LinearRegression
    # x = [[bv, v] for (bv, v) in zip(dfc['B-V'], dfc['Vmag'])]
    # y = list(dfc['Best_R_mag'])
    # reg = LinearRegression(fit_intercept=True)
    # reg.fit(x, y)
    # print('\nsklearn: ', reg.coef_, reg.intercept_)
    #
    # xx = dfc[['B-V', 'Vmag']]
    # yy = dfc['Best_R_mag']
    # reg.fit(xx, yy)
    # print('\nsklearn2: ', reg.coef_, reg.intercept_)

    # # statsmodel w/ formula api (R-style formulas) (fussy about column names):
    # import statsmodels.formula.api as sm
    # dfc['BV'] = dfc['B-V']  # column name B-V doesn't work in formula.
    # result = sm.ols(formula='Best_R_mag ~ BV + Vmag', data=dfc).fit()
    # print('\n' + 'sm.ols:')
    # print(result.summary())

    # statsmodel w/ dataframe-column api:
    import statsmodels.api as sm
    # make column BV as above
    # result = sm.OLS(dfc['Best_R_mag'], dfc[['BV', 'Vmag']]).fit()  # <--- without constant term
    result = sm.OLS(dfc['Best_R_mag'], sm.add_constant(dfc[['B-V', 'Vmag']])).fit()
    print('\n' + 'sm.ols:')
    print(result.summary())

    # statsmodel w/ dataframe-column api:
    import statsmodels.api as sm
    # make column BV as above
    # result = sm.OLS(dfc['Best_R_mag'], dfc[['BV', 'Vmag']]).fit()  # <--- without constant term
    result = sm.OLS(dfc['Best_R_mag'], sm.add_constant(dfc[['R_estimate']])).fit()
    print('\n' + 'sm.ols:')
    print(result.summary())
    # also available: result.params, .pvalues, .rsquared



# *********************************************************************************************
# *********************************************************************************************
# *********************************************************************************************
# ***********  the following comprises EXAMPLE WORKFLOW (which works fine) from early Nov 2019:
#
# class DataCamera:
#     def __init__(self):
#         # All transforms use V-I as color index.
#         self.filter_dict = \
#             {'V': {'z': -20.50, 'extinction': 0.18, 'transform': {'passband': 'V', 'value': -0.0149}},
#              'R': {'z': -20.35, 'extinction': 0.12, 'transform': {'passband': 'R', 'value': +0.0513}},
#              'I': {'z': -20.15, 'extinction': 0.09, 'transform': {'passband': 'I', 'value' : 0.0494}},
#              'Clear': {'z': -21.5, 'extinction': 0.14, 'transform': {'passband': 'R', 'value': +0.133}}
#              }
#
#     def instmag(self, target, image):
#         """
#         Returns instrument magnitude for one source in one image.
#         :param target: astronomical photon source (star or MP) [DataStandard object].
#         :param image: represents one image and its assoc. data [DataImage object].
#         :return:
#         """
#         passband = self.filter_dict[image.filter]['transform']['passband']
#         catalog_mag = target.mag(passband)
#         zero_point = self.filter_dict[image.filter]['z']
#         corrections = self.calc_corrections(target, image)
#         instrument_magnitude = catalog_mag + zero_point + corrections
#         return instrument_magnitude
#
#     def solve_for_z(self, instmag, target, image):
#         # Requires catalog data. Not for unknowns.
#         corrections = self.calc_corrections(target, image)
#         z = instmag - target.mag(image.filter) - corrections
#         return z
#
#     def solve_for_best_mag(self, instmag, target, image):
#         # Requires catalog data. Not for unknowns.
#         corrections = self.calc_corrections(target, image)
#         zero_point = self.filter_dict[image.filter]['z']
#         best_mag = instmag - corrections - zero_point
#         return best_mag
#
#     def calc_corrections(self, target, image):
#         """ Includes corrections from cat mag to experimental image, *excluding* zero-point. """
#         # Requires catalog data. Not for unknowns.
#         filter_info = self.filter_dict[image.filter]
#         transform_value = filter_info['transform']['value']
#         extinction = filter_info['extinction']
#         color_index = target.mag(passband='V') - target.mag(passband='I')  # color index always in V-I
#         corrections = image.delta + extinction * image.airmass + transform_value * color_index
#         return corrections
#
#     def make_image(self, filter, target_list, airmass, delta=0.0):
#         # Factory method for DataImage object. Includes no catalog data (thus OK for unknowns).
#         image = DataImage(self, filter, airmass, delta)
#         for target in target_list:
#             image.obs_dict[target.name] = self.instmag(target, image)
#         return image
#
#     def extract_color_index_value(self, untransformed_best_mag_v, untransformed_best_mag_i):
#         color_index_value = (untransformed_best_mag_v - untransformed_best_mag_i) / \
#                             (1.0 + self.filter_dict['V']['transform']['value'] -
#                              self.filter_dict['I']['transform']['value'])
#         return color_index_value
#
#     def __str__(self):
#         return 'DataCamera object'
#
#     def __repr__(self):
#         return 'DataCamera object'
#
#
# class DataStandard:
#     def __init__(self, name, mag_dict):
#         self.name = name
#         self.mag_dict = mag_dict
#
#     def mag(self, passband):
#         return self.mag_dict.get(passband)
#
#     def __str__(self):
#         return 'DataStandard ' + self.name
#
#     def __repr__(self):
#         return 'DataStandard ' + self.name
#
#
# class DataTarget:
#     def __init__(self, name, mag_dict):
#         self.name = name
#         self.mag_dict = mag_dict
#
#     def mag(self, passband):
#         return self.mag_dict.get(passband)
#
#     def __str__(self):
#         return 'DataTarget ' + self.name
#
#     def __repr__(self):
#         return 'DataTarget ' + self.name
#
#
# class DataImage:
#     # Includes NO catalog data (thus OK for unknowns).
#     def __init__(self, camera, filter, airmass, delta=0.0):
#         self.camera = camera
#         self.filter = filter
#         self.airmass = airmass
#         self.delta = delta
#         self.obs_dict = dict()  # {target_name: instmag, ...}
#
#     def untransformed_best_mag(self, target_name, delta=None):
#         instmag = self.obs_dict[target_name]
#         zero_point = self.camera.filter_dict[self.filter]['z']
#         if delta is None:
#             this_delta = self.delta  # for constructing and testing images.
#         else:
#             this_delta = delta  # set to zero if delta is unknown (obs images).
#         extinction = self.camera.filter_dict[self.filter]['extinction']
#         airmass = self.airmass
#         this_mag = instmag - zero_point - this_delta - extinction * airmass
#         return this_mag
#
#     def __str__(self):
#         return 'DataImage object'
#
#     def __repr__(self):
#         return 'DataImage object'
#





# def example_workflow():
#     """ Try out new MP photometric reduction process. """
#     cam = DataCamera()
#
#     # ============= DataStandard images: =============
#     stds = [DataStandard('std1', {'V': 12.2, 'R': 12.6, 'I': 12.95}),
#             DataStandard('std2', {'V': 12.5, 'R': 12.7, 'I': 12.9}),
#             DataStandard('std3', {'V': 12.88, 'R': 13.22, 'I': 13.5})]
#     std_instmags = dict()
#     std_images = {'V': cam.make_image('V', stds, 1.5, 0),
#                   'R': cam.make_image('R', stds, 1.52, 0),
#                   'I': cam.make_image('I', stds, 1.538, 0)}
#
#     for this_filter in std_images.keys():
#         std_instmags[this_filter] = [cam.instmag(std, std_images[this_filter]) for std in stds]
#
#     # Test for consistency:
#     # for this_filter in std_instmags.keys():
#     #     instmag_list = std_instmags[this_filter]
#     #     for instmag, std in zip(instmag_list, stds):
#     #         print(cam.solve_for_z(instmag, std, std_images[this_filter]),
#     #         cam.filter_dict[this_filter]['z'])
#
#     # Test by back-calculating best (catalog) mags:
#     # for this_filter in std_instmags.keys():
#     #     instmag_list = std_instmags[this_filter]
#     #     for instmag, std in zip(instmag_list, stds):
#     #         print(std.name, this_filter, cam.solve_for_best_mag(instmag, std, std_images[this_filter]))
#
#     # Test untransformed and transformed best mags from images (no ref to catalog mags):
#     # for this_filter in std_instmags.keys():
#     #     std_image = std_images[this_filter]
#     #     for this_std in stds:
#     #         cat_mag = this_std.mag(this_filter)
#     #         std_name = this_std.name
#     #         untr_best_mag = std_image.untransformed_best_mag(std_name)
#     #         print(std_name, this_filter, cat_mag, untr_best_mag)
#
#     # Test extraction of color index values & transformation of untransformed mags:
#     # print()
#     # for this_std in stds:
#     #     image_v = std_images['V']
#     #     image_i = std_images['I']
#     #     cat_mag_v = this_std.mag('V')  # to test against.
#     #     cat_mag_i = this_std.mag('I')  # "
#     #     std_name = this_std.name
#     #     untr_best_mag_v = image_v.untransformed_best_mag(std_name)
#     #     untr_best_mag_i = image_i.untransformed_best_mag(std_name)
#     #     best_color_index = cam.extract_color_index_value(untr_best_mag_v, untr_best_mag_i)
#     #     print(std_name, cat_mag_v, cat_mag_i, cat_mag_v - cat_mag_i, best_color_index)
#
#     # ============= Comp initial images (V & I): =============
#     comps = [DataTarget('comp1', {'V': 13.32, 'R': 12.44, 'I': 11.74}),
#              DataTarget('comp2', {'V': 12.52, 'R': 12.21, 'I': 12.01}),
#              DataTarget('comp3', {'V': 11.89, 'R': 11.23, 'I': 11.52})]
#     mp = [DataTarget('MP', {'V': 13.5, 'R': 14.02, 'I': 14.49})]
#     targets = comps + mp
#     pre_v_image = cam.make_image('V', targets, 1.61, 0)
#     pre_i_image = cam.make_image('I', targets, 1.61, 0)
#
#     # Measure comps' best color index values without knowing passband mags (i.e., to target obj at all):
#     comps_color_index = [cam.extract_color_index_value(pre_v_image.untransformed_best_mag(comp.name),
#                                                        pre_i_image.untransformed_best_mag(comp.name))
#                          for comp in comps]
#     # for i, comp in enumerate(comps):
#     #     print(comp.name, comp.mag('V') - comp.mag('I'),  comps_color_index[i])
#
#     mp_color_index = cam.extract_color_index_value(pre_v_image.untransformed_best_mag(mp[0].name),
#                                                    pre_i_image.untransformed_best_mag(mp[0].name))
#     # print(mp[0].name, mp[0].mag('V') - mp[0].mag('I'),  mp_color_index)
#
#     # Last prep step: take an image in R, derive best R comp mags (do not use catalog mags at all):
#     pre_r_image = cam.make_image('R', comps, 1.61, 0)
#     this_transform = cam.filter_dict['R']['transform']['value']
#     comp_r_mags = []
#     for comp, color_index in zip(comps, comps_color_index):
#         untr_best_mag_r = pre_r_image.untransformed_best_mag(comp.name)
#         best_mag_r = untr_best_mag_r - this_transform * color_index
#         comp_r_mags.append(best_mag_r)
#     # for comp, r_mag in zip(comps, comp_r_mags):
#     #     print(comp.name, r_mag)
#
#     # ============= We're now ready to make obs images in Clear filter, then back-calc best R mags for MP.
#     obs_images = [cam.make_image('Clear', targets, 1.610, +0.01),
#                   cam.make_image('Clear', targets, 1.620, -0.015),
#                   cam.make_image('Clear', targets, 1.633, +0.014)]
#     this_transform = cam.filter_dict['Clear']['transform']['value']
#     for im in obs_images:
#         derived_deltas = []
#         for comp, mag, ci in zip(comps, comp_r_mags, comps_color_index):
#             untr_best_mag_clear = im.untransformed_best_mag(comp.name, delta=0.0)
#             this_mag_r = untr_best_mag_clear - this_transform * ci  # delta is yet unknown.
#             this_derived_delta = this_mag_r - mag
#             derived_deltas.append(this_derived_delta)
#         print('derived_deltas: ', str(derived_deltas))
#         mean_derived_delta = sum(derived_deltas) / float(len(derived_deltas))
#         untr_best_mag_clear = im.untransformed_best_mag(mp[0].name, delta=mean_derived_delta)
#         mp_mag_r = untr_best_mag_clear - this_transform * mp_color_index
#         print('comp R mag:', mp_mag_r)  # this works...yay





# def get_apass9_comps(ra, dec, radius):
#     """ Get APASS 9 comps via Vizier catalog database.
#     :param ra: RA in degrees [float].
#     :param dec: Dec in degrees [float].
#     :param radius: in arcminutes [float].
#     :return:
#     """
#     from astroquery.vizier import Vizier
#     # catalog_list = Vizier.find_catalogs('APASS')
#     from astropy.coordinates import Angle
#     import astropy.units as u
#     import astropy.coordinates as coord
#     result = Vizier.query_region(coord.SkyCoord(ra=299.590 * u.deg, dec=35.201 * u.deg,
#                                                 frame='icrs'), width="30m", catalog=["APASS"])
#     df_apass = result[0].to_pandas()
#     return df_apass
#
#
# def refine_df_apass9(df_apass, r_min=None, r_max=None):
#     columns_to_keep = ['recno', 'RAJ2000', 'e_RAJ2000', 'DEJ2000', 'e_DEJ2000', 'nobs', 'mobs',
#                        'B-V', 'e_B-V', 'Vmag', 'e_Vmag']
#     df = df_apass[columns_to_keep]
#     df = df[df['e_RAJ2000'] < 2.0]
#     df = df[df['e_DEJ2000'] < 2.0]
#     df = df[~pd.isnull(df['B-V'])]
#     df = df[~pd.isnull(df['e_B-V'])]
#     df = df[~pd.isnull(df['Vmag'])]
#     df = df[~pd.isnull(df['e_Vmag'])]
#     df['R_estimate'] = df['Vmag'] - 0.5 * df['B-V']
#     df['e_R_estimate'] = np.sqrt(df['e_Vmag'] ** 2 + 0.25 * df['e_B-V'] ** 2)  # error in quadrature.
#     if r_min is not None:
#         df = df[df['R_estimate'] >= r_min]
#     if r_max is not None:
#         df = df[df['R_estimate'] <= r_max]
#     df = df[df['e_R_estimate'] <= 0.04]
#     return df