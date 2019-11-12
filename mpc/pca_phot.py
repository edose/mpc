__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
import shutil
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import wcs

DEFAULT_TOP_DIRECTORY = 'C:/Astro/MP_photometry_test'
CROPPED_FITS_SUBDIRECTORY = 'Cropped'
RECROPPED_FITS_SUBDIRECTORY = 'Recropped'
DF_FITS_SUBDIRECTORY = 'Results'

ISO_8601_FORMAT = '%Y-%m-%dT%H:%M:%S'

# The Principal Components (or non-negative regression) prospective workflow.

#   ##### MP Photometry WORKFLOW:
#   Put all MP FITS images in one top folder.
#   make_df_fits().
#   In MaxIm: Record MP's (x,y) of earliest (as mp_xy1) and latest (as mp_xy2) (uncropped) images.
#   In Seqplot: get MP-centered image size just large enough to get solar-type comp stars in frame.
#   first_crop(), with extra_margin that gets comp stars in frame.
#   align_maxim().
#   In MaxIm: Record MP's (x,y) of earliest (as mp_xy1) and latest (as mp_xy2) (cropped) images.
#   recrop().
#   ...


def make_df_fits(top_directory):
    # Make dataframe df_fits of all FITS file data (names, exp, center time, verify sizes equivalent):
    filenames, exposures, utc_mids, x_sizes, y_sizes = [], [], [], [], []
    for entry in os.scandir(top_directory):
        if entry.is_file():
            filenames.append(entry.name)  # string

            fullpath = os.path.join(top_directory, entry.name)
            hdulist = fits.open(fullpath)
            header = hdulist[0].header
            # all_header_keys = header.keys()
            exposure = header_value(header, ['EXPTIME', 'EXPOSURE'])
            exposures.append(exposure)  # seconds

            utc_string = header_value(header, 'DATE-OBS')
            utc_start = datetime.strptime(utc_string, ISO_8601_FORMAT).replace(tzinfo=timezone.utc)
            utc_mid = utc_start + timedelta(seconds=exposure / 2.0)
            utc_mids.append(utc_mid)  # python datetime object

            x_size = header_value(header, 'NAXIS1')
            y_size = header_value(header, 'NAXIS2')
            x_sizes.append(x_size)
            y_sizes.append(y_size)
            hdulist.close()

    if max(x_sizes) != min(x_sizes) or max(y_sizes) != min(y_sizes):
        print('>>>>> All FITS images must be same dimensions but are not. Stopping.')
        return
    df_fits = pd.DataFrame({'Filename': filenames, 'Exposure': exposures, 'UTC_mid': utc_mids,
                            'X_size': x_sizes, 'Y_size': y_sizes},
                           index=filenames).sort_values(by=['UTC_mid'])
    write_df_fits(top_directory, df_fits)


def first_crop(top_directory=DEFAULT_TOP_DIRECTORY, mp_xy1=None, mp_xy2=None,
               mp_radius=10, tracking_uncert=10, extra_margin=350):
    crop(top_directory, input_subdirectory_name='', output_subdirectory_name=CROPPED_FITS_SUBDIRECTORY,
         mp_xy1=mp_xy1, mp_xy2=mp_xy2,
         mp_radius=mp_radius, tracking_uncert=tracking_uncert, extra_margin=extra_margin)


def align_maxim(df_fits):
    import win32com.client
    app = win32com.client.Dispatch('MaxIm.Application')
    fullpaths = [os.path.join(DEFAULT_TOP_DIRECTORY, CROPPED_FITS_SUBDIRECTORY, name)
                 for name in df_fits['Filename']]
    for fullpath in fullpaths:
        print('Trying to open fullpath=' + fullpath)
        doc = win32com.client.Dispatch('MaxIm.Document')
        doc.OpenFile(fullpath)
    docs = app.Documents
    docs[0].AlignImages(1, True)
    for i, doc in enumerate(docs):
        doc.SaveFile(fullpaths[i], 3, False, 3, False)  # FITS, no stretch, floats, no compression.


def recrop(top_directory=DEFAULT_TOP_DIRECTORY, mp_xy1=None, mp_xy2=None,
           mp_radius=10, extra_margin=50):
    crop(top_directory, 'Cropped', 'Recropped',
         mp_xy1=mp_xy1, mp_xy2=mp_xy2, mp_radius=mp_radius, tracking_uncert=0, extra_margin=extra_margin)


def mask_mp(top_directory=DEFAULT_TOP_DIRECTORY, df_fits=None, mp_xy1=None, mp_xy2=None, mp_radius=10):
    """ Mask out MP signal, returning list of masked arrays.
    :param top_directory:
    :param df_fits:
    :param mp_xy1:
    :param mp_xy2:
    :param mp_radius:
    :return: image_list [list of masked arrays]
    """
    image_list = []
    return image_list





######################################################################################################
# phot.py UTILITY functions:


def clean_subdirectory(top_directory, subdirectory_name):
    """ Create new empty subdirectory, or empty an existing subdirectory.
    :param top_directory: e.g., 'C:/Astro/MP_images/' [string]
    :param subdirectory_name: subdir name (only), e.g., 'Cropped' [string]
    :return: None
    """
    subdir_path = os.path.join(top_directory, subdirectory_name)
    if os.path.exists(subdir_path):
        shutil.rmtree(subdir_path)
    os.makedirs(subdir_path)


def crop(top_directory=DEFAULT_TOP_DIRECTORY, input_subdirectory_name=None, output_subdirectory_name=None,
         mp_xy1=None, mp_xy2=None, mp_radius=10, tracking_uncert=0, extra_margin=350):
    """ Crop all FITS images in one directory, write cropped FITS images into a sibling directory.
    :param top_directory: e.g., 'C:/Astro/MP_images/' [string]
    :param input_subdirectory_name: subdir name (only), e.g., 'Cropped' or '' for top [string]
    :param output_subdirectory_name: subdir name (only), e.g., 'Recropped' [string]
    :param mp_xy1: x and y in pixels of MP in earliest image [2-tuple of floats]
    :param mp_xy2: x and y in pixels of MP in latest image [2-tuple of floats]
    :param mp_radius: flux aperture radius in pixels for MP [float]
    :param tracking_uncert: worst-case estimate of tracking error between images, in pixels [float]
    :param extra_margin: extra margin on EACH side, in pixels, e.g. to give enough comp stars [float]
    :return: None
    """
    if mp_xy1 is None or mp_xy2 is None:
        print(">>>>> mp_xy1 and mp_xy2 must be tuples of (x,y) in pixels for outermost MP positions.")
        return

    # Empty the output_subdirectory (create if needed):
    clean_subdirectory(top_directory, output_subdirectory_name)

    # Make directory strings:
    if input_subdirectory_name.strip() == '':
        input_directory = top_directory
    else:
        input_directory = os.path.join(top_directory, input_subdirectory_name)
    output_directory = os.path.join(top_directory, output_subdirectory_name)

    # Establish corners of cropping rectangle (same for all images):
    x1, y1 = mp_xy1
    x2, y2 = mp_xy2
    full_margin = mp_radius + tracking_uncert + extra_margin
    xlow = min(x1, x2) - full_margin
    ylow = min(y1, y2) - full_margin
    xhigh = max(x1, x2) + full_margin
    yhigh = max(y1, y2) + full_margin

    # For each FITS file, recrop and save into new subdirectory:
    df_fits = read_df_fits()
    output_filenames = []
    for filename in df_fits[input_subdirectory_name]:
        output_fullpath = os.path.join(output_directory, filename)
        with fits.open(output_fullpath) as hdulist:
            # Update image, then header. Ignore WCS (plate-solving) data--will be invalid in cropped FITS.
            image_fits = hdulist[0].data.astype(np.float32)  # axes as in FITS (transposed from MaxIm).
            output_image = image_fits[ylow:yhigh, xlow:xhigh]
            hdulist[0].data = output_image

            header = hdulist[0].header
            header['BITPIX'] = 32  # see float32, above.
            header['NAXIS1'] = xhigh - xlow
            header['NAXIS2'] = yhigh - ylow
            hdulist[0].header = header

            output_filename = tagged_filename(filename, tag=output_subdirectory_name)
            output_filenames.append(output_filename)
            output_fullpath = os.path.join(output_directory, output_filename)
            hdulist.writeto(output_fullpath)
            hdulist.close()


def tagged_filename(filename, tag):
    """ Tag an original FITS filename, preserving the file extension.
    :param filename: original FITS filename, e.g., 'MP_1187-0001-Clear.fit' [string]
    :param tag: tag to add, e.g., 'Cropped' [string]
    :return: tagged filename, e.g., 'MP_1176-0001-Clear_Cropped.fit' [string]
    """
    name_split = filename.split('.')
    name = '.'.join(name_split[:len(name_split) - 1])
    extension = name_split[-1]
    output_filename = name + '_' + tag + '.' + extension


def header_value(header, key):
    """ Returns value associated with key in FITS header (first one if a list of keys).
    Adapted from photrix.image.FITS.header_value(), 20191014
    :param header: [FITS header object]
    :param key: FITS header key [string] -OR- list of keys to try [list of strings]
    :return: value of FITS header entry, typically [float] if possible, else [string]
    """
    if isinstance(key, str):
        return header.get(key, None)
    for k in key:
        value = header.get(k, None)
        if value is not None:
            return value
    return None


def write_df_fits(top_directory, df_fits):
    clean_subdirectory(top_directory, DF_FITS_SUBDIRECTORY)
    fullpath = os.path.join(top_directory, DF_FITS_SUBDIRECTORY, 'df_fits.csv')
    df_fits.write_csv(fullpath)


def read_df_fits(top_directory):
    fullpath = os.path.join(top_directory, DF_FITS_SUBDIRECTORY, 'df_fits.csv')
    df_fits = pd.read_csv(fullpath)
    return df_fits
