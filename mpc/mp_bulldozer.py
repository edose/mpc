__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# ##############################################################################
# mp_bulldozer.py // July-August 2020
# Eric Dose, New Mexico Mira Project  [ <-- NMMP to be renamed late 2020 ]
# Albuquerque, New Mexico, (what's left of the) USA
#
# To whomever this egregious new repo file inflicts its ugly self upon...
#   No, it's not you, it's the code. Hold your nose for a couple of weeks, please.
#
# To wit: this is my drunken first attempt at a long-held wish to magically remove
# sky background from minor planet (MP) / asteroid images.
#
# Theory is: it should be possible, because the MP is moving, but the stars are
# essentially not. So, mask out the MP from all images,
# average these masked images, and you should have a synthetic image of that
# sky region as though the MP were not crossing it. Then custom-scale this
# synthetic image to each working image to 'bulldoze' its background.
# Just what you need to remove the accursed effect of stars on MP photometry.
#
# The original plan to use mixed-model regression should have worked, but
# statsmodel's implementation gives singular matrices. That's wrong, but I know of
# no other MMR implementation in python. So now we're trying a more
# classical photometry approach. It's going OK so far.
#
# It's going to require that I account not only for image-to-image background and
# sky transparency changes, but definitely for PSF changes as well. And possibly for
# airmass changes as the comp stars' differing extinction have their effect. Sigh.
#
# Please don't be alarmed by all the stuff hard-coded into this. This is just
# conceptual coding and testing--the hard-coding will all get pulled out for
# production (he said).
#
# We shall see. This is going to be a long slog. Wish me luck from a safe distance, and
# Cheers,
# Eric
#
# July 16, 2020 ... stay tuned
#
# ##############################################################################

import os
from math import sqrt, log, floor, ceil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
# import astropy.wcs as wcs
from astropy.stats import sigma_clipped_stats
from astropy.nddata import CCDData
from astropy.modeling import models, fitting
from astropy.modeling.models import Gaussian2D
from photutils import centroid_com
from astropy.convolution import convolve
from astropy.visualization import LogStretch, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize, imshow_norm, MinMaxInterval
from ccdproc import wcs_project, trim_image, Combiner
from photutils import make_source_mask, DAOStarFinder, CircularAperture, aperture_photometry,\
    CircularAnnulus, data_properties, create_matching_kernel, \
    SplitCosineBellWindow, CosineBellWindow

from .mp_phot import get_fits_filenames

FWHM_PER_SIGMA = 2 * sqrt(2 * log(2))  # ca. 2.355

TEST_FITS_DIR = 'C:/Astro/MP Photometry/MP_1111/AN20200617'
PIXEL_SHIFT_TOLERANCE = 200  # maximum image shift in pixels we expect from bad pointing, tracking, etc.


KERNEL_NOMINAL_SIZE = 80  # edge length in pixels
MID_IMAGE_PADDING = 25  # extra padding around bounding box, in pixels.
SUBIMAGE_PADDING = 80   # "
# TODO: need to make equal the MP and comp-star radii for aperture photometry.
MP_MASK_RADIUS_PER_SIGMA = 6
AP_PHOT_RADIUS_PER_SIGMA = 3
# R_APERTURE = 9      # "
# R_ANNULUS_IN = 15   # "
# R_ANNULUS_OUT = 20  # "
EDGE_PIXELS_REGRESSION = 5
MAX_LOOP_ITERATIONS = 3
MAX_FOR_CONVERGENCE = 0.025  # in pixels, root-mean-square-deviation.
CCD_GAIN = 1.57  # electrons per pixel

# SOURCE_MATCH_TOLERANCE = 2  # distance in pixels


def try_bulldozer():
    """ Dummy calling fn, using original FITS files."""
    directory_path = TEST_FITS_DIR
    filter = 'Clear'
    mp_file_early = 'MP_191-0001-Clear.fts'
    mp_file_late = 'MP_191-0028-Clear.fts'
    mp_pix_early = (826.4, 1077.4)
    mp_pix_late = (1144.3, 1099.3)
    ref_star_file = 'MP_191-0001-Clear.fts'
    ref_star_pix = (957.2, 1125.0)

    whatever = bulldozer(directory_path, filter,
                         mp_file_early, mp_file_late, mp_pix_early, mp_pix_late,
                         ref_star_file, ref_star_pix)
    return whatever


def try_bd_added(i_mp):
    """ Dummy calling fn, using '_Added' FITS files previously prepared.
    :param i_mp: number (0, 1, ...) of MP to measure.\
      """
    directory_path = os.path.join(TEST_FITS_DIR, 'Added')
    filter = 'Clear'
    mp_file_early = 'MP_191-0001-Clear_Added.fts'
    mp_file_late = 'MP_191-0028-Clear_Added.fts'
    sources = [{'name': 'Sparse_bkgd',  'xy1_early': (1510, 698),  'xy1_late': (1746, 646)},
               {'name': 'Dense_bkgd',   'xy1_early': (1899, 990),  'xy1_late': (2234, 1142)},
               {'name': 'One brt star', 'xy1_early': (1367.0, 1543.0), 'xy1_late': (1586.0, 1532.0)}]
    mp_pix_early = sources[i_mp]['xy1_early']
    mp_pix_late = sources[i_mp]['xy1_late']

    ref_star_file = 'MP_191-0001-Clear_Added.fts'
    ref_star_pix = [(1585.2, 587.2), (2006.8, 1337.2), (1461.6, 1370.0)][i_mp]

    whatever = bulldozer(directory_path, filter,
                         mp_file_early, mp_file_late, mp_pix_early, mp_pix_late,
                         ref_star_file, ref_star_pix)
    return whatever


def bulldozer(directory_path, filter,
              mp_file_early, mp_file_late, mp_pix_early, mp_pix_late,
              ref_star_file, ref_star_pix):
    """ A new sky-flattening algorithm.
    :param directory_path: where the FITS files are found. [string]
    :param filter: photometric filter in place when the FITS images were taken. [string]
    :param mp_file_early: name of a FITS file taken early in the session, to locate MP. [string]
    :param mp_file_late: name of a FITS file taken late in the session, to locate MP. [string]
    :param mp_pix_early: (x,y) FITS-convention pixel position of MP in mp_file_early. [2-tuple of floats]
    :param mp_pix_late: (x,y) FITS-convention pixel position of MP in mp_file_late. [2-tuple of floats]
    :param ref_star_file: name of FITS file in which ref star is easily located. [string]
    :param ref_star_pix: (x,y) pixel position of ref star in mp_star_file. [2-tuple of floats]
    :return: large dataframe, in which the most interesting column is 'Best_MP_flux',
             the MP flux in total ADUs, from this sky-subtracted algorithm. [pandas DataFrame]
    """

    # ###################################################################################
    # Initial setup: ####################################################################

    df_images = start_df_images(directory_path, filter)

    ref_star_radec = calc_ref_star_radec(df_images, ref_star_file, ref_star_pix)

    df_images = calc_mp_radecs(df_images, mp_file_early, mp_file_late, mp_pix_early, mp_pix_late)

    df_images = calc_image_offsets(df_images, ref_star_radec)

    # ###################################################################################
    # Make mid-images, ref-star kernels, and matching kernels (1/image): ################

    df_images = crop_to_mid_images(df_images, ref_star_radec)

    df_images = align_mid_images(df_images)

    df_images = recrop_mid_images(df_images)

    df_images = make_ref_star_psfs(df_images, ref_star_radec)

    target_sigma = calc_target_kernel_sigma(df_images)

    df_images = make_matching_kernels(df_images, target_sigma)

    df_images = convolve_mid_images(df_images)

    # return df_images

    # ###################################################################################
    # Make subimages (already aligned and convolved) & a few derivatives:  ##############

    df_images = crop_to_subimages(df_images, ref_star_radec)

    df_images = calc_first_ref_star_xy0(df_images, ref_star_radec)

    df_images = calc_first_mp_xy0(df_images)

    df_images = extract_first_subarrays(df_images)

    df_images = estimate_misalignments(df_images)

    # ###################################################################################
    # REFINEMENT LOOP:  #################################################################

    for i_iteration in range(MAX_LOOP_ITERATIONS):

        df_images = convolve_subarrays(df_images)

        df_images = calc_current_mp_xy0(df_images)

        df_images = mask_mp_from_current_subarrays(df_images, target_sigma)

        df_images = calc_subarray_backgrounds(df_images)

        df_images = subtract_background_from_subarrays(df_images)

        averaged_subarray = make_averaged_subarray(df_images)

        # ###################################################################################
        # Get best MP-only subimages : #######################################################

        # OLS each MP-masked image to a + b*sources (does a ~= above bkgd?)
        df_images = decompose_subarrays(df_images, averaged_subarray)

        # make best unmasked images from OLS (i.e., ~ MP-only on ~zero bkgd)
        df_images = make_mp_only_subarrays(df_images, averaged_subarray)

        df_images = estimate_misalignments(df_images)

        converged = evaluate_convergence(df_images)

        # return df_images

        if converged:
            print('breaking out of loop because converged.')
            break

    # ###################################################################################
    # Get best MP-only fluxes from aperture photometry: ##################################

    # aperture photometry to get best MP fluxes (do we even need sky annulae?)
    df_images = do_mp_aperture_photometry(df_images, target_sigma)

    return df_images


BULLDOZER_SUBFUNCTIONS_______________________________ = 0


def start_df_images(directory_path, filter):
    """ Make starting (small) version of key dataframe.
    :param directory_path: where the FITS files are comprising this MP photometry session, where
           "session" by convention = all images from one night, targeting one minor planet (MP). [string]
    :param filter: name of the filter through which the FITS images were taken. [string]
    :return: starting version of df_images, one row per FITS file.
             To be used for this session of MP photometry. This is the central data table,
             and it will be updated throughout the photometry workflow to follow. [pandas DataTable]
    """
    # Get all FITS filenames in directory, and read them into list of CCDData objects:
    filenames = [fn for fn in get_fits_filenames(directory_path) if fn.startswith('MP_')]
    images = [CCDData.read(os.path.join(directory_path, fn), unit='adu') for fn in filenames]

    # Keep only filenames and images in chosen filter:
    keep_image = [(im.meta['Filter'] == filter) for im in images]
    filenames = [f for (f, ki) in zip(filenames, keep_image) if ki is True]
    images = [im for (im, ki) in zip(images, keep_image) if ki is True]

    # Replace obsolete header key often found in FITS files derived from MaxIm DL:
    for i in images:
        if 'RADECSYS' in i.meta.keys():
            i.meta['RADESYSa'] = i.meta.pop('RADECSYS')  # both ops in one statement. cool.

    # Gather initial data from CCDData objects, then make df_images:
    filters = [im.meta['Filter'] for im in images]
    exposures = [im.meta['exposure'] for im in images]
    jds = [im.meta['jd'] for im in images]
    jd_mids = [jd + exp / 2 / 24 / 3600 for (jd, exp) in zip(jds, exposures)]
    df_images = pd.DataFrame(data={'Filename': filenames, 'Image': images, 'Filter': filters,
                                   'Exposure': exposures, 'JD_mid': jd_mids},
                             index=filenames)
    df_images = df_images.sort_values(by='JD_mid')
    print('start_df_images() done:', len(df_images), 'images.')
    return df_images


def calc_ref_star_radec(df_images, ref_star_file, ref_star_pix):
    """ Returns RA,Dec sky position of reference star."""
    origin = 1  # because user is reading pix positions from FITS image.
    ref_star_radec = df_images.loc[ref_star_file, 'Image'].wcs.all_pix2world([list(ref_star_pix)], origin)
    # print('ref_star_radec =', str(ref_star_radec))
    return ref_star_radec


def calc_mp_radecs(df_images, mp_file_early, mp_file_late, mp_pix_early, mp_pix_late):
    """ Adds MP RA,Dec to each image row in df_images, by linear interpolation.
        Assumes images are plate solved but not necessarily (& probably not) aligned.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :param mp_file_early: name of file holding image early in session. [string]
    :param mp_file_late: name of file holding image late in session. [string]
    :param mp_pix_early: (x,y) pixel position (origin-1) of MP for image mp_file_early. [2-tuple of floats]
    :param mp_pix_late: (x,y) pixel position (origin-1) of MP for image mp_file_late. [2-tuple of floats]
    :return: dataframe with new rows 'MP_RA' and 'MP_Dec'. [pandas DataFrame]
    """
    # RA, Dec positions will be our anchors (pixel locations will change with trimming and aligning):
    mp_radec_early = df_images.loc[mp_file_early, 'Image'].wcs.all_pix2world([list(mp_pix_early)], 1)
    mp_radec_late = df_images.loc[mp_file_late, 'Image'].wcs.all_pix2world([list(mp_pix_late)], 1)
    ra_early, dec_early = tuple(mp_radec_early[0])  # first pt for interpolation.
    ra_late, dec_late = tuple(mp_radec_late[0])     # last pt for interpolation.

    # Get MP rates of change in RA and Dec:
    d_ra = ra_late - ra_early
    d_dec = dec_late - dec_early
    jd_early = df_images.loc[mp_file_early, 'JD_mid']
    jd_late = df_images.loc[mp_file_late, 'JD_mid']
    d_jd = jd_late - jd_early
    ra_rate = d_ra / d_jd    # in degrees per day.
    dec_rate = d_dec / d_jd  # "

    # Inter-/Extrapolate RA,Dec locations for all images, write them into new df_images columns:
    # (We have to have these to make the larger bounding box.)
    jds = [df_images.loc[i, 'JD_mid'] for i in df_images.index]
    df_images['MP_RA'] = [ra_early + (jd - jd_early) * ra_rate for jd in jds]
    df_images['MP_Dec'] = [dec_early + (jd - jd_early) * dec_rate for jd in jds]
    print('add_mp_radecs() done.')
    return df_images


def calc_image_offsets(df_images, ref_star_radec):
    """ Calculate and store image-to-image shifts due to imperfect pointing, tracking, etc.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :param ref_star_radec: RA,Dec sky position of reference star. [2-list of floats]
    :return: dataframe updated with X and Y offsets (relative to first image). [pandas DataFrame]
    """
    xy0_list = [im.wcs.all_world2pix(ref_star_radec, 0, ra_dec_order=True)[0] for im in df_images['Image']]
    df_images['X_offset'] = [xy[0] - (xy0_list[0])[0] for xy in xy0_list]
    df_images['Y_offset'] = [xy[1] - (xy0_list[0])[1] for xy in xy0_list]
    # print("Pre-alignment image offsets from first image:")
    # for i in df_images.index:
    #     print('   ', df_images.loc[i, 'Filename'],
    #           '{0:.3f}'.format(df_images.loc[i, 'X_offset']),
    #           '{0:.3f}'.format(df_images.loc[i, 'Y_offset']))
    print('calc_image_offsets() done.')
    return df_images


def crop_to_mid_images(df_images, ref_star_radec):
    """ Crop full images to much smaller mid-sized_images, used (for speed) for alignment and convolutions.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :param ref_star_radec:
    :return: dataframe with new 'Mid_image' column. New images have updated WCS. [pandas DataFrame]
    """
    # Beginning, minimal bounding box (to be enlarged) must contain MP locations from all images, and
    #    reference star, too. We will use first image, because X- and Y-offsets are relative to that.
    mp_ra_earliest, mp_dec_earlies = (df_images.iloc[0])['MP_RA'], (df_images.iloc[0])['MP_Dec']
    mp_ra_latest, mp_dec_latest = (df_images.iloc[-1])['MP_RA'], (df_images.iloc[-1])['MP_Dec']
    mp_xy0_earliest = df_images.iloc[0]['Image'].wcs.all_world2pix([[mp_ra_earliest, mp_dec_earlies]], 0,
                                                                   ra_dec_order=True)[0]
    mp_xy0_latest = df_images.iloc[0]['Image'].wcs.all_world2pix([[mp_ra_latest, mp_dec_latest]], 0,
                                                                 ra_dec_order=True)[0]
    ref_star_xy0 = df_images.iloc[0]['Image'].wcs.all_world2pix(ref_star_radec, 0,
                                                                ra_dec_order=True)[0]
    min_x0 = min(mp_xy0_earliest[0], mp_xy0_latest[0], ref_star_xy0[0])  # beginning bounding box only.
    max_x0 = max(mp_xy0_earliest[0], mp_xy0_latest[0], ref_star_xy0[0])  # "
    min_y0 = min(mp_xy0_earliest[1], mp_xy0_latest[1], ref_star_xy0[1])  # "
    max_y0 = max(mp_xy0_earliest[1], mp_xy0_latest[1], ref_star_xy0[1])  # "

    # Calculate dimensions of full bounding box for mid-images:
    bb_min_x0 = int(round(min_x0 + min(df_images['X_offset']) -
                          2 * KERNEL_NOMINAL_SIZE - MID_IMAGE_PADDING))
    bb_max_x0 = int(round(max_x0 + max(df_images['X_offset']) +
                          2 * KERNEL_NOMINAL_SIZE + MID_IMAGE_PADDING))
    bb_min_y0 = int(round(min_y0 + min(df_images['Y_offset']) -
                          2 * KERNEL_NOMINAL_SIZE - MID_IMAGE_PADDING))
    bb_max_y0 = int(round(max_y0 + max(df_images['Y_offset']) +
                          2 * KERNEL_NOMINAL_SIZE + MID_IMAGE_PADDING))
    # print('Mid_image crop:    x: ', str(bb_min_x0), str(bb_max_x0),
    #       '      y: ', str(bb_min_y0), str(bb_max_y0))

    # Perform the crop, save to new 'Mid_image' column:
    df_images['Mid_image'] = None
    for i, im in zip(df_images.index, df_images['Image']):
        # Must reverse the two axes when using np arrays directly:
        df_images.loc[i, 'Mid_image'] = trim_image(im[bb_min_y0:bb_max_y0, bb_min_x0:bb_max_x0])
    # print((df_images.iloc[0])['Mid_image'].wcs.printwcs())
    # plot_images('Raw subimages', df_images['Filename'], df_images['Mid_image'])
    print('crop_to_mid_images() done.')
    return df_images


def align_mid_images(df_images):
    """ Align (reposition images, update WCS) the mid-sized images. OVERWRITE column 'Mid_image'.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe with 'Mid_image' column OVERWRITTEN by newly aligned mid-images. [pandas DataFrame]
    """
    # Align all mid-images to first mid-image:
    first_image_wcs = (df_images.iloc[0])['Mid_image'].wcs
    for i, im in zip(df_images.index, df_images['Mid_image']):
        df_images.loc[i, 'Mid_image'] = wcs_project(im, first_image_wcs)

    # # print("Post-alignment pix position of first image's MP RaDec (should be uniform):")
    # mp_ra_ref, mp_dec_ref = (df_images.iloc[0])['MP_RA'], (df_images.iloc[0])['MP_Dec']
    # for i in df_images.index:
    #     mp_pix_ref = df_images.loc[i, 'Mid_image'].wcs.all_world2pix([[mp_ra_ref, mp_dec_ref]], 0,
    #                                                                  ra_dec_order=True)[0]
        # print('   ', df_images.loc[i, 'Filename'], str(mp_pix_ref))  # s/be uniform across all images.
    # plot_images('Aligned mid-images', df_images['Filename'], df_images['Mid_image'])
    print('align_mid_images() done.')
    return df_images


def recrop_mid_images(df_images):
    """ Find largest rectangle of pixels having no masked or NaN values,
        then crop all (aligned) mid-values to that.
    :param df_images:
    :return: df_images with mid-images trimmed to contain no NaN values.
    """
    # Establish and write initial situation:
    pixel_sum = np.sum(im.data for im in df_images['Mid_image'])  # NaN if any images has NaN at that pix.
    # print('    Pre-recrop:', str(np.sum(np.isnan(pixel_sum))),
    #       'nan pixels of', str(pixel_sum.shape[1]), 'x', str(pixel_sum.shape[0]))

    # Set initial (prob. too large) x- and y-limits from prev. calculated offsets:
    ref_mid_image = df_images.iloc[0]['Mid_image']
    x0_min = max(0, int(floor(0 - min(df_images['X_offset']))) - 5)
    x0_max = min(ref_mid_image.shape[1], int(ceil(ref_mid_image.shape[1] - max(df_images['X_offset']))) + 5)
    y0_min = max(0, int(floor(0 + min(df_images['Y_offset']))) - 5)
    y0_max = min(ref_mid_image.shape[0], int(ceil(ref_mid_image.shape[0] - max(df_images['Y_offset']))) + 5)
    # print('    Rough recrop:', str(np.sum(np.isnan(pixel_sum[y0_min:y0_max, x0_min:x0_max]))),
    #       'nan pixels of', str(x0_max-x0_min), 'x', str(y0_max-y0_min))

    # Repeat a spiral trim, until proposed trimmed image has no more NaNs:
    while min(x0_max - x0_min, y0_max - y0_min) > 2 * KERNEL_NOMINAL_SIZE:
        before_limits = (x0_min, x0_max, y0_min, y0_max)
        if np.any(np.isnan(pixel_sum[y0_min:y0_min+1, x0_min:x0_max])):  # if top row has any NaN.
            y0_min += 1
        if np.any(np.isnan(pixel_sum[y0_max-1:y0_max, x0_min:x0_max])):  # if bottom row has any NaN.
            y0_max -= 1
        if np.any(np.isnan(pixel_sum[y0_min:y0_max, x0_min:x0_min+1])):  # if leftmost column has any NaN.
            x0_min += 1
        if np.any(np.isnan(pixel_sum[y0_min:y0_max, x0_max-1:x0_max])):  # if rightmost column has any NaN.
            x0_max -= 1
        after_limits = (x0_min, x0_max, y0_min, y0_max)
        # print('    Loop recrop:', str(np.sum(np.isnan(pixel_sum[y0_min:y0_max, x0_min:x0_max]))),
        #       'nan pixels of', str(x0_max-x0_min), 'x', str(y0_max-y0_min))
        if after_limits == before_limits:  # if no trims made in this iteration.
            break
    if np.any(np.isnan(pixel_sum[y0_min:y0_max, x0_min:x0_max])):
        print(' >>>>> ERROR: recrop_mid_images() did not succeed in removing all NaNs.')
    # else:
    #     print('    Final recrop:', str(np.sum(np.isnan(pixel_sum[y0_min:y0_max, x0_min:x0_max]))),
    #           'nan pixels of', str(x0_max-x0_min), 'x', str(y0_max-y0_min))

    # Do mid-image trims, based on the best limits found:
    recropped_mid_images = [trim_image(im[y0_min:y0_max, x0_min:x0_max]) for im in df_images['Mid_image']]
    df_images['Mid_image'] = recropped_mid_images
    # plot_images('Recropped mid-images', df_images['Filename'], df_images['Mid_image'])
    print('recrop_mid_images() done.')
    return df_images


def make_ref_star_psfs(df_images, ref_star_radec, nominal_size=KERNEL_NOMINAL_SIZE):
    """ Make ref star PSFs (very small kernel-sized images, bkdg-subtracted) and store them in df_images.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :param ref_star_radec: RA,Dec of reference star in degrees. [2-list of floats]
    :param nominal_size: pixels per (square) kernel's side; will be forced odd if not already. [int]
    :return: small array centered on ref-star, suitable as psf to make matching kernel. [numpy array]
    """
    half_size = int(nominal_size / 2.0)
    xy0_list = [tuple(im.wcs.all_world2pix(ref_star_radec, 0, ra_dec_order=True)[0])
                for im in df_images['Mid_image']]
    # print('Top of make_ref_star_psfs(), ref star positions (from radec):')
    # for (fn, xy0) in zip(df_images['Filename'], xy0_list):
    #     print('   ', fn, '{:.3f}'.format(xy0[0]), '{:.3f}'.format(xy0[1]))

    # TODO: Suspected math error: centers keep ending up at 41-42 rather than at ~ 40.
    x_centers = [int(round(xy0[0])) for xy0 in xy0_list]
    y_centers = [int(round(xy0[1])) for xy0 in xy0_list]
    x_mins = [xc - half_size for xc in x_centers]
    x_maxs = [xc + half_size + 1 for xc in x_centers]
    y_mins = [yc - half_size for yc in y_centers]
    y_maxs = [yc + half_size + 1 for yc in y_centers]
    arrays = [im.data[y_min:y_max, x_min:x_max].copy()
              for (im, x_min, x_max, y_min, y_max)
              in zip(df_images['Mid_image'], x_mins, x_maxs, y_mins, y_maxs)]
    medians = [sigma_clipped_stats(arr, sigma=3.0)[1] for arr in arrays]
    bkgd_subtracted_arrays = [array - median for (array, median) in zip(arrays, medians)]
    shape = bkgd_subtracted_arrays[0].shape
    window = (SplitCosineBellWindow(alpha=0.5, beta=0.25))(shape)  # window fn of same shape as kernels.
    raw_kernels = [bsa * window for bsa in bkgd_subtracted_arrays]
    normalized_kernels = [rk / np.sum(rk) for rk in raw_kernels]
    # TODO: Maybe trim to zero, for speed and possibly fewer artifacts.
    df_images['RefStarPSF'] = normalized_kernels
    # plot_images('Ref Star PSF', df_images['Filename'], df_images['RefStarPSF'])
    # print('Ref Star PSF x,y centroids:')
    # for (fn, psf) in zip(df_images['Filename'], df_images['RefStarPSF']):
    #     xc0, yc0 = centroid_com(psf)
    #     print('   ', fn, '{:.3f}'.format(xc0), '{:.3f}'.format(yc0))
    print('make_ref_star_psfs() done.')
    return df_images


def calc_target_kernel_sigma(df_images):
    """ Calculate target sigma for Gaussian target kernel.
        Should be just a little larger than ref_star's (largest) PSF across all the images.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :param ref_star_radec: RA,Dec of reference star in degrees. [2-list of floats]
    :return: good estimate of sigma for target kernel. [float]
    """
    sigma_list = []
    # print('Ref star PSF x, y, sigma:')
    for i, im in enumerate(df_images['RefStarPSF']):
        dps = data_properties(im)  # photutils.segmentation.SourceProperties object.
        sigma_list.append(dps.semimajor_axis_sigma.value)
        print('   ', df_images.iloc[i]['Filename'],
              '{:.2f}'.format(dps.xcentroid.value),
              '{:.2f}'.format(dps.ycentroid.value),
              '{:.3f}'.format(dps.semimajor_axis_sigma.value))
    max_sigma = max(sigma_list)
    target_sigma = 1.05 * max_sigma
    print('calc_target_kernel_sigma() done. Target sigma:', '{:.3f}'.format(target_sigma))
    return target_sigma


def make_matching_kernels(df_images, target_sigma):
    """ Make matching kernels to render subimages' ref_star Gaussian with target_sigma.
        Use mid-images here rather than subimages later, to minimize risk of bumping into image boundaries.
        Also, center target PSF on Ref Star PSF, to eliminate shifting that would slightly blur the
        averaged (MP-free) image made later, as well as shift the individual images from the averaged one.
    :param df_images: [pandas DataFrame]
    :param target_sigma: in pixels. [float]
    :return: df_images with new column 'MatchingKernel' with needed matching kernels. [pandas DataFrame]
    """
    matching_kernels = []
    # This would be too hard in list comprehensions:
    # print('In make_matching_kernels(), centroids of Ref Star PSF, target PSF, and matching kernel:')
    for (i, refstar_psf) in enumerate(df_images['RefStarPSF']):
        # Get centroids for source (ref star) PSFs:
        edge_length = refstar_psf.shape[0]
        y, x = np.mgrid[0:edge_length, 0:edge_length]
        x_center0, y_center0 = centroid_com(refstar_psf)
        # Make target PSFs (Gaussian):
        gaussian = Gaussian2D(1, x_center0, y_center0, target_sigma, target_sigma)
        target_psf = gaussian(x, y)
        target_psf /= np.sum(target_psf)  # ensure normalized.
        matching_kernel = create_matching_kernel(refstar_psf, target_psf, CosineBellWindow(alpha=0.35))
        matching_kernels.append(matching_kernel)
        # Print some position diagnostics (testing):
        x_target0, y_target0 = centroid_com(target_psf)
        x_matching0, y_matching0 = centroid_com(matching_kernel)
        # print('   ', df_images.iloc[i]['Filename'],
        #       '    refstar:', '{:.3f}'.format(x_center0), '{:.3f}'.format(y_center0),
        #       '    target:', '{:.3f}'.format(x_target0), '{:.3f}'.format(y_target0),
        #       '    matching:', '{:.3f}'.format(x_matching0), '{:.3f}'.format(y_matching0))
    df_images['MatchingKernel'] = matching_kernels
    # plot_images('Matching Kernels', df_images['Filename'], df_images['MatchingKernel'])
    print('make_matching_kernels() done.')
    return df_images


def convolve_mid_images(df_images):
    """ Convolve mid-images with matching kernel, to make source PSFs close to the target Gaussian PSF.
    :param df_images: [pandas DataFrame]
    :return: df_images with new column 'Mid_image_convolved'. [pandas DataFrame]
    """
    # Note that astropy convolve() operates on NDData arrays, not on CCDData objects.
    # So we operate on im.data, not on im itself.
    # Use a loop because I'm not entirely sure a list comprehension will get the job done:
    convolved_images = []
    for (im, kernel, fn) in zip(df_images['Mid_image'], df_images['MatchingKernel'], df_images['Filename']):
        print(fn + ': convolving mid-image...')
        mid_image_copy = im.copy()
        convolved_array = convolve(mid_image_copy.data, kernel, boundary='extend')
        mid_image_copy.data = convolved_array
        convolved_images.append(mid_image_copy)
    df_images['Mid_image_convolved'] = convolved_images
    # plot_images('Convolved mid-images', df_images['Filename'], df_images['Mid_image_convolved'])
    # diffs = [CCDData.subtract(conv, mid)
    #          for (conv, mid) in zip(df_images.iloc[0:4]['Mid_image_convolved'],
    #                                 df_images.iloc[8:12]['Mid_image_convolved'])]
    # plot_images('Diffs', df_images['Filename'], diffs)
    print('convolve_mid_images() done.')
    return df_images


def crop_to_subimages(df_images, ref_star_radec):
    """ Trim mid-images to the final, smallest size, using a small bounding box. Used for all MP photometry.
    :param df_images: dataframe, one row per session image in photometric filter.
                      Mid-images are assumed of uniform size, aligned, and convolved. [pandas DataFrame]
    :param ref_star_radec:
    :return: dataframe with new 'Subimage' column holding smallest images.
             New images have updated WCS. [pandas DataFrame]
    """
    # Find bounding box based on first and last mid-images, regardless of direction of MP motion:
    # Mid-images are aligned, so first and last mid-image will suffice.
    mid_image_first = (df_images.iloc[0])['Mid_image_convolved']
    mid_image_last = (df_images.iloc[-1])['Mid_image_convolved']
    mp_ra_first, mp_dec_first = (df_images.iloc[0])['MP_RA'], (df_images.iloc[0])['MP_Dec']
    mp_ra_last, mp_dec_last = (df_images.iloc[-1])['MP_RA'], (df_images.iloc[-1])['MP_Dec']
    mp_xy0_first = mid_image_first.wcs.all_world2pix([[mp_ra_first, mp_dec_first]], 0,
                                                     ra_dec_order=True)[0]
    mp_xy0_last = mid_image_last.wcs.all_world2pix([[mp_ra_last, mp_dec_last]], 0,
                                                   ra_dec_order=True)[0]
    ref_star_xy0 = df_images.iloc[0]['Mid_image_convolved'].wcs.all_world2pix(ref_star_radec, 0,
                                                                              ra_dec_order=True)[0]
    bb_x0_min = int(round(min(mp_xy0_first[0], mp_xy0_last[0], ref_star_xy0[0]) - SUBIMAGE_PADDING))
    bb_x0_max = int(round(max(mp_xy0_first[0], mp_xy0_last[0], ref_star_xy0[0]) + SUBIMAGE_PADDING))
    bb_y0_min = int(round(min(mp_xy0_first[1], mp_xy0_last[1], ref_star_xy0[1]) - SUBIMAGE_PADDING))
    bb_y0_max = int(round(max(mp_xy0_first[1], mp_xy0_last[1], ref_star_xy0[1]) + SUBIMAGE_PADDING))
    # print('Subimage crop:    x: ', str(bb_x0_min), str(bb_x0_max),
    #       '      y: ', str(bb_y0_min), str(bb_y0_max))

    # Perform the trims, save to new 'Subimage' column:
    df_images['Subimage'] = None
    for i, im in zip(df_images.index, df_images['Mid_image_convolved']):
        # Must reverse the two axes when using np arrays directly:
        df_images.loc[i, 'Subimage'] = trim_image(im[bb_y0_min:bb_y0_max, bb_x0_min:bb_x0_max])
    # print('\n\nFirst subimage WCS:\n')
    # (df_images.iloc[0])['Subimage'].wcs.printwcs()
    # print('\n\nLast subimage WCS:\n')
    # (df_images.iloc[-1])['Subimage'].wcs.printwcs()
    # plot_images('Newly cropped subimages', df_images['Filename'], df_images['Subimage'])
    print('crop_to_subimages() done.')
    return df_images


def calc_first_ref_star_xy0(df_images, ref_star_radec):
    """ Calculate first reference star position in pixels, using ref star RA,Dec, for each image.
        This will be the reference position to which misalignment corrections are added in the loop.
    :param df_images:
    :param ref_star_radec:
    :return:
    """
    xy0_list = [tuple(im.wcs.all_world2pix(ref_star_radec, 0, ra_dec_order=True)[0])
                for im in df_images['Subimage']]
    df_images['FirstRefStar_XY0'] = xy0_list
    print('calc_first_ref_star_xy0() done.')
    return df_images


def calc_first_mp_xy0(df_images):
    """ Calculate first MP positions in pixels, using each image's MP RA,Dec, for each image.
        This will be the reference position to which misalighment corrections are added to mask MPs.
    :param df_images:
    :return:
    """
    xy0_list = []
    for i, im in enumerate(df_images['Subimage']):
        x0, y0 = tuple(im.wcs.all_world2pix([[df_images.iloc[i]['MP_RA'],
                                              df_images.iloc[i]['MP_Dec']]], 0, ra_dec_order=True)[0])
        xy0_list.append((x0, y0))
    df_images['FirstMP_XY0'] = xy0_list
    print('calc_first_mp_xy0() done.')
    return df_images


def extract_first_subarrays(df_images):
    """ Extract masked data array from each CCDData object; from now on we use only masked ndarrays,
        as we have no more use for WCS information. We will have to be clever about getting pixel positions.
    :param df_images:
    :return:
    """
    df_images['Subarray_first'] = [im.data.copy() for im in df_images['Subimage']]     # immutable.
    df_images['Subarray_current'] = [sf.copy() for sf in df_images['Subarray_first']]  # to start the loop.
    df_images['Shift_current'] = len(df_images) * [(0, 0)]  # to be overwritten, each loop.
    df_images['Misalignment'] = None                        # to be overwritten, each loop.
    print('extract_first_subarrays() done.')
    return df_images


def estimate_misalignments(df_images):
    """ Calculate best estimate (for each image) of vector (dx,dy in pixels, to be used in convolution
        of first_subarray) to put image's ref-star centroid directly on first_subarray's average ref star
        pixel position.
    :param df_images:
    :return:
    """
    first_x0_list = [xy0[0] for xy0 in df_images['FirstRefStar_XY0']]
    first_y0_list = [xy0[1] for xy0 in df_images['FirstRefStar_XY0']]

    # Calculate ref star centroid position for each current subarray:
    ref_star_xy0_list = []
    centroid_cell_half_size = 20
    centroid_cells = []
    for i, ar in enumerate(df_images['Subarray_current']):
        first_x0 = first_x0_list[i]
        first_y0 = first_y0_list[i]
        shift_x = df_images.iloc[i]['Shift_current'][0]
        shift_y = df_images.iloc[i]['Shift_current'][1]
        current_ref_star_x0 = first_x0 + shift_x
        current_ref_star_y0 = first_y0 + shift_y
        cell_base_x0 = int(current_ref_star_x0) - centroid_cell_half_size
        cell_base_y0 = int(current_ref_star_y0) - centroid_cell_half_size

        # Must background-subtract before taking centroid [nsigma=2 is very aggressive mask: ok]:
        source_mask = make_source_mask(ar, nsigma=2, npixels=5, filter_fwhm=2, dilate_size=11)
        _, median, _ = sigma_clipped_stats(ar, sigma=3.0, mask=source_mask)
        ar_bkgd_subtr = ar.copy() - median
        # Make cell ~centered around Ref Star, and take centroid of it (i.e., of the star):
        centroid_cell = ar_bkgd_subtr[cell_base_y0:
                                      cell_base_y0 + 2 * centroid_cell_half_size,
                                      cell_base_x0:
                                      cell_base_x0 + 2 * centroid_cell_half_size]  # nb: indices y, x.
        centroid_cells.append(centroid_cell)
        cell_x0, cell_y0 = centroid_com(centroid_cell)
        # print('Cell_centroid', df_images.iloc[i]['Filename'],
        #       '{:3f}'.format(cell_x0), '{:3f}'.format(cell_y0),
        #       '-->', '{:3f}'.format(cell_x0 + cell_base_x0), '{:3f}'.format(cell_y0 + cell_base_y0))
        ref_star_x0 = cell_x0 + cell_base_x0
        ref_star_y0 = cell_y0 + cell_base_y0
        ref_star_xy0_list.append((ref_star_x0, ref_star_y0))  # tuple.
    # plot_images('Centroid Cells', df_images['Filename'], centroid_cells)

    # Calculate misalignment (x,y: true - target) for each image:
    target_x0, target_y0 = calc_average_starting_ref_star_position(df_images)
    print('Target x0, y0:', '{:3f}'.format(target_x0), '{:3f}'.format(target_y0))
    x_misalignments = [rspp[0] - target_x0 for rspp in ref_star_xy0_list]
    y_misalignments = [rspp[1] - target_y0 for rspp in ref_star_xy0_list]
    df_images['Misalignment'] = [xy for xy in zip(x_misalignments, y_misalignments)]
    for i, mis in enumerate(df_images['Misalignment']):
        print('Misalignment:', df_images.iloc[i]['Filename'],
              '{:3f}'.format(mis[0]), '{:3f}'.format(mis[1]))
    sum_square_misalignments = sum([mis[0]*mis[0] + mis[1]*mis[1] for mis in df_images['Misalignment']])
    rms_misalignment = sqrt(sum_square_misalignments / len(df_images))
    print('estimate_misalignments(): rms_misalignment=',
          '{:.1f}'.format(1000.0 * rms_misalignment), 'millipixels.')
    print('estimate_misalignments() done.')
    return df_images


def convolve_subarrays(df_images):
    """ Do convolution of first

    :param df_images:
    :return:
    """
    ALIGNMENT_KERNEL_EDGE_SIZE = 21  # should be odd
    half_size = ALIGNMENT_KERNEL_EDGE_SIZE // 2
    y, x = np.mgrid[0:ALIGNMENT_KERNEL_EDGE_SIZE, 0:ALIGNMENT_KERNEL_EDGE_SIZE]
    convolved_subarrays, new_shifts = [], []
    for ar, mis, sc in zip(df_images['Subarray_first'], df_images['Misalignment'],
                           df_images['Shift_current']):
        new_shift_x = sc[0] - mis[0]
        new_shift_y = sc[1] - mis[1]
        gaussian = Gaussian2D(1, half_size + new_shift_x, half_size + new_shift_y, 0.7, 0.7)
        kernel = gaussian(x, y)
        kernel /= np.sum(kernel)
        convolved_subarrays.append(convolve(ar, kernel))
        new_shifts.append((new_shift_x, new_shift_y))
    df_images['Subarray_current'] = convolved_subarrays
    df_images['Shift_current'] = new_shifts
    # plot_images('Convolved subarrays', df_images['Filename'], convolved_subarrays)
    print('convolve_subarrays() done.')
    return df_images


def calc_current_mp_xy0(df_images):
    """ Calculate MP (x,y) pixel position for each subarray, save in dataframe as new columns.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe with new 'MP_XY0_current' columns. [pandas DataFrame]
    """
    new_mp_xy_list = []
    for f, sc in zip(df_images['FirstMP_XY0'], df_images['Shift_current']):
        new_mp_x = f[0] + sc[0]
        new_mp_y = f[1] + sc[1]
        new_mp_xy_list.append((new_mp_x, new_mp_y))
    df_images['MP_XY0_current'] = new_mp_xy_list
    print('calc_current_mp_xy0() done.')
    return df_images


def mask_mp_from_current_subarrays(df_images, target_sigma):
    """ Make MP-masked subarrays; not yet background-subtracted. Save in new column 'Subarray_masked'.
        (We do this before calculating background statistics, to make them just a bit more accurate.)
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :param target_sigma: Gaussian sigma to which all subarrays have previously been convolved. [float]
    :return: same dataframe with new 'Subarray_masked' column. [pandas DataFrame]
    """
    mp_mask_radius = MP_MASK_RADIUS_PER_SIGMA * target_sigma
    mp_masked_subarrays = []
    for i, ar in enumerate(df_images['Subarray_current']):
        mp_masked_subarray = np.ma.array(data=ar.copy(), mask=False, copy=True)
        mp_x0, mp_y0 = df_images.iloc[i]['MP_XY0_current']
        mp_masked_subarray.mask = np.fromfunction(lambda i, j:
                                                  ((j - mp_x0)**2 + (i - mp_y0)**2) <= mp_mask_radius**2,
                                                  shape=mp_masked_subarray.data.shape)
        mp_masked_subarrays.append(mp_masked_subarray)
    df_images['Subarray_masked'] = mp_masked_subarrays
    # plot_images('Subarray masks', df_images['Filename'], [ar.mask for ar in df_images['Subarray_masked']])
    # plot_mp_masks(df_images)
    print('mask_mp_from_current_subarrays() done.')
    return df_images


def calc_subarray_backgrounds(df_images):
    """ For each subarray, calculate sigma-clipped mean, median background ADU levels & std. deviation;
        write to dataframe as new columns. We will use the sigma-clipped median as background ADUs.
        This all assumes constant background across this small subarray...go to 2-D later if really needed.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe with new 'Bkgd_mean', 'Bkgd_median', and 'Bkgd_std' columns. [pandas DataFrame]
    """
    source_masks = [make_source_mask(ar.data, nsigma=2, npixels=5, filter_fwhm=2, dilate_size=11)
                    for ar in df_images['Subarray_masked']]  # nsigma=2 is very aggressive mask: ok.
    # Logical-or masks each pixel masked by *either* mask (MP mask or source-ADU mask):
    mp_source_masks = [np.logical_or(ar.mask, source_mask)
                       for (ar, source_mask) in zip(df_images['Subarray_masked'], source_masks)]
    stats = [sigma_clipped_stats(ar.data, sigma=3.0, mask=b_mask)
             for (ar, b_mask) in zip(df_images['Subarray_masked'], mp_source_masks)]
    df_images['Bkgd_mean'], df_images['Bkgd_median'], df_images['Bkgd_std'] = tuple(zip(*stats))
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.set_title('MP+Source mask: ' + df_images.iloc[0]['Filename'])
    # ax1.imshow(mp_source_masks[0], origin='upper')
    # ax2.set_title('MP+Source mask: ' + df_images.iloc[-1]['Filename'])
    # ax2.imshow(mp_source_masks[-1], origin='upper')
    # plt.tight_layout()
    # plt.show()
    print('calc_subarray_backgrounds() done.')
    return df_images


def subtract_background_from_subarrays(df_images):
    """ Make background-subtracted (still MP-masked) subarrays ready for use in our actual photometry.
        Add separate, new column 'Subarray_bkgd_subtr'; does not alter other df_image columns.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: same dataframe with new 'Subarrray_bkgd_subtr' column. [pandas DataFrame]
    """
    # Use a loop because I'm not entirely sure a list comprehension will get the job done:
    subarrays_bkgd_subtr = []
    for (ar, bkgd_median) in zip(df_images['Subarray_masked'], df_images['Bkgd_median']):
        bkgd_array = np.full_like(ar, bkgd_median)
        ar_bkgd_subtr = np.ma.masked_array(data=ar.data.copy() - bkgd_array, mask=ar.mask)
        subarrays_bkgd_subtr.append(ar_bkgd_subtr)
    df_images['Subarray_bkgd_subtr'] = subarrays_bkgd_subtr

    # plot_images('Subimages, bkgd-subtr', df_images['Filename'], df_images['Subarray_bkgd_subtr'])
    print('subtract_background_from_subarrays() done.')
    return df_images


def make_averaged_subarray(df_images):
    """ Make and return reference (background-subtracted, MP-masked) averaged subarray.
        This is what this sliver of sky would look like, on average, if the MP were not there.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: one averaged, MP-masked subarray. [one numpy Masked Array]
    """

    subarrays_as_ccddata = [CCDData(data=sa.data, mask=sa.mask, unit='adu')
                            for sa in df_images['Subarray_bkgd_subtr']]
    combiner = Combiner(subarrays_as_ccddata)
    averaged_ccddata = combiner.average_combine()
    averaged_subarray = np.ma.masked_array(data=averaged_ccddata.data, mask=averaged_ccddata.mask)
    n_masked_out = np.sum(averaged_subarray.mask)
    if n_masked_out > 0:
        print(' >>>>> WARNING: averaged_subarray has', str(n_masked_out), 'pixels masked out.')
    plot_averaged_subarray(df_images, averaged_subarray)
    print('make_averaged_subarray() done.')
    return averaged_subarray


def decompose_subarrays(df_images, averaged_subarray):
    """ For each background-subtracted, MP-masked subarray, find:
        source_factor, relative to averaged subarray source fluxes, and
        background_offset, relative to averaged subarray background.
    :param df_images:
    :param averaged_subarray:
    :return:
    """
    fit = fitting.LinearLSQFitter()
    line_init = models.Linear1D()

    # Crop out edge pixels (~ 5 from each border) from x & y (avgd_sa has edge pixels masked out).
    x_cropped = averaged_subarray[EDGE_PIXELS_REGRESSION:-EDGE_PIXELS_REGRESSION,
                                  EDGE_PIXELS_REGRESSION:-EDGE_PIXELS_REGRESSION]
    x_raveled = np.ravel(x_cropped.data)
    source_factors, background_offsets = [], []
    for i, ar in enumerate(df_images['Subarray_bkgd_subtr']):
        y_cropped = ar[EDGE_PIXELS_REGRESSION:-EDGE_PIXELS_REGRESSION,
                       EDGE_PIXELS_REGRESSION:-EDGE_PIXELS_REGRESSION]
        y_raveled = np.ravel(y_cropped.data)
        to_keep = ~np.ravel(y_cropped.mask)
        x_fit = x_raveled[to_keep]
        y_fit = y_raveled[to_keep]
        # TODO: consider adding backgrounds' x- and y-gradients as fit parameters.
        # TODO: when done with above, also add plots & ADU sigma from the 2 different fits (per image).
        fitted_line = fit(line_init, x_fit, y_fit)
        source_factors.append(fitted_line.slope.value)
        background_offsets.append(fitted_line.intercept.value)
        # if i in [0, 1, 2, len(df_images) - 1]:
        #     plt.figure()
        #     plt.plot(x_fit, y_fit, 'ko')
        #     plt.show()
    df_images['SourceFactor'] = source_factors
    df_images['BackgroundOffset'] = background_offsets
    print('decompose_subarrays() done.')
    return df_images


def make_mp_only_subarrays(df_images, averaged_subarray):
    """ Make best-estimated of MP-only subarray by using aligned subarray,
            averaged subarray, source_factor, and background-offset.
    :param df_images:
    :param averaged_subarray:
    :return:
    """
    mp_only_subarrays = []
    for (sf, bo, sa, bk) in zip(df_images['SourceFactor'],
                                df_images['BackgroundOffset'],
                                df_images['Subarray_current'],
                                df_images['Bkgd_median']):
        sa_bg = np.ma.masked_array(data=sa.data, mask=np.ma.nomask) - bk
        best_background_subarray = averaged_subarray * sf + bo
        mp_only_subarray = sa_bg - best_background_subarray
        mp_only_subarrays.append(mp_only_subarray)
    df_images['Subarray_mp_only'] = mp_only_subarrays
    plot_images('MP-only subarrays', df_images['Filename'], df_images['Subarray_mp_only'])
    print('make_mp_only_subarrays() done.')
    return df_images


def evaluate_convergence(df_images):
    # Calculate root-mean-square-deviation of (new) ref star Misalignments.
    sum_square_misalignments = sum([mis[0]*mis[0] + mis[1]*mis[1] for mis in df_images['Misalignment']])
    rms_misalignment = sqrt(sum_square_misalignments / len(df_images))
    if rms_misalignment <= MAX_FOR_CONVERGENCE:  # Return True iff RMS <= convergence RMS.
        return True
    # Could add more criteria (either True or False) here.
    return False


def do_mp_aperture_photometry(df_images, target_sigma):
    """ Get MP net flux for each image.
    :param df_images:
    :param target_sigma: in pixels. [float]
    :return: df_images with 2 new columns of floatsL 'FluxMP' and 'FluxSigmaMP'.
    """
    mp_xy0_list = list(df_images['MP_XY0_current'])
    r_aperture = AP_PHOT_RADIUS_PER_SIGMA * target_sigma
    r_inner = r_aperture + 5.0
    r_outer = r_inner + 5.0
    print('aperture radii:', '{:.3f}'.format(r_aperture),
          '{:.3f}'.format(r_inner), '{:.3f}'.format(r_outer))
    apertures = CircularAperture(mp_xy0_list, r=r_aperture)

    # Aperture (MP flux) loop: unusual, as we use only one aperture per image:
    raw_flux_list = []
    for im, ap in zip(df_images['Subarray_mp_only'], apertures):
        phot = aperture_photometry(im, ap)
        raw_flux = phot[0]['aperture_sum']
        raw_flux_list.append(raw_flux)

    # Annulus (background) loop: also one annulus per image:
    annulae = CircularAnnulus(mp_xy0_list, r_in=r_inner, r_out=r_outer)
    annulus_masks = annulae.to_mask(method='center')
    bkgd_adu_list, ann_sigma_list = [], []
    for im, mask in zip(df_images['Subarray_mp_only'], annulus_masks):
        annulus_data = mask.multiply(im)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, ann_median, ann_sigma = sigma_clipped_stats(annulus_data_1d)
        bkgd_adu_list.append(ann_median)
        ann_sigma_list.append(ann_sigma)

    # Calculate and return flux and flux sigma values, each image:
    net_flux_list = [raw_flux - ap.area * bkgd_adu
                     for raw_flux, ap, bkgd_adu in zip(raw_flux_list, apertures, bkgd_adu_list)]
    sigma2_ap_list = [CCD_GAIN * (raw_flux + ap.area * bkgd_median)
                      for (raw_flux, ap, bkgd_median)
                      in zip(raw_flux_list, apertures, df_images['Bkgd_median'])]
    sigma2_ann_list = [ap.area * ann_sigma for ap, ann_sigma in zip(apertures, ann_sigma_list)]
    flux_sigma_list = [sqrt(sigma2_ap + sigma2_ann)
                       for sigma2_ap, sigma2_ann in zip(sigma2_ap_list, sigma2_ann_list)]
    df_images['FluxMP'] = net_flux_list
    df_images['FluxSigmaMP'] = flux_sigma_list

    # Now do the same for 'Image', to compare unbulldozed results with the above:
    mp_xy0_list = [im.wcs.all_world2pix([[mp_ra, mp_dec]], 0, ra_dec_order=True)[0]
                   for (im, mp_ra, mp_dec) in zip(df_images['Image'],
                                                  df_images['MP_RA'],
                                                  df_images['MP_Dec'])]
    r_aperture = MP_MASK_RADIUS_PER_SIGMA * target_sigma
    r_inner = r_aperture + 5.0
    r_outer = r_inner + 5.0
    apertures = CircularAperture(mp_xy0_list, r=r_aperture)

    # Aperture (MP flux) loop: unusual, as we use only one aperture per image:
    raw_flux_list = []
    for im, ap in zip(df_images['Image'], apertures):
        phot = aperture_photometry(im, ap)
        raw_flux = phot[0]['aperture_sum']
        raw_flux_list.append(raw_flux)

    # Annulus (background) loop: also one annulus per image:
    annulae = CircularAnnulus(mp_xy0_list, r_in=r_inner, r_out=r_outer)
    annulus_masks = annulae.to_mask(method='center')
    bkgd_adu_list, ann_sigma_list = [], []
    for im, mask in zip(df_images['Image'], annulus_masks):
        annulus_data = mask.multiply(im)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, ann_median, ann_sigma = sigma_clipped_stats(annulus_data_1d)
        bkgd_adu_list.append(ann_median)
        ann_sigma_list.append(ann_sigma)
    net_flux_list = [raw_flux.value - ap.area * bkgd_adu
                     for raw_flux, ap, bkgd_adu in zip(raw_flux_list, apertures, bkgd_adu_list)]
    sigma2_ap_list = [CCD_GAIN * (raw_flux.value + ap.area * bkgd_median)
                      for (raw_flux, ap, bkgd_median)
                      in zip(raw_flux_list, apertures, df_images['Bkgd_median'])]
    sigma2_ann_list = [ap.area * ann_sigma for ap, ann_sigma in zip(apertures, ann_sigma_list)]
    flux_sigma_list = [sqrt(sigma2_ap + sigma2_ann)
                       for sigma2_ap, sigma2_ann in zip(sigma2_ap_list, sigma2_ann_list)]
    df_images['FluxMP_raw'] = net_flux_list
    df_images['FluxSigmaMP_raw'] = flux_sigma_list

    # TODO: Add plots doing the comparison.
    x = df_images['JD_mid'] - floor(min(df_images['JD_mid']))
    y1 = df_images['FluxMP']
    err1 = df_images['FluxSigmaMP']
    y2 = df_images['FluxMP_raw']
    err2 = df_images['FluxSigmaMP_raw']

    # Plot both lightcurves (from bulldozer vs from images directly):
    def make_subplot(ax, x, y, y_err, title):
        ax.plot(x, y)
        ax.errorbar(x=x, y=y, yerr=y_err, fmt='none', color='black',
                    linewidth=0.5, capsize=3, capthick=0.5, zorder=-100)
        ax.set_title(title)
        ax.set_xlabel('JD_fract', labelpad=0)  # labelpad in points
        ax.set_ylabel('flux', labelpad=0)      # "
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))  # (w, h)
    make_subplot(ax1, x, y1, err1, 'Flux from bulldozer')
    make_subplot(ax2, x, y2, err2, 'Flux directly from images')
    y_min = min([min(yy) for yy in [y1, y2]])
    y_max = max([max(yy) for yy in [y1, y2]])
    y_min -= 0.04 * (y_max - y_min)
    y_max += 0.04 * (y_max - y_min)
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    plt.tight_layout(h_pad=20)
    plt.show()

    print('do_mp_aperture_photometry() done.')
    return df_images


PLOTTING_ETC_FUNCTIONS__________________________ = 0


def do_gamma(image, gamma=0.0625, dark_clip=0.002):
    im_min, im_max = np.min(image), np.max(image)
    # im_min += dark_clip * (im_max - im_min)
    im_scaled = np.clip((image - im_min) / (im_max - im_min), dark_clip, 1.0)
    return np.power(im_scaled, gamma)


def plot_images(figtitle, name_list, image_list):
    plots_per_figure = 4
    plots_per_row = 2
    rows_per_figure = plots_per_figure // plots_per_row
    axes = None  # keep IDE happy.
    plot_minimum = np.median([np.min(im) for im in image_list])
    plot_maximum = max([np.max(im) for im in image_list])
    # norm = ImageNormalize(vmin=plot_minimum, vmax=plot_maximum, stretch=LogStretch(1000.0))
    norm = ImageNormalize(vmin=plot_minimum, vmax=plot_maximum, stretch=LogStretch(200.0))
    print('Plot (' + figtitle + '):', '{:.3f}'.format(plot_minimum), '{:.3f}'.format(plot_maximum))
    for i, im in enumerate(image_list):
        _, i_plot_this_figure = divmod(i, plots_per_figure)
        i_row, i_col = divmod(i_plot_this_figure, plots_per_row)
        if i_plot_this_figure == 0:
            fig, axes = plt.subplots(ncols=plots_per_row, nrows=rows_per_figure,
                                     figsize=(7, 4 * rows_per_figure))
            fig.suptitle(figtitle)
        ax = axes[i_row, i_col]
        ax.set_title(name_list.iloc[i])
        im_plot = im.copy()
        ax.imshow(im_plot, origin='upper', cmap='Greys', norm=norm)
        if i_plot_this_figure == plots_per_figure - 1:
            plt.show()


def plot_first_last_subimages(df_images):
    """ For testing. """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title(df_images.iloc[0]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[0]['Subimage']), origin='upper', interpolation='none', cmap='Greys')
    ax2.set_title(df_images.iloc[-1]['Filename'])
    ax2.imshow(do_gamma(df_images.iloc[-1]['Subimage']), origin='upper', interpolation='none', cmap='Greys')
    plt.tight_layout()
    plt.show()


# def print_ref_star_morphology(df_images, ref_star_radec):
#     # This should use mid_images rather than subimages; less risk of bumping into image boundaries.
#     ims = df_images['Mid_image']
#     ref_star_xy0 = tuple(df_images.iloc[0]['Mid_image'].wcs.all_world2pix(ref_star_radec, 0,
#                                                                           ra_dec_order=True)[0])
#     print('ref_star_xy0 =', str(ref_star_xy0))
#     x, y = ref_star_xy0
#     x_min = int(floor(x - 25))
#     x_max = int(ceil(x + 25))
#     y_min = int(floor(y - 25))
#     y_max = int(ceil(y + 25))
#     plot_data = []
#     for i, im in enumerate(df_images['Mid_image']):
#         data = im.data[y_min:y_max, x_min:x_max].copy()  # x & y reversed for direct mp.array access
#         _, median, _ = sigma_clipped_stats(data, sigma=3.0)
#         data -= median  # subtract background
#         cat = data_properties(data)
#         columns = ['id', 'xcentroid', 'ycentroid', 'semimajor_axis_sigma',
#                    'semiminor_axis_sigma', 'orientation']
#         tbl = cat.to_table(columns=columns)
#         tbl['xcentroid'].info.format = '.3f'
#         tbl['ycentroid'].info.format = '.3f'
#         tbl['semimajor_axis_sigma'].info.format = '.3f'
#         tbl['semiminor_axis_sigma'].info.format = '.3f'
#         tbl['orientation'].info.format = '.3f'
#         print('\n#####', df_images.iloc[i]['Filename'])
#         print(tbl)
#         plot_data.append(data)
#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.set_title(df_images.iloc[3]['Filename'])
#     ax1.imshow(plot_data[3], origin='upper', interpolation='none', cmap='Greys')
#     ax2.set_title(df_images.iloc[14]['Filename'])
#     ax2.imshow(plot_data[14], origin='upper', interpolation='none', cmap='Greys')
#     plt.tight_layout()
#     plt.show()


def plot_ref_star_kernels(df_images, ref_star_radec):
    """ For each large_bb subimage, plot original kernel and psf_matching_kernel. """
    # This should use mid_images rather than subimages; less risk of bumping into image boundaries.
    nominal_size = KERNEL_NOMINAL_SIZE
    target_sigma = 5.176  # largest semi-major axis
    #
    kernels = df_images['RefStarKernel']
    # kernels = [get_ref_star_kernel(im, ref_star_radec, nominal_size) for im in df_images['Mid_image']]
    make_matching_kernels(df_images, target_sigma)
    matching_kernels = df_images['MKernels']
    for (fn, mk) in zip(df_images['Filename'], matching_kernels):
        print(fn, ' min, max =', '{0:.6f}'.format(np.min(mk)), '{0:.6f}'.format(np.max(mk)))

    n_rows = len(kernels)
    rows_per_fig = 4
    norm = ImageNormalize(stretch=LogStretch())
    rows_this_fig = 0
    axes = None  # keep IDE happy.
    for i_row in range(n_rows):
        if rows_this_fig <= 0:
            fig, axes = plt.subplots(ncols=2, nrows=rows_per_fig, figsize=(5, 3 * rows_per_fig))
        ax = axes[i_row % rows_per_fig, 0]
        ax.set_title('Ur: ' + df_images.iloc[i_row]['Filename'])
        ax.imshow(kernels[i_row], norm=norm, cmap='Greys', origin='upper', interpolation='none')
        ax = axes[i_row % rows_per_fig, 1]
        # ax.set_title('Target kernel')
        # ax.imshow(target_kernel, norm=norm, cmap='viridis', origin='upper', interpolation='none')
        ax.set_title('Matching kernel')
        ax.imshow(matching_kernels[i_row], norm=norm, cmap='viridis', origin='upper')
        rows_this_fig += 1
        if rows_this_fig >= rows_per_fig:
            # plt.tight_layout()
            plt.show()
            rows_this_fig = 0
    if rows_this_fig >= 1:
        plt.tight_layout()
        plt.show()


def plot_mp_masks(df_images):
    # Use subimages only.
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Subimage ' + df_images.iloc[0]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[0]['Subimage']), origin='upper', interpolation='none', cmap='Greys')
    ax2.set_title('MP Mask')
    ax2.imshow(df_images.iloc[0]['Subarray_masked'].mask, origin='upper')
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Subimage ' + df_images.iloc[-1]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[-1]['Subimage']), origin='upper', interpolation='none', cmap='Greys')
    ax2.set_title('MP Mask')
    ax2.imshow(df_images.iloc[-1]['Subarray_masked'].mask, origin='upper')
    plt.tight_layout()
    plt.show()


def plot_averaged_subarray(df_images, averaged_subarray):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    im = df_images.iloc[-1]['Subarray_first']
    source_mask = make_source_mask(im, nsigma=2, npixels=5, filter_fwhm=2, dilate_size=11)
    _, median, _ = sigma_clipped_stats(im, sigma=3.0, mask=source_mask)
    plot_minimum = median
    plot_maximum = np.max(im)
    norm = ImageNormalize(vmin=plot_minimum, vmax=plot_maximum, stretch=LogStretch(250.0))
    ax1.set_title('Subimage ' + df_images.iloc[-1]['Filename'])
    ax1.imshow(im, origin='upper', cmap='Greys', norm=norm)

    source_mask = make_source_mask(averaged_subarray, nsigma=2, npixels=5, filter_fwhm=2, dilate_size=11)
    _, median, _ = sigma_clipped_stats(averaged_subarray, sigma=3.0, mask=source_mask)
    plot_minimum = median
    plot_maximum = np.max(averaged_subarray)
    norm = ImageNormalize(vmin=plot_minimum, vmax=plot_maximum, stretch=LogStretch(250.0))
    ax2.set_title('AVERAGED subimage')
    ax2.imshow(averaged_subarray, origin='upper', cmap='Greys', norm=norm)
    plt.tight_layout()
    plt.show()


SUPPORT_FUNCTIONS_______________________________ = 0


def calc_average_starting_ref_star_position(df_images):
    # These are the individual ref star pixel positions:
    first_x0 = [xy0[0] for xy0 in df_images['FirstRefStar_XY0']]
    first_y0 = [xy0[1] for xy0 in df_images['FirstRefStar_XY0']]
    # This is *exactly* where we want all ref star centroids to fall. (Same for every loop iteration.)
    # Target is the average of initial ref star pixel positions over all images.
    target_x0 = sum(first_x0) / len(df_images)
    target_y0 = sum(first_y0) / len(df_images)
    return target_x0, target_y0


def background_subtract_array(array):
    """ Use source-masked, sigma-clipped statistics to subtract away background to zero.
    :param array: array to background-subtracted. [numpy array]
    :return: copy of array, background-subtracted. [numpy array]
    """
    copy = array.copy()
    bkgd_mask = make_source_mask(copy, nsigma=2, npixels=5, filter_fwhm=2, dilate_size=11)
    _, median, _ = sigma_clipped_stats(copy, sigma=3.0, mask=bkgd_mask)
    return copy - median


def add_test_mps(directory_path=TEST_FITS_DIR, filter='Clear', flux=100000, sigma=5):
    """ Add constant-flux, constant-shape, moving MP-like sources to all FITS files in a path.
        To each filename, add '_Added', so e.g., 'MP_191-0001-Clear_Added.fts'.
    :param directory_path:
    :param filter:
    :param flux:
    :param sigma:
    :return:
    """
    mp_file_early = 'MP_191-0001-Clear.fts'
    mp_file_late = 'MP_191-0028-Clear.fts'
    sources = [{'name': 'Sparse_bkgd',  'xy1_early': (1510, 698),  'xy1_late': (1746, 646)},
               {'name': 'Dense_bkgd',   'xy1_early': (1897.9, 989.1),  'xy1_late': (2233.0, 1141.0)},
               {'name': 'One brt star', 'xy1_early': (1368, 1544), 'xy1_late': (1587, 1533)}]
    half_size = int(floor(4 * sigma))
    edge_length = 2 * half_size + 1
    y, x = np.mgrid[0:edge_length, 0:edge_length]
    gaussian = Gaussian2D(1, half_size, half_size, sigma, sigma)
    source_psf = gaussian(x, y)
    source_psf *= (flux / np.sum(source_psf))  # set total area to desired flux.

    # To each image, add requested MP-like sources:
    df = start_df_images(directory_path, filter)
    hdu_list = [fits.open(os.path.join(TEST_FITS_DIR, fn))[0] for fn in df['Filename']]
    for source in sources:
        df = calc_mp_radecs(df, mp_file_early, mp_file_late, source['xy1_early'], source['xy1_late'])
        for i, ccddata in enumerate(df['Image']):
            x0, y0 = tuple(ccddata.wcs.all_world2pix([[df.iloc[i]['MP_RA'],
                                                       df.iloc[i]['MP_Dec']]], 0, ra_dec_order=True)[0])
            x_base = int(floor(x0)) - half_size  # position in image of PSF's (0,0) origin.
            y_base = int(floor(y0)) - half_size  # "
            x_psf_center = x0 - x_base  # offset from PSF's origin of Gaussian's center.
            y_psf_center = y0 - y_base  # offset from PSF's origin of Gaussian's center.
            gaussian = Gaussian2D(1, x_psf_center, y_psf_center, sigma, sigma)
            source_psf = gaussian(x, y)
            source_psf *= (flux / np.sum(source_psf))  # set total area to desired flux.
            source_psf_uint16 = np.round(source_psf).astype(np.uint16)
            hdu_list[i].data[y_base:y_base + edge_length,
                             x_base:x_base + edge_length] += source_psf_uint16  # np ndarray: [y,x].

    # Save updated image data to otherwise identical FITS files with new names:
    for i, filename in enumerate(df['Filename']):
        fn, ext = os.path.splitext(filename)
        fn_new = fn + '_Added' + ext  # e.g., 'MP_191-0001-Clear_Added.fts'
        hdu_list[i].writeto(os.path.join(TEST_FITS_DIR, fn_new))


DETRITUS________________________________________ = 0

# The following stuff is just no longer needed.


#
# def make_matching_kernels(df_images, ref_star_radec, target_sigma, nominal_size=80):
#     """ Make matching kernels to render subimages' ref_star Gaussian with target_sigma.
#         Use mid-images rather than subimages, to minimize risk of bumping into image boundaries.
#     :param df_images:
#     :param ref_star_radec:
#     :param target_sigma: in pixels. [float]
#     :param nominal_size: size of kernel, each edge. Could be incremented by 1 to make it odd. [int]
#     :return: df_images with new column 'MKernel' with needed matching kernels. [numpy ndarray]
#     """
#     kernels = [get_ref_star_kernel(im, ref_star_radec, nominal_size) for im in df_images['Mid_image']]
#     shape = kernels[0].shape
#     size = shape[0]
#     window = (SplitCosineBellWindow(alpha=0.5, beta=0.25))(shape)  # window fn of same shape as kernels.
#     kernels = [k * window for k in kernels]   # taper to zero all fluxes far from center.
#     kernels = [k / np.sum(k) for k in kernels]   # normalize.
#
#     center = int(size / 2)
#     y, x = np.mgrid[0:size, 0:size]
#     gaussian = Gaussian2D(1, center, center, target_sigma, target_sigma)
#     target_kernel = gaussian(x, y)
#     target_kernel /= np.sum(target_kernel)  # ensure normalized.
#     window = CosineBellWindow(alpha=0.35)
#     matching_kernels = [create_matching_kernel(k, target_kernel, window=window) for k in kernels]
#     df_images['MKernel'] = matching_kernels
#     return df_images
#
# (We're no longer planning to align full images.)
# def align_images(df_images, ref_star_radec):
#     """ Align (reposition images, update WCS) the full-sized images. OVERWRITE column 'Image'.
#     :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
#     :param ref_star_radec:
#     :return: dataframe with 'Image' column OVERWRITTEN by newly aligned images. [pandas DataFrame]
#     """
#     # Align all images to first image:
#     first_image_wcs = (df_images.iloc[0])['Image'].wcs
#     for i, im in zip(df_images.index, df_images['Image']):
#         print('    Aligning', i)
#         df_images.loc[i, 'Image'] = wcs_project(im, first_image_wcs)
#         # There's no benefit to saving these files, as they're not standard header.
#         # df_images.loc[i, 'Image'].data = df_images.loc[i, 'Image'].data.astype(np.int32)
#         # df_images.loc[i, 'Image'].write(os.path.join(TEST_FITS_DIR,
#         #                                            'Aligned_' + df_images.loc[i, 'Filename']))
#
#     # Print a summary (testing):
#     print("Post-alignment pix position of ref star (positions should be uniform):")
#     print("Ref Star RA,Dec:", str(ref_star_radec))
#     for i in df_images.index:
#         pix = df_images.loc[i, 'Image'].wcs.all_world2pix(ref_star_radec, 1,
#                                                           ra_dec_order=True)[0]
#         print('   ', df_images.loc[i, 'Filename'], str(pix))  # should be uniform, all images.
#     print('align_mid_images() done.')
#     return df_images


# df_averaged_image_sources = find_sources_in_averaged_subimage(averaged_subarray, df_images)
# df_sources = find_sources_in_all_subimages(df_images)
# df_sources = keep_only_matching_sources(df_sources, df_averaged_image_sources)
# df_sources = do_subimage_source_aperture_photometry(df_sources, df_images)
# df_qualified_sources = qualify_subimage_sources(df_sources)
# anchor_filename, df_anchor = select_anchor_subimage(df_qualified_sources, df_images)
# normalized_relative_fluxes = calc_normalized_relative_fluxes(df_qualified_sources, df_anchor, df_images)
# df_images = make_best_background_subimages(normalized_relative_fluxes, averaged_subarray, df_images)

# def get_ref_star_kernel(image, ref_star_radec, size):
#     """ Get a size x size array centered on the ref_star, background-subtracted, suitable as psf kernel.
#     :param image: the image (with wcs) in which to find the ref_star. [CCDData object]
#     :param ref_star_radec: RA,Dec of reference star in degrees. [2-list of floats]
#     :param size: number of pixels on (square) kernel's side; should be odd, will be made so if not. [int]
#     :return: small array centered on ref-star, suitable as psf kernel. [numpy array]
#     """
#     x, y = tuple(image.wcs.all_world2pix(ref_star_radec, 0, ra_dec_order=True)[0])
#     half_size = int(size / 2.0)
#     x_center, y_center = int(round(x)), int(round(y))
#     x_min = x_center - half_size
#     x_max = x_center + half_size + 1
#     y_min = y_center - half_size
#     y_max = y_center + half_size + 1
#     array = image.data[y_min:y_max, x_min:x_max].copy()
#     _, median, _ = sigma_clipped_stats(array, sigma=3.0)
#     array -= median
#     return array


# def find_sources_in_averaged_subimage(averaged_subimage, df_images):
#     """ Find sources in averaged subimage.
#     :param averaged_subimage: reference bkgd-subtracted, MP-masked subimage. [one astropy CCDData objects]
#     :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
#     :return: new dataframe of source info, one row per source. [pandas DataFrame]
#     """
#     expected_std = np.sqrt(sum([std**2 for std in df_images['Bkgd_std']]) / len(df_images)**2)
#     print('expected_std =', '{0:.3f}'.format(expected_std))
#     daofind = DAOStarFinder(fwhm=7.0, threshold=5.0 * expected_std, sky=0.0, exclude_border=True)
#     averaged_subimage_sources = daofind(data=averaged_subimage.data)  # returns an astropy Table object.
#     df_averaged_subimage_sources = averaged_subimage_sources.to_pandas() \
#         .sort_values(by='flux', ascending=False)[:N_AVERAGED_SUBIMAGE_SOURCES_TO_KEEP]
#     df_averaged_subimage_sources.index = range(len(df_averaged_subimage_sources))
#     df_averaged_subimage_sources = df_averaged_subimage_sources[['xcentroid', 'ycentroid']]
#     print('find_sources_in_averaged_subimage() done:', str(len(df_averaged_subimage_sources)),
#           'sources found in averaged subimage.')
#     return df_averaged_subimage_sources
#
#
# def find_sources_in_all_subimages(df_images):
#     """ Find sources in all subimages, without referring to averaged image at all.
#     :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
#     :return: new dataframe of source info, one row per source. [pandas DataFrame]
#     """
#     df_subimage_sources_list = []
#     for i, s_im in enumerate(df_images['Subimage_masked']):
#         daofind = DAOStarFinder(fwhm=7.0, threshold=5.0 * df_images.iloc[i]['Bkgd_std'], sky=0.0,
#                                 exclude_border=True)
#         df = daofind(data=s_im.data, mask=s_im.mask).to_pandas()
#         df['Filename'] = df_images.iloc[i]['Filename']
#         df_subimage_sources_list.append(df)
#     df_subimage_sources = pd.concat(df_subimage_sources_list)
#     df_subimage_sources.index = range(len(df_subimage_sources))
#     df_subimage_sources['SourceID'] = None
#     df_subimage_sources['FluxADU'] = None
#     df_subimage_sources = df_subimage_sources[['Filename', 'SourceID',
#                                                'xcentroid', 'ycentroid', 'FluxADU']]
#     print('find_sources_in_all_subimages() done:', str(len(df_subimage_sources)), 'sources found.')
#     return df_subimage_sources
#
#
# def keep_only_matching_sources(df_subimage_sources, df_averaged_subimage_sources):
#     """ Try to match sources found in all subimages to those found in averaged subimage,
#         keep only those sources that match, discard the rest.
#     :param df_subimage_sources: dataframe of source info from all subimages,
#     one row per source. [pandas DataFrame]
#     :param df_averaged_subimage_sources: source info, averaged subimage,
#                one row per source. [pandas DataFrame]
#     :return: df_subimage_sources, now with only matching sources retained. [pandas DataFrame]
#     """
#     for i_s in df_subimage_sources.index:
#         x_s, y_s = df_subimage_sources.loc[i_s, 'xcentroid'], df_subimage_sources.loc[i_s, 'ycentroid']
#         for i_as in df_averaged_subimage_sources.index:
#             x_as, y_as = df_averaged_subimage_sources.loc[i_as, 'xcentroid'], \
#                          df_averaged_subimage_sources.loc[i_as, 'ycentroid']
#             distance2 = (x_s - x_as)**2 + (y_s - y_as)**2
#             if distance2 <= SOURCE_MATCH_TOLERANCE**2:
#                 df_subimage_sources.loc[i_s, 'SourceID'] = i_as  # assign id from averaged-image sources.
#                 break
#     sources_to_keep = [s_id is not None for s_id in df_subimage_sources['SourceID']]
#     df_subimage_sources = df_subimage_sources.loc[sources_to_keep, :]
#     print('keep_only_matching_sources() done:', str(len(df_subimage_sources)), 'sources kept.')
#     return df_subimage_sources
#
#
# def do_subimage_source_aperture_photometry(df_subimage_sources, df_images):
#     """ Do aperture photometry on each kept source in each subimage, write results to dataframe.
#     :param df_subimage_sources: dataframe of source info from all subimages,
#     one row per source. [pandas DataFrame]
#     :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
#     :return: df_subimage_sources with added columns of aperture photometry results. [pandas DataFrame]
#     """
#     for i, fn in enumerate(df_images.index):
#         this_masked_subimage_data = df_images.loc[fn, 'Subimage_masked']
#         is_this_file = (df_subimage_sources['Filename'] == fn)
#         sources_index_list = df_subimage_sources.index[is_this_file]
#         x_positions = df_subimage_sources.loc[sources_index_list, 'xcentroid']
#         y_positions = df_subimage_sources.loc[sources_index_list, 'ycentroid']
#         positions = np.transpose((x_positions, y_positions))
#         apertures = CircularAperture(positions, r=R_APERTURE)
#         phot_table = aperture_photometry(this_masked_subimage_data, apertures)
#         df_phot = phot_table.to_pandas()
#         df_phot.index = sources_index_list
#         annulus_apertures = CircularAnnulus(positions, r_in=R_ANNULUS_IN, r_out=R_ANNULUS_OUT)
#         annulus_masks = annulus_apertures.to_mask(method='center')
#         bkgd_median_list = []
#         for mask in annulus_masks:
#             annulus_data = mask.multiply(this_masked_subimage_data)
#             annulus_data_1d = annulus_data[mask.data > 0]
#             _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
#             bkgd_median_list.append(median_sigclip)
#         bkgd_median = np.array(bkgd_median_list)
#         df_phot['annulus_median'] = bkgd_median
#         df_phot['aper_bkgd'] = bkgd_median * apertures.area
#         df_phot['final_phot'] = df_phot['aperture_sum'] - df_phot['aper_bkgd']
#         df_subimage_sources.loc[sources_index_list, 'FluxADU'] = df_phot['final_phot']
#     flux_is_ok = [flux is not None for flux in df_subimage_sources['FluxADU']]
#     df_subimage_sources = df_subimage_sources.loc[flux_is_ok, :]
#     print('do_subimage_source_aperture_photometry() done.')
#     return df_subimage_sources
#
#
# def qualify_subimage_sources(df_subimage_sources):
#     """ Keep only subimage sources having median flux at least 10% of highest median flux.
#     :param df_subimage_sources: source info from all subimages, one row per source. [pandas DataFrame]
#     :return: df_qualified_subimage_sources,
#     keeping only rows for sources with substantial flux. [pandas DataFrame]
#     """
#     source_ids = df_subimage_sources['SourceID'].drop_duplicates()
#     median_fluxes = []  # will be in same order as source_ids
#     for id in source_ids:
#         is_this_id = (df_subimage_sources['SourceID'] == id)
#         source_fluxes = df_subimage_sources.loc[is_this_id, 'FluxADU']
#         median_fluxes.append(source_fluxes.median())
#     max_median = max(median_fluxes)
#     median_flux_high_enough = [f >= 0.1 * max_median for f in median_fluxes]
#     qualified_source_ids = list(source_ids[median_flux_high_enough])
#     to_keep = [id in qualified_source_ids for id in df_subimage_sources['SourceID']]
#     df_qualified_subimage_sources = df_subimage_sources.loc[to_keep, :]
#     print('qualify_subimage_sources() done:', str(len(df_qualified_subimage_sources)), 'found.')
#     return df_qualified_subimage_sources
#
#
# def select_anchor_subimage(df_qualified_subimage_sources, df_images):
#     """ Find an image (or the first one if several) with a flux entry for every qualified source,
#         label this the "anchor image" for use in finding relative source strengths between images.
#     :param df_qualified_subimage_sources: data for qualified sources,
#     one row per source x image. [pandas DataFrame]
#     :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
#     :return: anchor_subimage_filename, df_anchor_subimage. [2-tuple of string, pandas DataFrame]
#     """
#     qualified_source_ids = df_qualified_subimage_sources['SourceID'].drop_duplicates()
#     anchor_subimage_filename = None
#     for fn in df_images.index:
#         if sum([f == fn for f in df_qualified_subimage_sources['Filename']]) == len(qualified_source_ids):
#             anchor_subimage_filename = fn
#             break
#     if anchor_subimage_filename is None:
#         print(' >>>>> ERROR: No image has a flux for every qualified source (unusual).')
#         return None
#     df_anchor_subimage = df_qualified_subimage_sources[df_qualified_subimage_sources['Filename'] ==
#                                                        anchor_subimage_filename].copy()
#     df_anchor_subimage.index = df_anchor_subimage['SourceID']
#     print('select_anchor_subimage() done.')
#     return anchor_subimage_filename, df_anchor_subimage
#
#
# def calc_normalized_relative_fluxes(df_qualified_subimage_sources, df_anchor_subimage, df_images):
#     """ Calculate normalized relative flux for each subimage, which will be used to scale averaged
#         image to subtract source images from subimages.
#     :param df_qualified_subimage_sources: qualified sources,
#     one row per source x image. [pandas DataFrame]
#     :param df_anchor_subimage: data for anchor subimage & sources. [pandas DataFrame]
#     :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
#     :return: dictionary of normalized relative fluxes for each subimage. [python dict]
#     """
#     qualified_source_ids = df_qualified_subimage_sources['SourceID'].drop_duplicates()
#     anchor_flux_dict = {id: df_anchor_subimage.loc[id, 'FluxADU']
#     for id in qualified_source_ids}  # lookup.
#
#     # Calculate normlized relative fluxes, for use in scaling background for each subimage:
#     relative_fluxes_dict = {fn: [] for fn in df_images.index}  # to be populated below.
#     for i in df_qualified_subimage_sources.index:
#         filename = df_qualified_subimage_sources.loc[i, 'Filename']
#         source_id = df_qualified_subimage_sources.loc[i, 'SourceID']
#         relative_fluxes_dict[filename].append(df_qualified_subimage_sources.loc[i, 'FluxADU'] /
#                                               anchor_flux_dict[source_id])
#     median_relative_fluxes_dict = {fn: np.median(relative_fluxes_dict[fn])
#                                    for fn in relative_fluxes_dict.keys()}
#     mean_median = np.mean([x for x in median_relative_fluxes_dict.values()])
#     normalized_relative_fluxes = {fn: x / mean_median for (fn, x) in median_relative_fluxes_dict.items()}
#     print('calc_normalized_relative_fluxes() done.')
#     return normalized_relative_fluxes
#
#
# def make_best_background_subimages(normalized_relative_fluxes, averaged_subimage, df_images):
#     """ Make best estimate of each subimage without MP (for best local background to MP).
#         Uses sum of flat background plus averaged subimage scaled by relative source flux.
#     :param normalized_relative_fluxes: normalized relative fluxes for each subimage. [python dict]
#     :param averaged_subimage: the averaged, MP-masked subimage. [astropy CCDData object]
#     :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
#     :return: df_images with new column 'Subimage_bkgd' [pandas DataFrame]
#     """
#     df_images['Subimage_bkgd'] = [df_images.loc[fn, 'Bkgd_median'] +
#                                   normalized_relative_fluxes[fn] * averaged_subimage
#                                   for fn in df_images['Filename']]
#     plot_best_background_subimages(df_images)
#     print('make_best_background_subimages() done.')
#     return df_images
#
#
# def plot_best_background_subimages(df_images):
#     image_indices = [0, 1, 2, len(df_images) - 1]
#     fig, axes = plt.subplots(ncols=3, nrows=len(image_indices))
#     for i_row, image_index in enumerate(image_indices):
#         image_index = image_indices[i_row]
#         subimage = df_images.iloc[image_index]['Subimage']
#         bkgd_subimage = df_images.iloc[image_index]['Bkgd_image']
#         ax = axes[i_row, 0]
#         ax.set_title(df_images.iloc[image_index]['Filename'])
#         ax.imshow(do_gamma(subimage), origin='upper', interpolation='none', cmap='Greys')
#         ax = axes[i_row, 1]
#         ax.set_title('Scaled bkgd')
#         ax.imshow(do_gamma(bkgd_subimage), origin='upper', interpolation='none', cmap='Greys')
#         ax = axes[i_row, 2]
#         ax.set_title('Bkdg-subtr')
#         ax.imshow(do_gamma(np.clip(subimage - bkgd_subimage, a_min=0, a_max=None)),
#                   origin='upper', interpolation='none', cmap='Greys')
#     fig.tight_layout()
#     plt.show()


# def hard_align_on_ref_star_centroids(df_images, ref_star_radec):
#     """ Get rough ref star (x,y), refine via local centroid, convolve to make tiny shifts.
#     :param df_images:
#     :param ref_star_radec:
#     :return:
#     """
#     # Calculate and print (very small) pre-existing ref star offset for each image:
#     print('Start of hard_align_on_ref_star_centroids():')
#     x0_image0, y0_image0 = (df_images.iloc[0])['Subimage'].wcs.all_world2pix(ref_star_radec,
#                                                                              0, ra_dec_order=True)[0]
#     x0_image0, y0_image0 = int(x0_image0), int(y0_image0)
#     ref_star_offsets = []
#     centroid_cell_half_size = 20
#     for i, im in enumerate(df_images['Subimage']):
#         x0, y0 = im.wcs.all_world2pix(ref_star_radec, 0, ra_dec_order=True)[0]
#         tiny_array = im[y0_image0-centroid_cell_half_size: y0_image0+centroid_cell_half_size,
#                         x0_image0-centroid_cell_half_size: x0_image0+centroid_cell_half_size].data
#         x_refstar, y_refstar = centroid_com(tiny_array)
#         ref_star_offsets.append((x_refstar - centroid_cell_half_size,
#                                  y_refstar - centroid_cell_half_size))
#         print('   ', df_images.iloc[i]['Filename'],
#               '   ref star at world2pix:', '{:.3f}'.format(x0), '{:.3f}'.format(y0),
#               '   at centroid:', '{:.3f}'.format(x_refstar + (x0_image0-centroid_cell_half_size)),
#               '{:.3f}'.format(y_refstar + (y0_image0-centroid_cell_half_size)))
#
#     # Convolve each image to make an array in which its ref star offset has been reversed:
#     half_size = 10
#     edge_pixels = 2 * half_size + 1
#     y, x = np.mgrid[0:edge_pixels, 0:edge_pixels]
#     print('After hard_align_on_ref_star_centroids():')
#     for i, im in enumerate(df_images['Subimage']):
#         x_offset, y_offset = ref_star_offsets[i]
#         gaussian = Gaussian2D(1, half_size - x_offset, half_size - y_offset, 0.7, 0.7)
#         kernel = gaussian(x, y)
#         kernel /= np.sum(kernel)
#         hard_aligned_array = convolve(im.data, kernel)
#         im.data = hard_aligned_array
#         x0, y0 = im.wcs.all_world2pix(ref_star_radec, 0, ra_dec_order=True)[0]
#         tiny_array = im[y0_image0-centroid_cell_half_size: y0_image0+centroid_cell_half_size,
#                         x0_image0-centroid_cell_half_size: x0_image0+centroid_cell_half_size].data
#         x_refstar, y_refstar = centroid_com(tiny_array)
#         print('   ', df_images.iloc[i]['Filename'],
#               '   ref star at world2pix:', '{:.3f}'.format(x0), '{:.3f}'.format(y0),
#               '   at centroid:', '{:.3f}'.format(x_refstar + (x0_image0-centroid_cell_half_size)),
#               '{:.3f}'.format(y_refstar + (y0_image0-centroid_cell_half_size)))
#     return df_images
