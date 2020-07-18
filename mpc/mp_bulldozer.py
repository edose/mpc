__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# ##############################################################################
# mp_bulldozer.py // July-August 2020
# Eric Dose, New Mexico Mira Project  [ <-- NMMP to be renamed late 2020 ]
# Albuquerque, New Mexico, (what's left of the) USA
#
# To whomever is unlucky enough to stumble across this file in this repo 'mpc':
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

# import astropy.io.fits as fits
# import astropy.wcs as wcs
from astropy.stats import sigma_clipped_stats
from astropy.nddata import CCDData
from astropy.modeling.models import Gaussian2D
from astropy.visualization import LogStretch, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from ccdproc import ImageFileCollection, wcs_project, trim_image, Combiner
from photutils import make_source_mask, DAOStarFinder, CircularAperture, aperture_photometry,\
    CircularAnnulus, data_properties, create_matching_kernel, \
    TukeyWindow, TopHatWindow, SplitCosineBellWindow, CosineBellWindow

from .mp_phot import get_fits_filenames

FWHM_PER_SIGMA = 2 * sqrt(2 * log(2))  # ca. 2.355

TEST_FITS_DIR = 'C:/Astro/MP Photometry/MP_1111/AN20200617'
PIXEL_SHIFT_TOLERANCE = 200  # maximum image shift in pixels we expect from bad pointing, tracking, etc.
BB_LARGE_PADDING = 100       # min distance MP-image edge, for large bounding box images.
BB_LARGE_MARGIN = PIXEL_SHIFT_TOLERANCE + BB_LARGE_PADDING
BB_SMALLER_MARGIN = 80  # margin around bounding box, in pixels.
R_MP_MASK = 12      # radius in pixels
R_APERTURE = 9      # "
R_ANNULUS_IN = 15   # "
R_ANNULUS_OUT = 20  # "
N_AVERAGED_IMAGE_SOURCES_TO_KEEP = 5

SOURCE_MATCH_TOLERANCE = 2  # distance in pixels


def try_bulldozer():
    """ Dummy calling fn. """
    directory_path = TEST_FITS_DIR
    filter='Clear'
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

    df_images = make_df_images(directory_path, filter)

    ref_star_radec = calc_ref_star_radec(df_images, ref_star_file, ref_star_pix)

    df_images = add_mp_radecs(df_images, mp_file_early, mp_file_late, mp_pix_early, mp_pix_late)

    df_images = trim_to_larger_bb(df_images)

    df_images = align_images_in_larger_bb(df_images)

    print_ref_star_morphology(df_images, ref_star_radec)

    plot_ref_star_kernels(df_images, ref_star_radec, ref_star_pix)
    return df_images  # ################################################################

    df_images = trim_to_smaller_bb(df_images)

    df_images = add_mp_pixel_positions(df_images)

    # plot_first_last(df_images)

    df_images = add_background_data(df_images)

    df_images = add_mp_masked_images(df_images)

    # plot_mp_masks(df_images)

    averaged_image = make_averaged_image(df_images)

    df_averaged_image_sources = find_averaged_image_sources(averaged_image, df_images)

    df_sources = find_sources(df_images)

    df_sources = keep_only_matching_sources(df_sources, df_averaged_image_sources)

    df_sources = do_source_aperture_photometry(df_sources, df_images)

    df_qualified_sources = qualify_sources(df_sources)

    anchor_filename, df_anchor = select_anchor_image(df_qualified_sources, df_images)

    normalized_relative_fluxes = calc_normalized_relative_fluxes(df_qualified_sources, df_anchor, df_images)

    df_images = make_best_background_images(normalized_relative_fluxes, averaged_image, df_images)





    # Construct flux per pixel in every MP mask, subtract it from MP flux. Return that and raw flux too.


    return df_images


BULLDOZER_SUBFUNCTIONS_______________________________ = 0


def make_df_images(directory_path, filter):
    """ Make starting (small) version of key dataframe.
    :param directory_path: where the FITS files are comprising this MP photometry session, where
           "session" by convention = all images from one night, targeting one minor planet (MP). [string]
    :param filter: name of the filter through which the FITS images were taken. [string]
    :return: starting version of df_images, one row per FITS file.
             To be used for this session of MP photometry. This is the central data table,
             and it will be updated throughout the photometry workflow to follow. [pandas DataTable]
    """
    # Get all FITS filenames in directory, and read them into list of CCDData objects:
    filenames = get_fits_filenames(directory_path)
    images = [CCDData.read(os.path.join(TEST_FITS_DIR, f), unit='adu') for f in filenames]

    # Keep only filenames and images in chosen filter:
    keep_image = [(im.meta['Filter'] == filter) for im in images]
    filenames = [f for (f, ki) in zip(filenames, keep_image) if ki is True]
    images = [im for (im, ki) in zip(images, keep_image) if ki is True]

    # Replace obsolete header key often found in FITS files derived from MaxIm DL.
    for i in images:
        i.meta['RADESYSa'] = i.meta.pop('RADECSYS')  # both ops in one statement. cool.

    # Gather initial data from CCDData objects, to go into df_images:
    filters = [im.meta['Filter'] for im in images]
    exposures = [im.meta['exposure'] for im in images]
    jds = [im.meta['jd'] for im in images]
    jd_mids = [jd + exp / 2 / 24 / 3600 for (jd, exp) in zip(jds, exposures)]
    df_images = pd.DataFrame(data={'Filename': filenames, 'Image': images, 'Filter': filters,
                                   'Exposure': exposures, 'JD_mid': jd_mids},
                             index=filenames)
    df_images = df_images.sort_values(by='JD_mid')
    print('make_df_images() done:', len(df_images), 'images.')
    return df_images


def calc_ref_star_radec(df_images, ref_star_file, ref_star_pix):
    """ Returns RA,Dec sky position of reference star (for later use, but must be done on full image). """
    ref_star_radec = df_images.loc[ref_star_file, 'Image'].wcs.all_pix2world([list(ref_star_pix)], 1)
    print('ref_star_radec =', str(ref_star_radec))
    return ref_star_radec


def add_mp_radecs(df_images, mp_file_early, mp_file_late, mp_pix_early, mp_pix_late):
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


def trim_to_larger_bb(df_images):
    """ Trim images to a large bounding box (mostly to save effort).
        Images not yet aligned, so bounding box is made large enough to account for image-to-image shifts.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe with 'Image' full-size images OVERWRITTEN by larger-bb-trimmed images.
             New images have updated WCS but are not yet aligned. [pandas DataFrame]
    """
    # Make list of MP (RA, Dec) for each image:
    mp_pix_list = [im.wcs.all_world2pix([[mp_ra, mp_dec]], 1, ra_dec_order=True)[0]
                   for (im, mp_ra, mp_dec) in zip(df_images['Image'],
                                                  df_images['MP_RA'],
                                                  df_images['MP_Dec'])]
    print('initial images: first and last mp_pix:', str(mp_pix_list[0]), str(mp_pix_list[-1]))
    print()

    # Do first trim to large bounding box (throw away most pixels as they are unneeded):
    bb_x_min = int(floor(min([mp_pix[0] for mp_pix in mp_pix_list]) - BB_LARGE_MARGIN))
    bb_x_max = int(ceil(max([mp_pix[0] for mp_pix in mp_pix_list]) + BB_LARGE_MARGIN))
    bb_y_min = int(floor(min([mp_pix[1] for mp_pix in mp_pix_list]) - BB_LARGE_MARGIN))
    bb_y_max = int(ceil(max([mp_pix[1] for mp_pix in mp_pix_list]) + BB_LARGE_MARGIN))

    for i, im in zip(df_images.index, df_images['Image']):
        # Remember, reverse the FITS/pixel axes when addressing np arrays directly:
        df_images.loc[i, 'Image'] = trim_image(im[bb_y_min:bb_y_max, bb_x_min:bb_x_max])
    # df_images.iloc[0]['Image'].wcs.printwcs()
    print('trim_to_larger_bb() done.')
    return df_images


def align_images_in_larger_bb(df_images):
    """ Align (reposition images, update WCS) the image data within larger bounding box.
        OVERWRITE old image data.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe with 'Image' images OVERWRITTEN by newly aligned images. [pandas DataFrame]
    """
    # Align all images to first image:
    first_image_wcs = (df_images.iloc[0])['Image'].wcs
    for i, im in zip(df_images.index, df_images['Image']):
        df_images.loc[i, 'Image'] = wcs_project(im, first_image_wcs)

    print("Post-alignment pix position of first image's MP RaDec (should be uniform):")
    mp_ra_ref, mp_dec_ref = (df_images.iloc[0])['MP_RA'], (df_images.iloc[0])['MP_Dec']
    for i in df_images.index:
        mp_pix_ref = df_images.loc[i, 'Image'].wcs.all_world2pix([[mp_ra_ref, mp_dec_ref]], 1,
                                                                 ra_dec_order=True)[0]
        print('   ', df_images.loc[i, 'Filename'], str(mp_pix_ref))  # should be uniform across all images.
    print('align_images_in_larger_bb() done.')
    return df_images


def trim_to_smaller_bb(df_images):
    """ Trim images to the final, smaller bounding box to be used for all photometry.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe with 'Image' larger-bounding-box images OVERWRITTEN by smaller-bb-trimmed images.
             New images have updated WCS. [pandas DataFrame]
    """
    # Find bounding box based on first and last images, regardless of direction of MP motion:
    image_first = (df_images.iloc[0])['Image']  # Images are aligned, so first and last image will suffice.
    image_last = (df_images.iloc[-1])['Image']  # "
    mp_ra_first, mp_dec_first = (df_images.iloc[0])['MP_RA'], (df_images.iloc[0])['MP_Dec']
    mp_ra_last, mp_dec_last = (df_images.iloc[-1])['MP_RA'], (df_images.iloc[-1])['MP_Dec']
    mp_pix_first = image_first.wcs.all_world2pix([[mp_ra_first, mp_dec_first]], 1, ra_dec_order=True)[0]
    mp_pix_last = image_last.wcs.all_world2pix([[mp_ra_last, mp_dec_last]], 1, ra_dec_order=True)[0]
    bb_x_min = int(round(min(mp_pix_first[0], mp_pix_last[0]) - BB_SMALLER_MARGIN))
    bb_x_max = int(round(max(mp_pix_first[0], mp_pix_last[0]) + BB_SMALLER_MARGIN))
    bb_y_min = int(round(min(mp_pix_first[1], mp_pix_last[1]) - BB_SMALLER_MARGIN))
    bb_y_max = int(round(max(mp_pix_first[1], mp_pix_last[1]) + BB_SMALLER_MARGIN))
    print('x: ', str(bb_x_min), str(bb_x_max), '      y: ', str(bb_y_min), str(bb_y_max))

    # Perform the trims, OVERWRITE old 'Image' entries:
    for i, im in zip(df_images.index, df_images['Image']):
        # Remember, reverse the two axes when using np arrays directly:
        df_images.loc[i, 'Image'] = trim_image(im[bb_y_min:bb_y_max, bb_x_min:bb_x_max])
    # print((df_images.iloc[0])['Image'].wcs.printwcs())
    print('trim_to_smaller_bb() done.')
    return df_images


def add_mp_pixel_positions(df_images):
    """ Calculate MP (x,y) pixel position for each image, save in dataframe as new columns.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe with new MP_x and 'Image' larger-bounding-box images OVERWRITTEN by smaller-bb-trimmed images.
             New images have updated WCS. [pandas DataFrame]
    """
    mp_pix_list = [im.wcs.all_world2pix([[mp_ra, mp_dec]], 0, ra_dec_order=True)[0]  # origin ZERO, arrays.
                   for (im, mp_ra, mp_dec) in zip(df_images['Image'],
                                                  df_images['MP_RA'],
                                                  df_images['MP_Dec'])]
    df_images['MP_x'] = [x for (x, y) in mp_pix_list]
    df_images['MP_y'] = [y for (x, y) in mp_pix_list]

    print('MP pixel positions in aligned small-bb images:')
    for fn, pix in zip(df_images['Filename'], mp_pix_list):
        print('   ', fn, str(pix))
        # fullpath = os.path.join(TEST_FITS_DIR, 'Lil_' + fn)
        # df_images.loc[fn, 'Image'].write(fullpath)
    print('add_mp_pixel_positions() done.')
    return df_images


def add_background_data(df_images):
    """ For each image, calculate sigma-clipped mean, median background ADU levels & std. deviation;
        write to dataframe as new columns. Probably will use the sigma-clipped median as background ADUs.
        This all assumes constant background across this small subimage. Go to 2-D later if really needed.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe with new 'Bkgd_mean', 'Bkgd_median', and 'Bkgd_std' columns. [pandas DataFrame]
    """
    bkgd_masks = [make_source_mask(im.data, nsigma=2, npixels=5, filter_fwhm=2, dilate_size=11)
                  for im in df_images['Image']]
    stats = [sigma_clipped_stats(im.data, sigma=3.0, mask=b_mask)
             for (im, b_mask) in zip(df_images['Image'], bkgd_masks)]
    df_images['Bkgd_mean'], df_images['Bkgd_median'], df_images['Bkgd_std'] = tuple(zip(*stats))
    print('add_background_data() done.')
    return df_images


def add_mp_masked_images(df_images):
    """ Make background-subtracted, MP-masked images to be used in our actual photometry.
        Return df_images updated with new column; does not alter 'Image' column.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: same dataframe with new 'MP_masked_image' column. [pandas DataFrame]
    """
    mp_masked_images = []
    for i, im in enumerate(df_images['Image']):
        mp_masked_image = im.copy()
        mp_masked_image.data = mp_masked_image.data - df_images.iloc[i]['Bkgd_median']
        radec_mp = [df_images.iloc[i]['MP_RA'], df_images.iloc[i]['MP_Dec']]

        # Origin is zero here, to address numpy array cells rather than FITS pixels.
        x_mp, y_mp = tuple(mp_masked_image.wcs.all_world2pix([radec_mp], 0, ra_dec_order=True)[0])
        # Arrays are addressed y-first:
        mp_masked_image.mask = np.fromfunction(lambda i, j: (j - x_mp)**2 + (i - y_mp)**2 <= R_MP_MASK**2,
                                               shape=mp_masked_image.data.shape, dtype=int)
        mp_masked_images.append(mp_masked_image)
    df_images['MP_masked_image'] = mp_masked_images
    print('add_mp_masked_images() done.')
    return df_images


def make_averaged_image(df_images):
    """ Make and return reference (background-subtracted, MP-masked) average image.
        This is what the sky would look like, on average, if the MP were not there.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: one small-bounding-box averaged, MP-masked image. [one astropy CCDData object]
    """
    combiner = Combiner(df_images['MP_masked_image'])
    averaged_image = combiner.average_combine()
    plot_averaged_image(df_images, averaged_image)
    print('make_averaged_image() done.')
    return averaged_image


def find_averaged_image_sources(averaged_image, df_images):
    """ Find sources in averaged image.
    :param averaged_image: reference bkgd-subtracted, MP-masked image. [one astropy CCDData objects]
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe of source info, one row per source. [pandas DataFrame]
    """
    expected_std = np.sqrt(sum([std**2 for std in df_images['Bkgd_std']]) / len(df_images)**2)
    print('expected_std =', '{0:.3f}'.format(expected_std))
    daofind = DAOStarFinder(fwhm=7.0, threshold=5.0 * expected_std, sky=0.0, exclude_border=True)
    averaged_image_sources = daofind(data=averaged_image.data)  # returns an astropy Table object.
    df_averaged_image_sources = averaged_image_sources.to_pandas() \
        .sort_values(by='flux', ascending=False)[:N_AVERAGED_IMAGE_SOURCES_TO_KEEP]
    df_averaged_image_sources.index = range(len(df_averaged_image_sources))
    df_averaged_image_sources = df_averaged_image_sources[['xcentroid', 'ycentroid']]
    print('fine_averaged_image_sources() done:', str(len(df_averaged_image_sources)),
          'averaged-image sources found.')
    return df_averaged_image_sources


def find_sources(df_images):
    """ Find sources in all subimages, without referring to averaged image at all.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe of source info, one row per source. [pandas DataFrame]
    """
    df_sources_list = []
    for i, s_im in enumerate(df_images['MP_masked_image']):
        daofind = DAOStarFinder(fwhm=7.0, threshold=5.0 * df_images.iloc[i]['Bkgd_std'], sky=0.0,
                                exclude_border=True)
        df = daofind(data=s_im.data, mask=s_im.mask).to_pandas()
        df['Filename'] = df_images.iloc[i]['Filename']
        df_sources_list.append(df)
    df_sources = pd.concat(df_sources_list)
    df_sources.index = range(len(df_sources))
    df_sources['SourceID'] = None
    df_sources['FluxADU'] = None
    df_sources = df_sources[['Filename', 'SourceID', 'xcentroid', 'ycentroid', 'FluxADU']]
    print('find_sources() done', str(len(df_sources)), 'sources found.')
    return df_sources


def keep_only_matching_sources(df_sources, df_averaged_image_sources):
    """ Try to match sources found in all subimages to those found in averaged image,
        keep only those matching, discard the rest.
    :param df_sources: dataframe of source info from all subimages, one row per source. [pandas DataFrame]
    :param df_averaged_image_sources: source info, averaged image, one row per source. [pandas DataFrame]
    :return: df_sources with only matching sources retained. [pandas DataFrame]
    """
    for i_s in df_sources.index:
        x_s, y_s = df_sources.loc[i_s, 'xcentroid'], df_sources.loc[i_s, 'ycentroid']
        for i_as in df_averaged_image_sources.index:
            x_as, y_as = df_averaged_image_sources.loc[i_as, 'xcentroid'], \
                         df_averaged_image_sources.loc[i_as, 'ycentroid']
            distance2 = (x_s - x_as)**2 + (y_s - y_as)**2
            if distance2 <= SOURCE_MATCH_TOLERANCE**2:
                df_sources.loc[i_s, 'SourceID'] = i_as  # assign id from averaged-image sources.
                break
    sources_to_keep = [s_id is not None for s_id in df_sources['SourceID']]
    df_sources = df_sources.loc[sources_to_keep, :]
    print('keep_only_matching_sources() done:', str(len(df_sources)), 'sources kept.')
    return df_sources


def do_source_aperture_photometry(df_sources, df_images):
    """ Do aperture photometry on each kept source in each subimage, write results to dataframe.
    :param df_sources: dataframe of source info from all subimages, one row per source. [pandas DataFrame]
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: df_sources with added columns of aperture photometry results. [pandas DataFrame]
    """
    for i, fn in enumerate(df_images.index):
        this_masked_image_data = df_images.loc[fn, 'MP_masked_image']
        is_this_file = (df_sources['Filename'] == fn)
        sources_index_list = df_sources.index[is_this_file]
        x_positions = df_sources.loc[sources_index_list, 'xcentroid']
        y_positions = df_sources.loc[sources_index_list, 'ycentroid']
        positions = np.transpose((x_positions, y_positions))
        apertures = CircularAperture(positions, r=R_APERTURE)
        phot_table = aperture_photometry(this_masked_image_data, apertures)
        df_phot = phot_table.to_pandas()
        df_phot.index = sources_index_list
        annulus_apertures = CircularAnnulus(positions, r_in=R_ANNULUS_IN, r_out=R_ANNULUS_OUT)
        annulus_masks = annulus_apertures.to_mask(method='center')
        bkgd_median_list = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(this_masked_image_data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkgd_median_list.append(median_sigclip)
        bkgd_median = np.array(bkgd_median_list)
        df_phot['annulus_median'] = bkgd_median
        df_phot['aper_bkgd'] = bkgd_median * apertures.area
        df_phot['final_phot'] = df_phot['aperture_sum'] - df_phot['aper_bkgd']
        df_sources.loc[sources_index_list, 'FluxADU'] = df_phot['final_phot']
    flux_is_ok = [flux is not None for flux in df_sources['FluxADU']]  # could add warning if != len.
    df_sources = df_sources.loc[flux_is_ok, :]
    print('do_source_aperture_photometry() done.')
    return df_sources


def qualify_sources(df_sources):
    """ Keep only sources having median flux at least 10% of highest median flux.
    :param df_sources: dataframe of source info from all subimages, one row per source. [pandas DataFrame]
    :return: df_qualified_sources, keeping only rows for sources with substantial flux. [pandas DataFrame]
    """
    source_ids = df_sources['SourceID'].drop_duplicates()
    median_fluxes = []  # will be in same order as source_ids
    for id in source_ids:
        is_this_id = (df_sources['SourceID'] == id)
        source_fluxes = df_sources.loc[is_this_id, 'FluxADU']
        median_fluxes.append(source_fluxes.median())
    max_median = max(median_fluxes)
    median_flux_high_enough = [f >= 0.1 * max_median for f in median_fluxes]
    qualified_source_ids = list(source_ids[median_flux_high_enough])
    to_keep = [id in qualified_source_ids for id in df_sources['SourceID']]
    df_qualified_sources = df_sources.loc[to_keep, :]
    print('qualify_sources() done:', str(len(df_qualified_sources)), 'found.')
    return df_qualified_sources


def select_anchor_image(df_qualified_sources, df_images):
    """ Find an image (or the first one if several) with a flux entry for every qualified source,
        label this the "anchor image" for use in finding relative source strengths between images.
    :param df_qualified_sources: data for qualified sources, one row per source x image. [pandas DataFrame]
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: anchor_filename, df_anchor. [2-tuple of string, pandas DataFrame]
    """
    qualified_source_ids = df_qualified_sources['SourceID'].drop_duplicates()
    anchor_filename = None
    for fn in df_images.index:
        if sum([f == fn for f in df_qualified_sources['Filename']]) == len(qualified_source_ids):
            anchor_filename = fn
            break
    if anchor_filename is None:
        print(' >>>>> ERROR: No image has a flux for every qualified source (unusual).')
        return None
    df_anchor = df_qualified_sources[df_qualified_sources['Filename'] == anchor_filename].copy()
    df_anchor.index = df_anchor['SourceID']
    print('select_anchor_image() done.')
    return anchor_filename, df_anchor


def calc_normalized_relative_fluxes(df_qualified_sources, df_anchor, df_images):
    """ Calculate normalized relative flux for each subimage, which will be used to scale averaged
        image to subtract source images from subimages.
    :param df_qualified_sources: data for qualified sources, one row per source x image. [pandas DataFrame]
    :param df_anchor: data for anchor subimage & sources. [pandas DataFrame]
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dictionary of normalized relative fluxes for each subimage. [python dict]
    """
    qualified_source_ids = df_qualified_sources['SourceID'].drop_duplicates()
    anchor_flux_dict = {id: df_anchor.loc[id, 'FluxADU'] for id in qualified_source_ids}  # lookup table.

    # Calculate normlized relative fluxes, for use in scaling background for each image:
    relative_fluxes_dict = {fn: [] for fn in df_images.index}  # to be populated below.
    for i in df_qualified_sources.index:
        filename = df_qualified_sources.loc[i, 'Filename']
        source_id = df_qualified_sources.loc[i, 'SourceID']
        relative_fluxes_dict[filename].append(df_qualified_sources.loc[i, 'FluxADU'] /
                                              anchor_flux_dict[source_id])
    median_relative_fluxes_dict = {fn: np.median(relative_fluxes_dict[fn])
                                   for fn in relative_fluxes_dict.keys()}
    mean_median = np.mean([x for x in median_relative_fluxes_dict.values()])
    normalized_relative_fluxes = {fn: x / mean_median for (fn, x) in median_relative_fluxes_dict.items()}
    print('calc_normalized_relative_fluxes() done.')
    return normalized_relative_fluxes


def make_best_background_images(normalized_relative_fluxes, averaged_image, df_images):
    """ Make best estimate of each image without MP (best background to MP).
        Uses sum of flat background plus averaged image scaled by relative source flux.
    :param normalized_relative_fluxes: normalized relative fluxes for each subimage. [python dict]
    :param averaged_image: small-bounding-box averaged, MP-masked image. [astropy CCDData object]
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: df_images with new column 'Bkdg_image' [pandas DataFrame]
    """
    df_images['Bkgd_image'] = [df_images.loc[fn, 'Bkgd_median'] +
                               normalized_relative_fluxes[fn] * averaged_image
                               for fn in df_images['Filename']]
    plot_best_background_images(df_images)
    print('make_best_background_images() done.')
    return df_images


def do_mp_aperture_photometry():
    pass



PLOTTING_ETC_FUNCTIONS__________________________ = 0


def do_gamma(image, gamma=0.0625, dark_clip=0.002):
    im_min, im_max = np.min(image), np.max(image)
    # im_min += dark_clip * (im_max - im_min)
    im_scaled = np.clip((image - im_min) / (im_max - im_min), dark_clip, 1.0)
    return np.power(im_scaled, gamma)


def plot_first_last(df_images):
    """ For testing. """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title(df_images.iloc[0]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[0]['Image']), origin='upper', interpolation='none', cmap='Greys')
    ax2.set_title(df_images.iloc[-1]['Filename'])
    ax2.imshow(do_gamma(df_images.iloc[-1]['Image']), origin='upper', interpolation='none', cmap='Greys')
    plt.tight_layout()
    plt.show()


def print_ref_star_morphology(df_images, ref_star_radec):
    ims = df_images['Image']
    ref_star_xy0 = tuple(df_images.iloc[0]['Image'].wcs.all_world2pix(ref_star_radec, 0,
                                                                      ra_dec_order=True)[0])
    print('ref_star_xy0 =', str(ref_star_xy0))
    x, y = ref_star_xy0
    x_min = int(floor(x - 25))
    x_max = int(ceil(x + 25))
    y_min = int(floor(y - 25))
    y_max = int(ceil(y + 25))
    plot_data = []
    for i, im in enumerate(df_images['Image']):
        data = im.data[y_min:y_max, x_min:x_max].copy()  # x & y reversed for direct mp.array access
        _, median, _ = sigma_clipped_stats(data, sigma=3.0)
        data -= median  # subtract background
        cat = data_properties(data)
        columns = ['id', 'xcentroid', 'ycentroid', 'semimajor_axis_sigma',
                   'semiminor_axis_sigma', 'orientation']
        tbl = cat.to_table(columns=columns)
        tbl['xcentroid'].info.format = '.3f'
        tbl['ycentroid'].info.format = '.3f'
        tbl['semimajor_axis_sigma'].info.format = '.3f'
        tbl['semiminor_axis_sigma'].info.format = '.3f'
        tbl['orientation'].info.format = '.3f'
        print('\n#####', df_images.iloc[i]['Filename'])
        print(tbl)
        plot_data.append(data)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title(df_images.iloc[3]['Filename'])
    ax1.imshow(plot_data[3], origin='upper', interpolation='none', cmap='Greys')
    ax2.set_title(df_images.iloc[14]['Filename'])
    ax2.imshow(plot_data[14], origin='upper', interpolation='none', cmap='Greys')
    plt.tight_layout()
    plt.show()


def plot_ref_star_kernels(df_images, ref_star_radec, ref_star_pix):
    """ For each large_bb subimage, plot original kernel and psf_matching_kernel. """
    nominal_size = 80
    target_sigma = 5.176  # largest semi-major axis

    kernels = [get_ref_star_kernel(im, ref_star_radec, nominal_size) for im in df_images['Image']]
    shape = kernels[0].shape
    size = shape[0]
    window = (SplitCosineBellWindow(alpha=0.5, beta=0.25))(shape)  # window fn of same shape as kernels.
    # window = (TukeyWindow(alpha=0.4))(shape)  # window function of same shape as kernels.
    kernels = [k * window for k in kernels]   # taper to zero all fluxes far from center.
    kernels = [k / np.sum(k) for k in kernels]   # normalize.

    center = int(size / 2)
    y, x = np.mgrid[0:size, 0:size]
    gaussian = Gaussian2D(1, center, center, target_sigma, target_sigma)
    target_kernel = gaussian(x, y)
    target_kernel /= np.sum(target_kernel)  # ensure normalized.

    # window = TukeyWindow(alpha=0.4)
    # window = TopHatWindow(0.35)
    window = CosineBellWindow(alpha=0.35)
    matching_kernels = [create_matching_kernel(k, target_kernel, window=window) for k in kernels]
    for (fn, mk) in zip(df_images['Filename'], matching_kernels):
        print(fn, ' min, max =', '{0:.6f}'.format(np.min(mk)), '{0:.6f}'.format(np.max(mk)))

    # Plot example window functions:
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # alpha = 0.1
    # ax1.set_title('CosineBellWindow' + str(alpha))
    # ax1.imshow((CosineBellWindow(alpha=alpha))(shape),
    #            origin='upper', interpolation='none', cmap='viridis')
    # alpha = 0.4
    # ax2.set_title('CosineBellWindow' + str(alpha))
    # ax2.imshow((CosineBellWindow(alpha=alpha))(shape),
    #            origin='upper', interpolation='none', cmap='viridis')
    # plt.tight_layout()
    # plt.show()

    n_rows = len(kernels)
    rows_per_fig = 4

    norm = ImageNormalize(stretch=LogStretch())
    rows_this_fig = 0
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
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Subimage ' + df_images.iloc[0]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[0]['Image']), origin='upper', interpolation='none', cmap='Greys')
    ax2.set_title('MP Mask')
    ax2.imshow(df_images.iloc[0]['MP_masked_image'].mask, origin='upper')
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Subimage ' + df_images.iloc[-1]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[-1]['Image']), origin='upper', interpolation='none', cmap='Greys')
    ax2.set_title('MP Mask')
    ax2.imshow(df_images.iloc[-1]['MP_masked_image'].mask, origin='upper')
    plt.tight_layout()
    plt.show()


def plot_averaged_image(df_images, averaged_image):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Subimage ' + df_images.iloc[-1]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[-1]['Image']), origin='upper', interpolation='none', cmap='Greys')
    ax2.set_title('Avg image')
    ax2.imshow(do_gamma(averaged_image), origin='upper', interpolation='none', cmap='Greys')
    plt.tight_layout()
    plt.show()


def plot_best_background_images(df_images):
    image_indices = [0, 1, 2, len(df_images) - 1]
    fig, axes = plt.subplots(ncols=3, nrows=len(image_indices))
    for i_row, image_index in enumerate(image_indices):
        image_index = image_indices[i_row]
        image = df_images.iloc[image_index]['Image']
        bkgd_image = df_images.iloc[image_index]['Bkgd_image']
        ax = axes[i_row, 0]
        ax.set_title(df_images.iloc[image_index]['Filename'])
        ax.imshow(do_gamma(image), origin='upper', interpolation='none', cmap='Greys')
        ax = axes[i_row, 1]
        ax.set_title('Scaled bkgd')
        ax.imshow(do_gamma(bkgd_image), origin='upper', interpolation='none', cmap='Greys')
        ax = axes[i_row, 2]
        ax.set_title('Bkdg-subtr')
        ax.imshow(do_gamma(np.clip(image - bkgd_image, a_min=0, a_max=None)),
                  origin='upper', interpolation='none', cmap='Greys')
    fig.tight_layout()
    plt.show()


SUPPORT_FUNCTIONS_______________________________ = 0


def background_subtract_array(array):
    """ Use source-masked, sigma-clipped statistics to subtract away background to zero.
    :param array: array to background-subtracted. [numpy array]
    :return: copy of array, background-subtracted. [numpy array]
    """
    copy = array.copy()
    bkgd_mask = make_source_mask(copy, nsigma=2, npixels=5, filter_fwhm=2, dilate_size=11)
    _, median, _ = sigma_clipped_stats(copy, sigma=3.0, mask=bkgd_mask)
    return copy - median


def get_ref_star_kernel(image, ref_star_radec, size):
    """ Get a size x size array centered on the ref_star, background-subtracted, suitable as psf kernel.
    :param image: the image (with wcs) in which to find the ref_star. [CCDData object]
    :param ref_star_radec: RA,Dec of reference star in degrees. [2-list of floats]
    :param size: number of pixels on (square) kernel's side; should be odd, will be made so if not. [int]
    :return: small array centered on ref-star, suitable as psf kernel. [numpy array]
    """
    x, y = tuple(image.wcs.all_world2pix(ref_star_radec, 0, ra_dec_order=True)[0])
    half_size = int(size / 2.0)
    x_center, y_center = int(round(x)), int(round(y))
    x_min = x_center - half_size
    x_max = x_center + half_size + 1
    y_min = y_center - half_size
    y_max = y_center + half_size + 1
    array = image.data[y_min:y_max, x_min:x_max].copy()
    _, median, _ = sigma_clipped_stats(array, sigma=3.0)
    array -= median
    return array


DETRITUS________________________________________ = 0

