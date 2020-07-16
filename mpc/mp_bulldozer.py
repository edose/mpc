__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
from math import log10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import astropy.io.fits as fits
# import astropy.wcs as wcs
from astropy.stats import sigma_clipped_stats
from astropy.nddata import CCDData
from ccdproc import ImageFileCollection, wcs_project, trim_image, Combiner
from photutils import make_source_mask, DAOStarFinder, CircularAperture, aperture_photometry,\
    CircularAnnulus

from photrix.image import Image, FITS
from photrix.util import MixedModelFit

TEST_FITS_DIR = 'C:/Astro/MP Photometry/MP_1111/AN20200617'
BB_LARGE_MARGIN = 400
BB_SMALLER_MARGIN = 50  # margin around bounding box, in pixels.
R_MP_MASK = 12      # radius in pixels
R_APERTURE = 9      # "
R_ANNULUS_IN = 15   # "
R_ANNULUS_OUT = 20  # "
N_AVG_SOURCES_TO_KEEP = 5

MATCH_TOLERANCE = 2  # distance in pixels


def try_bulldozer():
    # Get test images as CCDData objects into :
    ic = ImageFileCollection(location=TEST_FITS_DIR, keywords=['filter', 'date-obs'],
                             glob_include='MP_*.f*')  # mishandles CCDData units.
    ic.sort(['date-obs'])
    filenames = ic.files_filtered(include_path=False)
    images = [CCDData.read(os.path.join(TEST_FITS_DIR, f), unit='adu') for f in filenames]
    for i in images:
        i.meta['RADESYSa'] = i.meta.pop('RADECSYS')  # Replace obsolete key.
    filters = [im.meta['Filter'] for im in images]
    # date_obs = [im.meta['date-obs'] for im in images]
    exposures = [im.meta['exposure'] for im in images]
    jds = [im.meta['jd'] for im in images]
    jd_mids = [jd + exp / 2 / 24 / 3600 for (jd, exp) in zip(jds, exposures)]
    df_images = pd.DataFrame({'Filename': filenames, 'Image': images, 'Filter': filters,
                              'Exposure': exposures, 'JD_mid': jd_mids})
    df_images.index = filenames
    df_images = df_images.loc[df_images['Filter'] == 'Clear', :]

    # # NB: regular dictionaries are ordered in python 3.7+
    # image_dict = {f: i for (f, i) in zip(filenames, images) if i.meta['FILTER'] == 'Clear'}
    # for fn in image_dict.keys():
    #     image_dict[fn].meta['RADESYSa'] = image_dict[fn].meta.pop('RADECSYS')  # Replace obsolete key.

    # radec1 = image_dict['MP_191-0001-Clear.fts'].wcs.all_pix2world([[200, 300]], 1)
    # for f, i in image_dict.items():
    #     radec = i.wcs.all_pix2world([[200, 300]], 1)
    #     print(f, str((3600 / 0.682) * (radec - radec1)))

    # Call bulldozer():
    # best_mp_fluxes = bulldozer(df_images)

    whatever = bulldozer(df_images)
    return whatever


def bulldozer(df_images):
    mp_file_early = 'MP_191-0001-Clear.fts'
    mp_file_late = 'MP_191-0028-Clear.fts'
    mp_pix_early = (826.4, 1077.4)
    mp_pix_late = (1144.3, 1099.3)

    df_images = bd_add_mp_radec(df_images, mp_file_early, mp_file_late, mp_pix_early, mp_pix_late)

    df_images = bd_trim_to_larger_bb(df_images)

    df_images = bd_align_images_in_larger_bb(df_images)

    df_images = bd_trim_to_smaller_bb(df_images)

    df_images = bd_add_mp_pix_positions(df_images)

    plot_first_last(df_images)

    df_images = bd_add_background_data(df_images)

    sources_images = make_sources_images(df_images)

    averaged_image = make_averaged_image(df_images)

    df_avg_sources = find_avg_sources(averaged_image, df_images)

    df_sources = find_sources(sources_images, df_images)

    df_sources = keep_only_matching_sources(df_sources, df_avg_sources)

    df_sources = do_source_aperture_photometry(df_sources, sources_images, df_images)

    df_qualified_sources = qualify_sources(df_sources)

    anchor_filename, df_anchor = select_anchor_image(df_qualified_sources, df_images)

    normalized_relative_fluxes = calc_normalized_relative_fluxes(df_qualified_sources, df_anchor)

    df_images = make_best_background_images(normalized_relative_fluxes, averaged_image, df_images)





    # Construct flux per pixel in every MP mask, subtract it from MP flux. Return that and raw flux too.

    return


BULLDOZER_SUBFUNCTIONS_______________________________ = 0


def bd_add_mp_radec(df_images, mp_file_early, mp_file_late, mp_pix_early, mp_pix_late):
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
    df_images['MP_RA'] = None
    df_images['MP_Dec'] = None
    for i in df_images.index:
        jd = df_images.loc[i, 'JD_mid']
        df_images.loc[i, 'MP_RA'] = ra_early + (jd - jd_early) * ra_rate
        df_images.loc[i, 'MP_Dec'] = dec_early + (jd - jd_early) * dec_rate
    return df_images


def bd_trim_to_larger_bb(df_images):
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
                                                  df_images['MP_dec'])]
    print('initial images: first and last mp_pix:', str(mp_pix_list[0]), str(mp_pix_list[-1]))
    print()

    # Do first trim to large bounding box (throw away most pixels as they are unneeded):
    bb_x_min = min([mp_pix[0] for mp_pix in mp_pix_list]) - BB_LARGE_MARGIN
    bb_x_max = max([mp_pix[0] for mp_pix in mp_pix_list]) + BB_LARGE_MARGIN
    bb_y_min = min([mp_pix[1] for mp_pix in mp_pix_list]) - BB_LARGE_MARGIN
    bb_y_max = max([mp_pix[1] for mp_pix in mp_pix_list]) + BB_LARGE_MARGIN

    for fn, im in zip(df_images['Filename'], df_images['Image']):
        # Remember, reverse the FITS/pixel axes when addressing np arrays directly:
        df_images.loc[fn, 'Image'] = trim_image(im[bb_y_min:bb_y_max, bb_x_min:bb_x_max])
    print((df_images.iloc[0])['Image'].wcs.printwcs())

    return df_images


def bd_align_images_in_larger_bb(df_images):
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
        print(df_images.loc[i, 'Filename'], str(mp_pix_ref))  # should be uniform across all images.

    return df_images


def bd_trim_to_smaller_bb(df_images):
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
    print(df_images.iloc[0])['Image'].wcs.printwcs()

    return df_images


def bd_add_mp_pix_positions(df_images):
    """ Calculate MP (x,y) pixel position for each image, save in dataframe as new columns.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe with new MP_x and 'Image' larger-bounding-box images OVERWRITTEN by smaller-bb-trimmed images.
             New images have updated WCS. [pandas DataFrame]
    """
    mp_pix_list = [im.wcs.all_world2pix([[mp_ra, mp_dec]], 0, ra_dec_order=True)[0]  # origin ZERO, arrays.
                   for (im, mp_ra, mp_dec) in zip(df_images['Image'],
                                                  df_images['MP_RA'],
                                                  df_images['MP_dec'])]
    df_images['MP_x'] = [x for (x, y) in mp_pix_list]
    df_images['MP_y'] = [y for (x, y) in mp_pix_list]

    for fn, pix in zip(df_images['Filename'], mp_pix_list):
        print(fn, str(pix))
        # fullpath = os.path.join(TEST_FITS_DIR, 'Lil_' + fn)
        # df_images.loc[fn, 'Image'].write(fullpath)

    return df_images


def do_gamma(image, gamma=0.0625, dark_clip=0.002):
    im_min, im_max = np.min(image), np.max(image)
    # im_min += dark_clip * (im_max - im_min)
    im_scaled = np.clip((image - im_min) / (im_max - im_min), dark_clip, 1.0)
    return np.power(im_scaled, gamma)


def plot_first_last(df_images):
    """ For testing. """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title(df_images.iloc[0]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[0]['Image']), origin='upper',
               interpolation='none', cmap='Greys')
    ax2.set_title(df_images.iloc[-1]['Filename'])
    ax2.imshow(do_gamma(df_images.iloc[-1]['Image']), origin='upper',
               interpolation='none', cmap='Greys')
    plt.tight_layout()
    plt.show()


def bd_add_background_data(df_images):
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
    return df_images


def make_sources_images(df_images):
    """ Make background-subtracted, MP-masked images to be used in our actual photometry.
        Return list of images (in same order as df_images). Does not alter df_images at all.
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: list of bkgd-subtracted, MP-masked images. [list of astropy CCDData objects]
    """
    sources_images = []
    for i, im in enumerate(df_images['Image']):
        sources_image = im.copy()
        sources_image.data = sources_image.data - df_images.iloc[i]['Bkgd_median']
        radec_mp = [df_images.iloc[i]['MP_ra'], df_images.iloc[i]['MP_dec']]
        # Origin is zero here, to address numpy array cells rather than FITS pixels.
        x_mp, y_mp = tuple(sources_image.wcs.all_world2pix([radec_mp], 0, ra_dec_order=True)[0])
        # Arrays are addressed y-first:
        sources_image.mask = np.fromfunction(lambda i, j: (j - x_mp)**2 + (i - y_mp)**2 <= R_MP_MASK**2,
                                             shape=sources_image.data.shape, dtype=int)
        sources_images.append(sources_image)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Subimage ' + df_images.iloc[0]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[0]['Image']), origin='upper',
               interpolation='none', cmap='Greys')
    ax2.set_title('MP Mask')
    ax2.imshow(sources_images[0].mask, origin='upper')
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Subimage ' + df_images.iloc[-1]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[-1]['Image']), origin='upper',
               interpolation='none', cmap='Greys')
    ax2.set_title('MP Mask')
    ax2.imshow(sources_images[-1].mask, origin='upper')
    plt.tight_layout()
    plt.show()

    return sources_images


def make_averaged_image(sources_images, df_images):
    """ Make and return reference (background-subtracted, MP-masked) average image.
        This is what the sky would look like, on average, if the MP were not there.
    :param sources_images: list of bkgd-subtracted, MP-masked images. [list of astropy CCDData objects]
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: one small-bounding-box averaged, MP-masked image. [one astropy CCDData object]
    """
    combiner = Combiner(sources_images)
    averaged_image = combiner.average_combine()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Subimage ' + df_images.iloc[-1]['Filename'])
    ax1.imshow(do_gamma(df_images.iloc[-1]['Image']), origin='upper', interpolation='none', cmap='Greys')
    ax2.set_title('Avg image')
    ax2.imshow(do_gamma(averaged_image), origin='upper', interpolation='none', cmap='Greys')
    plt.tight_layout()
    plt.show()

    return averaged_image


def find_avg_sources(averaged_image, df_images):
    """ Find sources in averaged image.
    :param averaged_image: reference bkgd-subtracted, MP-masked image. [one astropy CCDData objects]
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe of source info, one row per source. [pandas DataFrame]
    """
    expected_std = np.sqrt(sum([std**2 for std in df_images['Bkgd_std']]) / len(df_images)**2)
    print('expected_std =', '{0:.3f}'.format(expected_std))
    daofind = DAOStarFinder(fwhm=7.0, threshold=5.0 * expected_std, sky=0.0, exclude_border=True)
    avg_sources = daofind(data=averaged_image.data)  # returns an astropy Table object.
    df_avg_sources = avg_sources.to_pandas().sort_values(by='flux', ascending=False)[:N_AVG_SOURCES_TO_KEEP]
    df_avg_sources.index = range(len(df_avg_sources))
    df_avg_sources = df_avg_sources[['xcentroid', 'ycentroid']]
    return df_avg_sources


def find_sources(sources_images, df_images):
    """ Find sources in all subimages, without referring to averaged image at all.
    :param sources_images: list of bkgd-subtracted, MP-masked images. [list of astropy CCDData objects]
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: dataframe of source info, one row per source. [pandas DataFrame]
    """
    df_sources_list = []
    for i, s_im in enumerate(sources_images):
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
    return df_sources


def keep_only_matching_sources(df_sources, df_avg_sources):
    """ Try to match sources found in all subimages to those found in averaged image,
        keep only those matching, discard the rest.
    :param df_sources: dataframe of source info from all subimages, one row per source. [pandas DataFrame]
    :param df_avg_sources: dataframe of source info, averaged image one row per source. [pandas DataFrame]
    :return: df_sources with only matching sources retained. [pandas DataFrame]
    """
    for i_s in df_sources.index:
        x_s, y_s = df_sources.loc[i_s, 'xcentroid'], df_sources.loc[i_s, 'ycentroid']
        for i_as in df_avg_sources.index:
            x_as, y_as = df_avg_sources.loc[i_as, 'xcentroid'], df_avg_sources.loc[i_as, 'ycentroid']
            distance2 = (x_s - x_as)**2 + (y_s - y_as)**2
            if distance2 <= MATCH_TOLERANCE**2:
                df_sources.loc[i_s, 'SourceID'] = i_as  # assign id from avg sources.
                break
    sources_to_keep = [s_id is not None for s_id in df_sources['SourceID']]
    df_sources = df_sources.loc[sources_to_keep, :]
    return df_sources


def do_source_aperture_photometry(df_sources, sources_images, df_images):
    """ Do aperture photometry on each kept source in each subimage, write results to dataframe.
    :param df_sources: dataframe of source info from all subimages, one row per source. [pandas DataFrame]
    :param sources_images: list of bkgd-subtracted, MP-masked images. [list of astropy CCDData objects]
    :param df_images: dataframe, one row per session image in photometric filter. [pandas DataFrame]
    :return: df_sources with added columns of aperture photometry results. [pandas DataFrame]
    """
    for i, fn in enumerate(df_images.index):
        is_this_file = (df_sources['Filename'] == fn)
        sources_index_list = df_sources.index[is_this_file]
        x_positions = df_sources.loc[sources_index_list, 'xcentroid']
        y_positions = df_sources.loc[sources_index_list, 'ycentroid']
        positions = np.transpose((x_positions, y_positions))
        apertures = CircularAperture(positions, r=R_APERTURE)
        phot_table = aperture_photometry(sources_images[i].data, apertures)
        df_phot = phot_table.to_pandas()
        df_phot.index = sources_index_list
        annulus_apertures = CircularAnnulus(positions, r_in=R_ANNULUS_IN, r_out=R_ANNULUS_OUT)
        annulus_masks = annulus_apertures.to_mask(method='center')
        bkgd_median_list = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(sources_images[i].data)
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
        relative_fluxes_dict[filename].append(df_qualified_sources.loc[i, 'FluxADU'] / anchor_flux_dict[source_id])
    median_relative_fluxes_dict = {fn: np.median(relative_fluxes_dict[fn])
                                   for fn in relative_fluxes_dict.keys()}
    mean_median = np.mean([x for x in median_relative_fluxes_dict.values()])
    normalized_relative_fluxes = {fn: x / mean_median for (fn, x) in median_relative_fluxes_dict.items()}
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

    image_indices = [0, 1, 2, len(df_images) - 1]
    fig, axes = plt.subplots(ncols=3, nrows=4)

    for i_row in range(4):
        image_index = image_indices[i_row]
        ax = axes[i_row, 0]
        image = df_images.iloc[image_index]['Image']
        bkgd_image = df_images.iloc[image_index]['Bkgd_image']
        ax.set_title(df_images.iloc[image_index]['Filename'])
        ax.imshow(do_gamma(image), origin='upper', interpolation='none', cmap='Greys')
        ax = axes[i_row, 1]
        ax.imshow(do_gamma(bkgd_image), origin='upper', interpolation='none', cmap='Greys')
        ax = axes[i_row, 2]
        ax.imshow(do_gamma(np.clip(image - bkgd_image, a_min=0, a_max=None)),
                  origin='upper', interpolation='none', cmap='Greys')
    plt.tight_layout()
    plt.show()

    return df_images


def do_mp_aperture_photometry():
    pass



DETRITUS________________________________________ = 0

