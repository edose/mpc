__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
# import sys
# from io import StringIO
from datetime import datetime, timezone
from math import floor, cos, pi

# External packages:
# import numpy as np
import pandas as pd
# import requests
# import statsmodels.api as sm

# From external (EVD) package photrix:
from photrix.image import Image, FITS
# from photrix.fov import FOV_DIRECTORY, Fov

# APASS_10_URL = 'https://www.aavso.org/cgi-bin/apass_dr10_download.pl'
ATLAS_REFCAT2_DIRECTORY = 'J:/Astro/Catalogs/ATLAS-refcat2/mag-0-16/'
RP1_LIMIT = 9  # arcseconds; closeness limit for flux = 0.1 * star flux
R1_LIMIT = 14   # arcseconds; closeness limit for flux = 1 * star flux
R10_LIMIT = 20  # arcseconds; closeness limit for flux = 10 * star flux
ATLAS_REFCAT2_EPOCH_UTC = (datetime(2015, 1, 1) +
                           (datetime(2016, 1, 1) - datetime(2015, 1, 1)) / 2.0)\
    .replace(tzinfo=timezone.utc)  # refcat2 proper-motion epoch is 2015.5.

DAYS_PER_YEAR_NOMINAL = 365.25
DEGREES_PER_RADIAN = 180.0 / pi

# # APASS10... To screen comps immediately on reading them in (wide limits; will narrow selections later):
# MP_COMP_RADEC_ERROR_MAX = 1.0  # in arcseconds
# R_ESTIMATE_MIN = 9  # wide, as this will be refined later.
# R_ESTIMATE_MAX = 16  # "
# ERR_R_ESTIMATE_MAX = 0.25  # fairly high (inclusive), as there will be more screens, later.
# COLOR_BV_MIN = 0.2  # inclusive (will be refined using V-I later).
# COLOR_BV_MAX = 1.2  # "

# # APASS10...For estimating Best R mag from catalog B and V mags:
# R_ESTIMATE_V_COEFF = 0.975  # from regression on Best_R_mag
# R_ESTIMATE_BV_COLOR_COEFF = -0.419  # "
# R_ESTIMATE_INTERCEPT = 0  # "

MATCH_TOLERANCE_ARCSEC = 4  # to match stars across catalogs


# APASS_10_URL_FUNCTIONS______________________________________________ = 0
#
#
# def get_apass10_comps(ra, dec, radius, r_min=None, r_max=None,
#                       mp_color_only=True, add_rmag_estimate=True):
#     """  Renders a dataframe with all needed comp info, including estimated R mags.
#     Tests OK ~ 20191106.
#     :param ra: center Right Ascension for comp search, in degrees only [float].
#     :param dec: center Declination for comp search, in degrees only [float].
#     :param radius: radius of circular search area, in degrees [float].
#     :param r_min: minimum R magnitude (brightest limit) [float]. None = no limit.
#     :param r_max: maximum R magnitude (faintest limit) [float]. None = no limit.
#     :param mp_color_only: True means keep only comps close to typical MP color [boolean].
#     :param add_rmag_estimate: True means add R_estimate column and its error [boolean].
#     :return: dataframe of comp data [pandas Dataframe].
#      Returned columns = ID, degRA, e_RA, degDec, e_Dec, Vmag, e_Vmag, Vnobs, Bmag, e_Bmag, Bnobs,
#               BminusV, e_BminusV, R_estimate, e_R_estimate; index=ID column [string].
#     """
#     result = requests.post(url=APASS_10_URL,
#                            data={'ra': ra, 'dec': dec, 'radius': radius, 'outtype': 1})
#     df = pd.read_csv(StringIO(result.text), sep=',')
#     df = df.rename(columns={'radeg': 'degRA', 'raerr(")': 'e_RA',
#                             'decdeg': 'degDec', 'decerr(")': 'e_Dec',
#                             'Johnson_V (V)': 'Vmag', 'Verr': 'e_Vmag',
#                             'Johnson_B (B)': 'Bmag', 'Berr': 'e_Bmag',
#                             'Sloan_g (SG)': 'SGmag', 'SGerr': 'e_SGmag',
#                             'Sloan_r (SR)': 'SRmag', 'SRerr': 'e_SRmag',
#                             'Sloan_i (SI)': 'SImag', 'SIerr': 'e_SImag'})
#     df.index = [str(i) for i in df.index]
#     df['ID'] = df.index
#     columns_to_keep = ['ID', 'degRA', 'e_RA', 'degDec', 'e_Dec',
#                        'Vmag', 'e_Vmag', 'Vnobs',
#                        'Bmag', 'e_Bmag', 'Bnobs',
#                        'SGmag', 'e_SGmag', 'SGnobs',
#                        'SRmag', 'e_SRmag', 'SRnobs',
#                        'SImag', 'e_SImag', 'SInobs']
#     df = df[columns_to_keep]
#     df = df[df['e_RA'] < MP_COMP_RADEC_ERROR_MAX]
#     df = df[df['e_Dec'] < MP_COMP_RADEC_ERROR_MAX]
#     df = df[~pd.isnull(df['Vmag'])]
#     df = df[~pd.isnull(df['e_Vmag'])]
#     df = df[~pd.isnull(df['Bmag'])]
#     df = df[~pd.isnull(df['e_Bmag'])]
#     df = df[~pd.isnull(df['SRmag'])]
#     df = df[~pd.isnull(df['e_SRmag'])]
#     df['BminusV'] = df['Bmag'] - df['Vmag']
#     df['e_BminusV'] = np.sqrt(df['e_Vmag'] ** 2 + df['e_Bmag'] ** 2)
#     if add_rmag_estimate:
#         df['R_estimate'] = R_ESTIMATE_V_COEFF * df['Vmag'] +\
#                            R_ESTIMATE_BV_COLOR_COEFF * df['BminusV'] +\
#                            R_ESTIMATE_INTERCEPT
#         df['e_R_estimate'] = np.sqrt(R_ESTIMATE_V_COEFF**2 * df['e_Vmag']**2 +
#                                      R_ESTIMATE_BV_COLOR_COEFF**2 * df['e_BminusV']**2)
#         if r_min is not None:
#             df = df[df['R_estimate'] >= r_min]
#         if r_max is not None:
#             df = df[df['R_estimate'] <= r_max]
#         df = df[df['e_R_estimate'] <= ERR_R_ESTIMATE_MAX]
#     df_comps = df.copy()
#     if mp_color_only is True:
#         above_min = COLOR_BV_MIN <= df_comps['BminusV']
#         below_max = df_comps['BminusV'] <= COLOR_BV_MAX
#         color_ok = list(above_min & below_max)
#         df_comps = df_comps[color_ok]
#     return df_comps


ATLAS_REFCAT2_____________________________________________________ = 0


class Refcat2:
    """ Container for ATLAS refcat2 star data within a RA,Dec rectangle."""
    def __init__(self, ra_deg_range=(None, None), dec_deg_range=(None, None),
                 sort_ra=True, directory=ATLAS_REFCAT2_DIRECTORY):
        """ Basic constructor. Other constructors call this one.
        :param ra_deg_range: (min RA, max RA) tuple. Handles zero-crossing gracefully [2-tuple of floats].
        :param dec_deg_range: (min Dec, max Dec) tuple [2-tuple of floats].
        :param sort_ra: True iff star data are to be sorted in RA order [boolean].
        :param directory: where ATLAS refcat2 is found [string].
        Columns in core dataframe: RA_deg, Dec_deg, PM_ra, dPM_ra, PM_dec, dPM_dec,
               G_gaia, dG_gaia, BP_gaia, dBP_gaia, RP_gaia, dRP_gaia,
               T_eff, dupvar, RP1, R1, R10, g, dg, r, dr, i, di; index=unique integers.
        """
        ra_spec_first = int(floor(ra_deg_range[0])) % 360  # will always be in [0, 360).
        ra_spec_last = int(floor(ra_deg_range[1])) % 360   # "
        if ra_spec_last < ra_spec_first:
            ra_spec_last += 360
        dec_spec_first = int(floor(dec_deg_range[0]))
        dec_spec_first = max(dec_spec_first, -90)
        dec_spec_last = int(floor(dec_deg_range[1]))
        dec_spec_last = min(dec_spec_last, 89)
        # print('RA: ', str(ra_spec_first), str((ra_spec_last % 360) + 1))
        # print('Dec:', str(dec_spec_first), str(dec_spec_last))
        df_list = []
        for ra_spec in range(ra_spec_first, ra_spec_last + 1):
            for dec_spec in range(dec_spec_first, dec_spec_last + 1):
                df_degsq = read_one_refcat2_sqdeg(directory, ra_spec % 360, dec_spec)
                # print('From:', str(ra_spec % 360), str(dec_spec), ' -> ', str(len(df_degsq)), 'rows.')
                df_list.append(df_degsq)
        df = pd.DataFrame(pd.concat(df_list, ignore_index=True))  # new index of unique integers
        print('\nRefcat2: begin with', str(len(df)), 'stars.')

        # Trim dataframe based on user's actual limits on RA and Dec:
        ra_too_low = (df['RA_deg'] < ra_deg_range[0]) & (df['RA_deg'] >= ra_spec_first)
        ra_too_high = (df['RA_deg'] > ra_deg_range[1]) & (df['RA_deg'] <= (ra_spec_last % 360) + 1)
        dec_too_low = df['Dec_deg'] < dec_deg_range[0]
        dec_too_high = df['Dec_deg'] > dec_deg_range[1]
        # print(str(sum(ra_too_low)), str(sum(ra_too_high)), str(sum(dec_too_low)), str(sum(dec_too_high)))
        radec_outside_requested = ra_too_low | ra_too_high | dec_too_low | dec_too_high
        df = df[~radec_outside_requested]
        print('Refcat2: RADec-trimmed to', str(len(df)), 'stars.')

        # Add columns for synthetic B-V color & synthetic APASS (~Sloan) R magnitude:
        df.loc[:, 'BminusV'] = [0.830 * g - 0.803 * r for (g, r) in zip(df['g'], df['r'])]
        df.loc[:, 'APASS_R'] = [0.950 * r + 0.05 * i for (r, i) in zip(df['r'], df['i'])]

        if sort_ra is True:
            self.df_raw = df.copy().sort_values(by='RA_deg')  # in case all are needed (unlikely)
        self.df_selected = self.df_raw.copy()  # the working copy.
        self.epoch = ATLAS_REFCAT2_EPOCH_UTC

    @classmethod
    def from_fits_object(cls, fits_object):
        ra_deg_min, ra_deg_max, dec_deg_min, dec_deg_max = get_bounding_ra_dec(fits_object)
        return cls((ra_deg_min, ra_deg_max), (dec_deg_min, dec_deg_max), sort_ra=True)

    @classmethod
    def from_fits_file(cls, fits_directory, fits_filename):
        fits_object = FITS(fits_directory, '', fits_filename)
        return cls.from_fits_object(fits_object)

    def n_stars(self):
        return len(self.df_selected)

    def selected_columns(self, column_list):
        return self.df_selected.loc[:, column_list].copy()

    def select_min_r_mag(self, min_r_mag):
        rows_to_keep = [(r >= min_r_mag) for r in self.df_selected['r']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_max_r_mag(self, max_r_mag):
        rows_to_keep = [(r <= max_r_mag) for r in self.df_selected['r']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_max_g_uncert(self, max_dg):
        rows_to_keep = [(dg <= max_dg) for dg in self.df_selected['dg']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_max_r_uncert(self, max_dr):
        rows_to_keep = [(dr <= max_dr) for dr in self.df_selected['dr']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_max_i_uncert(self, max_di):
        rows_to_keep = [(di <= max_di) for di in self.df_selected['di']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_bv_color(self, min_bv=None, max_bv=None):
        above_min = [True if min_bv is None else (bv >= min_bv) for bv in self.df_selected['BminusV']]
        below_max = [True if max_bv is None else (bv <= max_bv) for bv in self.df_selected['BminusV']]
        rows_to_keep = [a and b for (a, b) in zip(above_min, below_max)]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def select_dgaia(self):
        rows_to_keep = [(dgaia > 0) for dgaia in self.df_selected['dG_gaia']]
        self.df_selected = self.df_selected.loc[rows_to_keep, :]

    def remove_overlapping(self):
        rp1_too_close = pd.Series([False if pd.isnull(rp1) else (rp1 < RP1_LIMIT)
                                   for rp1 in self.df_selected['RP1']])
        r1_too_close = pd.Series([False if pd.isnull(r1) else (r1 < R1_LIMIT)
                                  for r1 in self.df_selected['R1']])
        r10_too_close = pd.Series([False if pd.isnull(r10) else (r10 < R10_LIMIT)
                                   for r10 in self.df_selected['R10']])
        is_overlapping = rp1_too_close | r1_too_close | r10_too_close
        self.df_selected = self.df_selected.loc[list(~is_overlapping), :].copy()

    def update_epoch(self, new_datetime_utc):
        d_years = (new_datetime_utc - ATLAS_REFCAT2_EPOCH_UTC).total_seconds() /\
                  (DAYS_PER_YEAR_NOMINAL * 24 * 3600)
        ra_date = [(ra_epoch + d_years * pm_ra / 3600.0) % 360
                   for (ra_epoch, pm_ra) in zip(self.df_selected['RA_deg'], self.df_selected['PM_ra'])]
        dec_date = [dec_epoch + d_years * pm_dec / 3600.0
                    for (dec_epoch, pm_dec) in zip(self.df_selected['Dec_deg'], self.df_selected['PM_dec'])]
        df_comps_new_date = self.df_selected.copy()
        df_comps_new_date.loc[:, 'RA_deg'] = ra_date
        df_comps_new_date.loc[:, 'Dec_deg'] = dec_date
        self.df_selected = df_comps_new_date
        self.epoch = new_datetime_utc


def read_one_refcat2_sqdeg(directory=ATLAS_REFCAT2_DIRECTORY, ra_deg_min=None, dec_deg_min=None):
    ra_deg_int = int(ra_deg_min)
    dec_deg_int = int(dec_deg_min)
    filename = '{:03d}'.format(ra_deg_int) + '{:+03d}'.format(dec_deg_int) + '.rc2'
    fullpath = os.path.join(directory, filename)
    df = pd.read_csv(fullpath, sep=',', engine='python', header=None,
                     skip_blank_lines=True, error_bad_lines=False,
                     usecols=[0, 1, 4, 5, 6, 7,
                              8, 9, 10, 11, 12, 13,
                              14, 16, 18, 19, 20,
                              21, 22, 25, 26, 29, 30, 33, 34], prefix='col')
    df.columns = ['RA_deg', 'Dec_deg', 'PM_ra', 'dPM_ra', 'PM_dec', 'dPM_dec',
                  'G_gaia', 'dG_gaia', 'BP_gaia', 'dBP_gaia', 'RP_gaia', 'dRP_gaia',
                  'T_eff', 'dupvar', 'RP1', 'R1', 'R10',
                  'g', 'dg', 'r', 'dr', 'i', 'di', 'z', 'dz']
    df['RA_deg'] *= 0.00000001
    df['Dec_deg'] *= 0.00000001
    df['PM_ra'] *= 0.00001    # proper motion in arcsec/year
    df['dPM_ra'] *= 0.00001   # uncert in PM, arcsec/year
    df['PM_dec'] *= 0.00001   # proper motion in arcsec/year
    df['dPM_dec'] *= 0.00001  # uncert in PM, arcsec/year
    df['G_gaia'] *= 0.001  # in magnitudes; dG_gaia remains in millimagnitudes
    df['BP_gaia'] *= 0.001  # in magnitudes; dBP_gaia remains in millimagnitudes
    df['RP_gaia'] *= 0.001  # in magnitudes; dRP_gaia remains in millimagnitudes
    df['RP1'] = [None if rp1 == 999 else rp1 / 10.0 for rp1 in df['RP1']]  # radius in arcseconds
    df['R1'] = [None if r1 == 999 else r1 / 10.0 for r1 in df['R1']]       # "
    df['R10'] = [None if r10 == 999 else r10 / 10.0 for r10 in df['R10']]  # "
    df['g'] *= 0.001  # in magnitudes; dg remains in millimagnitudes
    df['r'] *= 0.001  # in magnitudes; dr remains in millimagnitudes
    df['i'] *= 0.001  # in magnitudes; di remains in millimagnitudes
    df['z'] *= 0.001  # in magnitudes; dz remains in millimagnitudes
    id_prefix = '{:03d}'.format(ra_deg_int) + '{:+03d}'.format(dec_deg_int) + '_'
    id_list = [id_prefix + '{:0>6d}'.format(i + 1) for i in range(len(df))]  # unique in entire catalog.
    df.insert(0, 'CatalogID', id_list)
    print('Refcat2 sqdeg [' + str(ra_deg_int) + ', ' + str(dec_deg_int) + ']: ' + str(len(df)) + ' stars.')
    return df


def find_matching_comp(df_comps, ra_deg, dec_deg):
    """ Find ATLAS refcat2 (as df_comps) stars matching input ra_deg, dec_deg; closest if >1 matching.
    :return: Index of matching star in df_comps.
    """
    tol_deg = MATCH_TOLERANCE_ARCSEC / 3600.0
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
        # Here, we choose the *closest* df_comps comp and return its index:
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


def get_bounding_ra_dec(fits_object):
    image = Image(fits_object)
    ps = fits_object.plate_solution  # a pandas Series
    ra_list, dec_list = [], []
    for xfract in [-0.5, 0.5]:
        dx = xfract * image.xsize
        for yfract in [-0.5, 0.5]:
            dy = yfract * image.ysize
            d_east_west = 1.03 * (dx * ps['CD1_1'] + dy * ps['CD1_2'])  # in degrees
            d_ra = d_east_west / cos(ps['CRVAL2'] / DEGREES_PER_RADIAN)      # "
            d_dec = 1.03 * (dx * ps['CD2_1'] + dy * ps['CD2_2'])        # "
            ra_list.append(ps['CRVAL1'] + d_ra)
            dec_list.append(ps['CRVAL2'] + d_dec)
    ra_deg_min = min(ra_list) % 360.0
    ra_deg_max = max(ra_list) % 360.0
    dec_deg_min = min(dec_list)
    dec_deg_max = max(dec_list)
    return ra_deg_min, ra_deg_max, dec_deg_min, dec_deg_max


# FOV_and_LANDOLT_FUNCTIONS________________________________________ = 0
#
#
# def get_landolt_fovs(fov_directory=FOV_DIRECTORY):
#     """ Return list of FOV objects, one for each Landolt standard field."""
#     all_filenames = pd.Series([e.name for e in os.scandir(fov_directory) if e.is_file()])
#     fov_list = []
#     for filename in all_filenames:
#         fov_name = filename.split('.')[0]
#         if not fov_name.startswith(('$', 'Std_NGC')):  # these are not Landolt FOVs
#             fov_object = Fov(fov_name)
#             if fov_object.is_valid:
#                 fov_list.append(fov_object)
#     return fov_list
#
# CATALOG_COMPARISON_TESTS________________________________________ = 0
#
# def regress_apass10_on_refcat2(ra_deg, dec_deg, radius_deg, apass_band, refcat2_bands, intercept=False):
#     """ Run linear regressions on APASS 10 band vs ATLAS refcat2 bands (esp. Pan-Starrs griz).
#     :param ra_deg:
#     :param dec_deg:
#     :param radius_deg:
#     :param apass_band: band from APASS 10 to use as dependent variable [string].
#     :param refcat2_bands: band or list of bands from refcat2 to use as indep vars [str or list of strs].
#     :return: results [regression object from statsmodels package].
#     """
#     if not isinstance(refcat2_bands, list):
#         refcat2_bands = [str(refcat2_bands)]
#
#     # Get ATLAS refcat2 stars (independent variables):
#     cos_dec = cos(dec_deg / DEGREES_PER_RADIAN)
#     ra_deg_min = (ra_deg - radius_deg / cos_dec) % 360
#     ra_deg_max = (ra_deg + radius_deg / cos_dec) % 360
#     dec_deg_min = (dec_deg - radius_deg)
#     dec_deg_max = (dec_deg + radius_deg)
#     df_refcat2 = get_refcat2(ra_deg_min=ra_deg_min, ra_deg_max=ra_deg_max,
#                              dec_deg_min=dec_deg_min, dec_deg_max=dec_deg_max)
#     df_refcat2 = remove_overlapping_comps(df_refcat2)
#     r_mag_ok = pd.Series([(r >= 10.0) and (r <= 16.0) for r in df_refcat2.loc[:, 'r']])
#     b_v_estimate = pd.Series([0.830 * g - 0.803 * r
#                               for (g, r) in zip(df_refcat2.loc[:, 'g'], df_refcat2.loc[:, 'r'])])
#     b_v_ok = pd.Series([(bv >= 0.5) and (bv <= 0.95) for bv in b_v_estimate])
#     dupvar_ok = pd.Series([(d == 0 or d == 2) for d in df_refcat2['dupvar']])
#     dgaia_ok = pd.Series([d > 0 for d in df_refcat2['dG_gaia']])
#     dg_ok = pd.Series([dg <= 20 for dg in df_refcat2['dg']])
#     dr_ok = pd.Series([dr <= 20 for dr in df_refcat2['dr']])
#     di_ok = pd.Series([dg <= 20 for dg in df_refcat2['di']])
#     rp1_ok = pd.Series([True if pd.isnull(rp1) else (rp1 >= 9) for rp1 in df_refcat2['RP1']])
#     r1_ok = pd.Series([True if pd.isnull(r1) else (r1 >= 13) for r1 in df_refcat2['R1']])
#     keep_rows = r_mag_ok & b_v_ok & dgaia_ok & dg_ok & dr_ok & di_ok & rp1_ok & r1_ok
#     df_refcat2 = df_refcat2[list(keep_rows)]
#
#     # Get APASS 10 stars (dependent variable):
#     df_apass = get_apass10_comps(ra_deg, dec_deg, radius_deg, mp_color_only=False)
#     mag_ok = pd.Series([mag is not None for mag in df_apass.loc[:, apass_band]])
#     e_sr_ok = pd.Series([(e <= 0.15) for e in df_apass.loc[:, 'e_SRmag']])
#     e_si_ok = pd.Series([(e <= 0.25) for e in df_apass.loc[:, 'e_SImag']])
#     keep_rows = mag_ok & e_sr_ok & e_si_ok
#     df_apass = df_apass[list(keep_rows)]
#
#     # For each APASS 10 star, match a refcat2 star if possible:
#     mag_dict_list = []
#     for i_apass in df_apass.index:
#         mag_dict = dict()
#         i_refcat2 = find_matching_comp(df_refcat2, df_apass.loc[i_apass, 'degRA'],
#                                        df_apass.loc[i_apass, 'degDec'])
#         if i_refcat2 is not None:
#             mag_dict['index'] = 'apass_' + str(i_apass)
#             mag_dict['y_' + apass_band] = df_apass.loc[i_apass, apass_band]
#             # mag_dict['y_BminusV'] = df_apass.loc[i_apass, 'Bmag'] - df_apass.loc[i_apass, 'Vmag']
#             for band in refcat2_bands:
#                 mag_dict[band] = df_refcat2.loc[i_refcat2, band]
#             if all([val is not None for val in mag_dict.values()]):
#                 mag_dict_list.append(mag_dict)
#     this_index = [mag_dict['index'] for mag_dict in mag_dict_list]
#     df_mags = pd.DataFrame(data=mag_dict_list, index=this_index)
#
#     # Perform regression; APASS band is indep var, refcat2 bands are dep vars:
#     df_y = df_mags.loc[:, 'y_' + apass_band]
#     # df_y = df_mags.loc[:, 'y_BminusV']
#     df_x = df_mags.loc[:, refcat2_bands]
#     if intercept is True:
#         df_x.loc[:, 'intercept'] = 1.0
#     weights = len(df_mags) * [1.0]
#     result = sm.WLS(df_y, df_x, weights).fit()  # see bulletin2.util
#     print(result.summary())
#     print('mse_resid =', '{0:.4f}'.format(result.mse_resid), ' mag.')
#     return result


# def regress_landolt_r_mags():
#     """ Regress Landolt R magnitudes against matching ATLAS refcat2 Pan-STARRS g, r, and i magnitudes."""
#     # First, collect all data into dataframe df_mags:
#     fov_list = get_landolt_fovs(FOV_DIRECTORY)
#     mag_dict_list = []
#     for fov in fov_list:
#         if fov.target_type.lower() == 'standard':
#             # Get ATLAS refcat2 stars within vicinity of FOV stars:
#             ra_degs = [star.ra for star in fov.aavso_stars]
#             ra_deg_min = min(ra_degs)
#             ra_deg_max = max(ra_degs)
#             dec_degs = [star.dec for star in fov.aavso_stars]
#             dec_deg_min = min(dec_degs)
#             dec_deg_max = max(dec_degs)
#             df_refcat2 = get_refcat2(ra_deg_min=ra_deg_min, ra_deg_max=ra_deg_max,
#                                      dec_deg_min=dec_deg_min, dec_deg_max=dec_deg_max)
#             df_refcat2 = remove_overlapping_comps(df_refcat2)
#
#             for fov_star in fov.aavso_stars:
#                 refcat2_matching = find_matching_comp(df_refcat2, fov_star.ra, fov_star.dec)
#                 if refcat2_matching is not None:
#                     g_mag = df_refcat2.loc[refcat2_matching, 'G']
#                     r_mag = df_refcat2.loc[refcat2_matching, 'R']
#                     i_mag = df_refcat2.loc[refcat2_matching, 'I']
#                     if g_mag is not None and r_mag is not None and i_mag is not None:
#                         try:
#                             mag_dict = {'fov': fov.fov_name, 'fov_star': fov_star.star_id,
#                                         'Landolt_R': fov_star.mags['R'][0],
#                                         'g': g_mag, 'r': r_mag, 'i': i_mag}
#                             mag_dict_list.append(mag_dict)
#                         except KeyError:
#                             print(' >>>>> Caution:', fov.fov_name, fov_star.star_id, 'is missing R mag.')
#     this_index = [mag_dict['fov'] + '_' + mag_dict['fov_star'] for mag_dict in mag_dict_list]
#     df_mags = pd.DataFrame(data=mag_dict_list, index=this_index)
#
#     # Perform regression; Landolt R is indep var, and dep variables are matching refcat2 g, r, and i mags:
#     # return df_mags  # for now.
#     df_y = df_mags[['Landolt_R']]
#     df_x = df_mags[['r']]
#     # df_x = df_mags[['g', 'r', 'i']]
#     df_x.loc[:, 'intercept'] = 1.0
#     weights = len(df_mags) * [1]
#     result = sm.WLS(df_y, df_x, weights).fit()  # see bulletin2.util
#     print(result.summary())
#     return result
