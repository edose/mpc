__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.formula.api as smf

# From this (mpc) package:
from mpc.catalogs import Refcat2, get_bounding_ra_dec
from mpc.mp_planning import all_mpfile_names, MPfile
import mpc.ini
import mpc.mp_phot

# From EVD package photrix (deprecated import--> prefer astropak):
# from photrix.image import Image, FITS

# From EVD package astropak (preferred):
import astropak.ini
import astropak.stats
import astropak.util

# MPC_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BOOT_INI_FILENAME = 'defaults.ini'

# For color handling:
# DEFAULT_FILTERS_FOR_MP_COLOR_INDEX = ('V', 'I')
# COLOR_INDEX_PASSBANDS = ('r', 'i')
# DEFAULT_MP_RI_COLOR = 0.22  # close to known Sloan mean (r-i) for MPs.
MAX_COLOR_COMP_UNCERT = 0.015  # mag
# COLOR_CONTROL_FILENAME = 'color_control.ini'
# COLOR_CONTROL_TEMPLATE_FILENAME = 'color_control.template'
# COLOR_LOG_FILENAME = 'color_log.txt'

# DF_OBS_ALL_FILENAME = 'df_obs_all.csv'
# DF_IMAGES_ALL_FILENAME = 'df_images_all.csv'
# DF_COMPS_ALL_FILENAME = 'df_comps_all.csv'
# COLOR_IMAGE_PREFIX = 'Image_Color_'


_____INITIALIZE_COLOR_WORKFLOW______________________________ = 0


def write_control_files():
    defaults_dict = mpc.ini.make_defaults_dict()
    context = mpc.mp_phot.get_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
    log_file = open(defaults_dict['color log filename'], mode='a')  # set up append to log file.

    # Write color control template file if it doesn't already exist:
    color_control_template_filename = defaults_dict['color control template filename']
    fullpath = os.path.join(this_directory, color_control_template_filename)
    if not os.path.exists(fullpath):
        _write_color_control_ini_template(this_directory, defaults_dict)
        log_file.write('New ' + color_control_template_filename + ' file written.\n')

    # Write color control ini file (stub) if it doesn't already exist:
    color_control_filename = defaults_dict['color control filename']
    fullpath = os.path.join(this_directory, color_control_filename)
    if not os.path.exists(fullpath):
        _write_color_control_ini_stub(this_directory, defaults_dict)
        log_file.write('New ' + color_control_filename + ' file written.\n')


_____COLOR_INDEX_PHOTOMETRY________________________________________________ = 0


def do_color_SRI():
    """ Convenience function. """
    do_color_2filters(filters=('SR', 'SI'), target_color=('SR', 'SI'), color_index_passbands=('SR', 'SI'))


def do_color_SRI_from_johnson():
    """ Convenience function. """
    do_color_2filters(filters=('R', 'I'), target_color=('SR', 'SI'), color_index_passbands=('SR', 'SI'))


def do_color_2filters(filters=None, target_color=None, color_index_passbands=None):
    """ Estimate ONE MP Color Index from images in TWO FILTERS.
        First run the 3 initial mp_phot functions (start(), assess(), make_dfs()) to make the dataframes.
        Then run this to get color index.
        ALL PARAMETERS MANDATORY.
    :param filters: specify the two filters for which images are to be used, e.g., ('SR', 'SI') to
        use Sloan r' and i' images. [2-tuple or 2-list of strings] MANDATORY: no default.
    :param target_color: specify the two passbands defining the desired color to measure,
        e.g., ('SG', 'SR') to measure color SG-SR ('SGR' in LCDB parlance). MANDATORY: no default.
        [2-tuple or 2-list of strings]
    :param color_index_passbands: specify the two passbands defining the color index, e.g., ('SR', 'SI') to
        define color in Sloan SR and SI (typical case). In any case, the two passbands must be
        represented in the ATLAS refcat2 catalog (prob. as 'r' and 'i') and must be included
        in the local catalog object. [2-tuple of strings] MANDATORY: no default.
    :return: None. Only writes to screen and to log file, makes diagnostic plots.
    """
    # Verify that parms passed in are of correct form:
    filters_parm_ok, target_color_parm_ok, color_index_passbands_ok = False, False, False
    if isinstance(filters, tuple) or isinstance(filters, list):
        filters = tuple(filters)  # ensure tuple.
        if len(filters) == 2:
            if all([isinstance(f, str) for f in filters]):
                filters_parm_ok = True
    if isinstance(target_color, tuple) or isinstance(target_color, list):
        target_color = tuple(target_color)
        if len(target_color) == 2:
            if all([isinstance(pb, str) for pb in target_color]):
                target_color_parm_ok = True
    if isinstance(color_index_passbands, tuple) or isinstance(color_index_passbands, list):
        color_index_passbands = tuple(color_index_passbands)
        if len(color_index_passbands) == 2:
            if all([isinstance(pb, str) for pb in color_index_passbands]):
                color_index_passbands_ok = True
    if not filters_parm_ok:
        print(' >>>>> ERROR: filters invalid:', str(filters))
    if not target_color_parm_ok:
        print(' >>>>> ERROR: target_color invalid:', str(target_color))
    if not color_index_passbands_ok:
        print(' >>>>> ERROR: color_index_passbands invalid:', str(color_index_passbands))
    if not all([filters_parm_ok, target_color_parm_ok, color_index_passbands_ok]):
        return
    fa, fb = filters
    pb_a, pb_b = target_color
    ci_a, ci_b = color_index_passbands
    print(' >>>>> Starting do_color_2filters(): color', pb_a + '-' + pb_b, 'from filters', fa, '&', fb,
          'using color index', ci_a + '-' + ci_b)

    # ===== Get context, write log file header and color control stub:
    defaults_dict = mpc.ini.make_defaults_dict()
    context = mpc.mp_phot.get_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    log_file = open(defaults_dict['color log filename'], mode='a')  # set up append to log file.
    log_file.write('\n===== do_color()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # ===== Get needed data, verify all required data are present (e.g. extinctions, transforms):


    color_control_dict = make_color_control_dict(this_directory, defaults_dict)
    observer_dict = mpc.ini.make_observer_dict(defaults_dict)
    iiii = 4




    # ===== Load and prepare session's master dataframe, and rename passband columns, e.g. 'r' -> 'SR':
    df_all = mpc.mp_phot.make_df_all(filters_to_include=(fa, fb), comps_only=False,
                                     require_mp_obs_each_image=True)
    df_all = df_all.rename(columns={'g': 'SG',   'r': 'SR',   'i': 'SI',   'z': 'SZ',
                                    'dg': 'dSG', 'dr': 'dSR', 'di': 'dSI', 'dz': 'dSZ'})

    # ===== Make list of images taken with both filters:
    df_mps = df_all.loc[df_all['Type'] == 'MP', :]
    fa_filenames = df_mps.loc[df_mps['Filter'] == fa, 'FITSfile'].to_list()
    fb_filenames = df_mps.loc[df_mps['Filter'] == fb, 'FITSfile'].to_list()
    if len(fa_filenames) <= 0 or len(fb_filenames) <= 0:
        print(' >>>>> ERROR: no images for one or both of filters: \'' + fa + '\', \'' + fb + '\'')
        return
    # all_filenames = fa_filenames + fb_filenames
    del fa_filenames, fb_filenames  # at risk of invalidation during screening.

    # ===== Remove images per user request (color_control_dict):
    df_screened_obs = df_all.copy()
    remove_by_image = df_screened_obs['FITSfile'].isin(color_control_dict['omit images'])
    obs_to_keep = (~ remove_by_image).to_list()
    df_screened_obs = df_screened_obs.loc[obs_to_keep, :]

    # ===== Remove comps (from both filters) per user request (color_control_dict):
    to_remove_by_comp_id = df_screened_obs['SourceID'].isin(color_control_dict['omit comps'])
    is_comp = (df_screened_obs['Type'] == 'Comp')
    obs_to_remove = (to_remove_by_comp_id & is_comp)
    obs_to_keep = (~ obs_to_remove).to_list()
    df_screened_obs = df_screened_obs.loc[obs_to_keep, :]

    # ===== Remove obs (MP and comp) with unusable (None or NaN) instmag or instmagsigma:
    df_screened_obs = df_screened_obs.dropna(subset=['InstMag', 'InstMagSigma'])

    # ===== Identify (and keep) images having exactly one MP obs:
    mp_obs_by_filename = df_screened_obs.loc[df_screened_obs['Type'] == 'MP', 'FITSfile'].value_counts()
    filenames_one_mp = [fn for fn in mp_obs_by_filename.index if mp_obs_by_filename[fn] == 1]
    obs_to_keep = df_screened_obs['FITSfile'].isin(filenames_one_mp).to_list()
    df_screened_obs = df_screened_obs.loc[obs_to_keep, :]

    # ===== Remove *images* whose MP obs uncertainty is too high:
    is_mp_obs = (df_screened_obs['Type'] == 'MP')
    obs_uncert_ok = (df_screened_obs['InstMagSigma'] <=
                     color_control_dict['max mp obs mag uncertainty'])
    row_is_good_mp = (is_mp_obs & obs_uncert_ok).to_list()
    filenames_to_keep = df_screened_obs.loc[row_is_good_mp, 'FITSfile']
    obs_to_keep = df_screened_obs['FITSfile'].isin(filenames_to_keep).to_list()
    df_screened_obs = df_screened_obs.loc[obs_to_keep, :]

    # ===== Remove comps whose observation uncertainty is too high:
    # TODO: ? make max uncert relative to instrumental mag (to more accurately remove true outliers) ?
    is_comp_obs = df_screened_obs['Type'] == 'Comp'
    uncert_too_high = df_screened_obs['InstMagSigma'] > color_control_dict['max comp obs mag uncertainty']
    obs_to_remove = (is_comp_obs & uncert_too_high)
    obs_to_keep = (~ obs_to_remove).to_list()
    df_screened_obs = df_screened_obs.loc[obs_to_keep, :]

    df_screened_comp_obs = df_screened_obs.loc[df_screened_obs['Type'] == 'Comp', :]
    df_screened_mp_obs = df_screened_obs.loc[df_screened_obs['Type'] == 'MP', :]

    # Apply selection criteria to comp obs (MP obs don't have catalog values):
    remove_by_min_mag = (df_screened_comp_obs['SR'] < color_control_dict['min sr mag'])
    remove_by_max_mag = (df_screened_comp_obs['SR'] > color_control_dict['max sr mag'])
    remove_by_cat_mag_uncert = (df_screened_comp_obs['dSR'] > color_control_dict['max catalog dsr mmag'])
    color = (df_screened_comp_obs['SR'] - df_screened_comp_obs['SI'])
    remove_by_min_color = (color < color_control_dict['min sloan ri color'])
    remove_by_max_color = (color > color_control_dict['max sloan ri color'])
    obs_to_remove = (remove_by_min_mag | remove_by_max_mag | remove_by_cat_mag_uncert |
                     remove_by_min_color | remove_by_max_color)
    obs_to_keep = (~ obs_to_remove).to_list()
    df_screened_comp_obs = df_screened_comp_obs.loc[obs_to_keep, :]

    # Recombine screened obs of comps and MPs:
    df_screened_obs = df_screened_comp_obs.append(df_screened_mp_obs)

    # Remove obviously bad images (number of comps less than half max number of comps in any image):
    comp_counts = df_screened_obs.loc[df_screened_obs['Type'] == 'Comp', 'FITSfile'].value_counts()
    max_comp_count = comp_counts.max()
    fn_too_few_comps = [fn for fn in comp_counts.index if comp_counts[fn] < max_comp_count / 2.0]
    remove_too_few_comps = df_screened_obs['FITSfile'].isin(fn_too_few_comps)
    obs_to_keep = (~ remove_too_few_comps).to_list()
    df_screened_obs = df_screened_obs.loc[obs_to_keep, :]

    # Remove comps absent from any image (so that all images have same comp set at this point):
    image_count = len(df_screened_obs.loc[df_screened_obs['Type'] == 'Comp', 'FITSfile'].drop_duplicates())
    image_count_per_comp = df_screened_obs.loc[df_screened_obs['Type'] == 'Comp', 'SourceID'].value_counts()
    comps_not_in_every_image = [id for id in image_count_per_comp.index
                                if image_count_per_comp[id] < image_count]
    remove_comps_not_in_every_image = df_screened_obs['SourceID'].isin(comps_not_in_every_image)
    obs_to_keep = (~ remove_comps_not_in_every_image).to_list()
    df_screened_obs = df_screened_obs.loc[obs_to_keep, :]

    # Apply Omit Obs criterion (this should be done separately, and after all others):
    remove_by_obs_id = df_screened_obs['Serial'].isin(color_control_dict['omit obs'])
    obs_to_keep = (~ remove_by_obs_id).to_list()
    df_screened_obs = df_screened_obs.loc[obs_to_keep, :]

    # Lastly: remove (with warning) images with fewer than minimum number of comps (extremely rare):
    comp_count_per_image = df_screened_obs.loc[df_screened_obs['Type'] == 'Comp', 'FITSfile'].value_counts()
    images_too_few_comps = [fn for fn in comp_count_per_image.index
                            if comp_count_per_image[fn] < color_control_dict['min valid comps per image']]
    remove_too_few_comps = df_screened_obs['FITSfile'].isin(images_too_few_comps)
    obs_to_keep = (~ remove_too_few_comps).to_list()
    df_screened_obs = df_screened_obs.loc[obs_to_keep, :].sort_values(by=['JD_mid', 'Type', 'SourceID'])

    iiii = 4


    # For each filter, build and run model, then get no-CI MP mag for each image:
    df_results = None # accumulator for final mag results, both filters.
    for (f, pb) in [(fa, pb_a), (fb, pb_b)]:
        df_model_this_filter = df_screened_obs.loc[df_screened_obs['Filter'] == f, :].copy()
        # Here, count MP obs in this filter, error if not at least one.
        model = SessionModel_Color(df_model_this_filter, f, pb, color_index_passbands, color_control_dict)
        if df_results is None:
            df_results = model.df_mp_mags
        else:
            df_results = df_results.append(model.df_mp_mags)

    # Run second (vs JD) regression, backcalculating for best CI & sigma.


    # Generate diagnostic plots (each filter):


    # Write to console and to log file:




    # # Generate plot (may as well do it here rather than in another fn, as all data is already gathered):
    # fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(11, 8.5))  # (width, height) in "inches"
    # ax = axes  # not subscripted if just one subplot in Figure
    # page_title = 'MP ' + mp_string + '   AN ' + an_string + '   ::   Color Index regression'
    # ax.set_title(page_title, color='darkblue', fontsize=20, pad=30)
    # plot_annotation = '{0:.3f}'.format(mp_color) +\
    #     ' ' + u'\u00B1' + ' {0:.3f}'.format(sigma_color) + ' from ' + filter_string +\
    #     ' (' + str(n_ci_comps) + ' comps used)'
    # fig.text(x=0.5, y=0.87, s=plot_annotation,
    #          verticalalignment='top', horizontalalignment='center', fontsize=12)
    # x_values = [x[1] for x in result.model.exog]  # unpack from 2-d array.
    # y_values = result.model.endog
    # ax.set_xlabel('Diff(instrumental magnitudes): ' + filter_string)
    # ax.set_ylabel('Color index: ' + passband_string)
    # ax.grid(True, color='lightgray', zorder=-1000)
    # ax.scatter(x=x_values, y=y_values, alpha=0.7, color='black', zorder=+1000)
    # x_mp = [mag_diff]
    # y_mp = [mp_color]
    # ax.scatter(x=x_mp, y=y_mp, alpha=1, s=144, marker='X', color='orangered', zorder=+1001)
    #
    # # Label with comp ID: (1) outlier residuals, (2) lowest and highest inst. mag diffs:
    # is_outlier = [abs(t) > 2.5 for t in result.resid_pearson]
    # is_highest = [x == max(x_values) for x in x_values]
    # is_lowest = [x == min(x_values) for x in x_values]
    # to_label = [(o or h or l) for (o, h, l) in zip(is_outlier, is_highest, is_lowest)]
    # labels = df_fit['comp_id']
    # for x, y, label, add_label, t in zip(x_values, y_values, labels, to_label, result.resid_pearson):
    #     if add_label:
    #         if t > 0:
    #             ax.annotate(label, xy=(x, y), xytext=(-4, 4),
    #                         textcoords='offset points', ha='right', va='bottom', rotation=-40)
    #         else:
    #             ax.annotate(label, xy=(x, y), xytext=(4, -4),
    #                         textcoords='offset points', ha='left', va='top', rotation=-40)
    # plt.show()
    # # fig.savefig(COLOR_PLOT_FILENAME)
    #
    # # Write final results to screen and to log:
    # output = 'MP color index (' + passband_string + ') = ' + '{0:.3f}'.format(mp_color) +\
    #          ' ' + u'\u00B1' + ' {0:.3f}'.format(sigma_color) + ' from ' + filter_string +\
    #          '  (' + str(n_ci_comps) + ' comps used)'
    # print(output + '\n   Write this to ' + CONTROL_FILENAME + ': #MP_RI_COLOR  ' +
    #       '{0:.3f}'.format(mp_color))
    # log_file.write(output)
    # log_file.close()


class SessionModel_Color:
    """ Applies to *one* filter in one session.
        NOTE 20201021: For now we use only 1st-order (linear in Color Index) transforms.
        [2nd-order (quadratic in CI) will be too hard to backsolve, and probably unstable.]
    """
    def __init__(self, df_model, filter, passband, color_index_passbands, color_control_dict):
        """ :param df_model:
            :param filter:
            :param passband:
            :param color_index_passbands:
            :param color_control_dict:
        """
        self.df_model = df_model
        self.filter = filter
        self.passband = passband
        self.color_index_passbands = color_index_passbands

        self.transform_option = color_control_dict['Transform'][(filter, passband)]  # TODO: correct this.
        self.extinction_option = color_control_dict['Extinction'][filter]
        self.fit_vignette = color_control_dict['fit_extinction']
        self.fit_xy = color_control_dict['fit_xy']
        self.fit_jd = color_control_dict['fit_jd']

        self.df_used = df_model.copy().loc[df_model['UseInModel'], :]  # only observations used in model.
        self.df_used_comps_only = self.df_used.loc[(self.df_used['Type'] == 'Comp'), :].copy()
        self.df_used_mps_only = self.df_used.loc[(self.df_used['Type'] == 'MP'), :].copy()
        self.image_filenames = list(self.df_used['FITSfile'].drop_duplicates())
        self.n_images = len(self.image_filenames)

        self.dep_var_name = 'InstMag_with_offsets'
        self.mm_fit = None      # placeholder for the fit result [photrix MixedModelFit object].
        self.transform = None   # placeholder for this fit parameter result [scalar].
        self.transform_fixed = None     # "
        self.extinction = None          # "
        self.vignette = None            # "
        self.x = None                   # "
        self.y = None                   # "
        self.jd1 = None                 # "
        self.statsmodels_object = None  # placeholder for regression model object.
        self.df_mp_mags = None          # placeholder for results Dataframe.
        self.is_valid = False  # will be updated to True iff everything completes normally.

        if self.n_images <= 0:
            print(' >>>>> ERROR: SessionModel_Color(', filter, passband, 'was passed no images.')
        else:
            self.regression_type = 'ols' if self.n_images == 1 else 'mixed-model'
            self._prep_and_do_regression()
            self._calc_mp_mags()

    def _prep_and_do_regression(self):
        fit_summary_lines = []
        fixed_effect_var_list = []

        # Initiate dependent-variable offset, which will aggregate all such offset terms:
        dep_var_offset = self.df_used_comps_only[self.passband].copy()  # *copy* to avoid damage.

        # Add columns CI and CI2 as indep variables, depending on color index used:
        self.df_used_comps_only['CI'] = self.df_used_comps_only[self.color_index_passbands[0]] -\
                                        self.df_used_comps_only[self.color_index_passbands[1]]
        self.df_used_comps_only['CI2'] = [ci ** 2 for ci in self.df_used_comps_only['CI']]

        # Handle transform option (transform=coeff(s), catalog CI, CI2=indep var(s)):
        transform_handled = False
        if self.transform_option == 'fit=1':
            fixed_effect_var_list.append('CI')
            transform_handled = True
        # elif self.transform_option == 'fit=2':
        #     fixed_effect_var_list.extend(['CI', 'CI2'])
        #     transform_handled = True
        elif isinstance(self.transform_option, list):
            if self.transform_option[0] == 'use':
                if len(self.transform_option) in [2, 3]:
                    coeffs = (self.transform_option[1:] + [0.0])[:2]  # 2nd-order coeff -> 0 if absent.
                    dep_var_offset += coeffs[0] * self.df_used_comps_only['CI'] +\
                                      coeffs[1] * self.df_used_comps_only['CI2']
                    transform_handled = True
        if not transform_handled:
            print(' >>>>> WARNING: _prep_and_do_regression() cannot handle transform option =',
                  str(self.transform_option))
            return None

        # Handle extinction option (extinction=coeff, ObsAirmass=indep var):
        extinction_handled = False
        if self.extinction_option == 'fit':
            fixed_effect_var_list.append()
            extinction_handled = True
        elif isinstance(self.extinction_option, list):
            if self.extinction_option[0] == 'use':
                coeff = self.extinction_option[1]
                dep_var_offset += coeff * self.df_used_comps_only['ObsAirmass']
                extinction_handled = True
        if not extinction_handled:
            print(' >>>>> WARNING: _prep_and_do_regression() cannot handle extinction option =',
                  str(self.extinction_option))
            return None

        # Build all other fixed-effect (independent) variable lists and dep-var offsets:
        if self.fit_vignette:
            fixed_effect_var_list.append('Vignette')
        if self.fit_xy:
            fixed_effect_var_list.extend(['X1024', 'Y1024'])
        if self.fit_jd:
            fixed_effect_var_list.append('JD_fract')
        if len(fixed_effect_var_list) == 0:
            fixed_effect_var_list = ['JD_fract']  # as statsmodels requires >= 1 fixed-effect variable.

        # Build dependent (y) variable:
        self.df_used_comps_only[self.dep_var_name] = self.df_used_comps_only['InstMag'] - dep_var_offset

        if self.n_images == 1:
            # Do std OLS regression on comps (no mixed-model or random effect possible):
            import statsmodels.api as sm
            dep_var = self.df_used_comps_only[self.dep_var_name]
            indep_vars = self.df_used_comps_only[fixed_effect_var_list]
            indep_vars_with_constant = sm.add_constant(indep_vars)  # adds column 'const' of ones.
            self.statsmodels_object = sm.OLS(dep_var, indep_vars_with_constant).fit
        else:
            # Do mixed-model regression (random effect = FITSfile column):
            import warnings
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
            warnings.simplefilter('ignore', ConvergenceWarning)
            random_effect_var_name = 'FITSfile'  # cirrus effect is per-image.
            self.mm_fit = astropak.stats.MixedModelFit(data=self.df_used_comps_only,
                                                       dep_var=self.dep_var_name,
                                                       fixed_vars=fixed_effect_var_list,
                                                       group_var=random_effect_var_name)
            self.statsmodels_object = self.mm_fit.statsmodels_object

        # Print summaries, save fit summary file, exit.
        print(60 * '*', '\nFilter', self.filter + ':', )
        print(self.statsmodels_object.summary())
        print('sigma =', '{0:.1f}'.format(1000.0 * self.mm_fit.sigma), 'mMag.')
        if not self.mm_fit.converged:
            msg = ' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.'
            print(msg)
            fit_summary_lines.append(msg)
        color_index_string = self.color_index_passbands[0] + '-' + self.color_index_passbands[1]
        filename = 'fit_summary_' + color_index_string + '_' + self.filter + '.txt'
        mpc.mp_phot.write_text_file(filename,
                                    'Mixed-model Regression for directory ' +
                                    mpc.mp_phot.get_context()[0] + '\n\n' +
                                    'Color index ' + color_index_string +
                                    '   Filter ' + self.filter + '\n' +
                                    '\n'.join(fit_summary_lines) +
                                    self.mm_fit.statsmodels_object.summary().as_text() +
                                    '\n\nsigma = ' +
                                    '{0:.1f}'.format(1000.0 * self.mm_fit.sigma) + ' mMag.')

    def _calc_mp_mags(self):
        """ Evaluate MP magnitudes, writes them into a dataframe self.mp_mags.
        :return: [None, writes results to self.mp_mags, a Dataframe]
        """
        # This fn doesn't care whether model is OLS or Mixed-model.
        # Set CI and CI2 to zero for now; we will backsolve for CI separately (later).
        self.df_used_mps_only['CI'] = 0.0
        self.df_used_mps_only['CI2'] = 0.0

        # "Predict" inst mags but employing all catmag=0 and CI,CI2 = 0.
        predictions_from_model_only = self.statsmodels_object.predict(self.df_used_mps_only)

        # Correct for dep_var_offsets:
        dep_var_offsets = len(self.df_used_mps_only) * [0.0]
        if isinstance(self.extinction_option, list):
            if self.extinction_option[0] == 'use':
                coeff = self.extinction_option[1]
                dep_var_offsets += coeff * self.df_used_comps_only['ObsAirmass']
        predictions_with_offsets = predictions_from_model_only + dep_var_offsets

        # Make small dataframe with columns: FITSfile, Filter, PredictedInstMag, InstMag, InstMagSigma,
        #     Transform, JD_mid.
        self.df_mags = self.df_used_mps_only[['FITSfile', 'Filter', 'InstMag',
                                              'InstMagSigma', 'JD_mid']].copy()
        self.df_mags['PredictedInstMag'] = predictions_with_offsets
        self.df_mags['Transform'] = self.transform


_____SUPPORT_FUNCTIONS______________________________________ = 0


def _write_color_control_ini_template(this_directory, defaults_dict):
    color_control_template_filename = defaults_dict['color control template filename']
    fullpath = os.path.join(this_directory, color_control_template_filename)

    lines = ['#----- This is template file ' + color_control_template_filename,
             '#----- in directory ' + this_directory,
             '#',
             '[Selection]',
             'Omit Comps  = string  ->  omit comps',
             'Omit Obs    = string  ->  omit obs',
             'Omit Images = string  ->  omit images',
             'Max MP Mag Uncertainty = string -> max mp mag uncertainty',
             'Min SR Mag  = float   ->  min sr mag',
             'Max SR Mag  = float   ->  max sr mag',
             'Max Catalog dSR mmag  = float  ->  max catalog dsr mmag',
             'Min Sloan RI Color    = float  ->  min sloan ri color',
             'Max Sloan RI Color    = float  ->  max sloan ri color',
             '',
             '[Regression]',
             'MP Sloan RI Color = float   ->  mp sloan ri color',
             'Transform         = string  ->  transform',
             'Extinction        = string  ->  extinction',
             'Fit Vignette      = boolean ->  fit vignette',
             'Fit XY            = boolean ->  fit_xy',
             'Fit JD            = boolean ->  fit_jd',
             '#----- end of template'
             ]
    lines = [line + '\n' for line in lines]
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(lines)


def _write_color_control_ini_stub(this_directory, defaults_dict):
    """ Write user-ready stub of color control file, only if file doesn't already exist.
    :param this_directory:
    :param defaults_dict:
    :param log_file:
    :return:
    """
    color_control_filename = defaults_dict['color control filename']
    fullpath = os.path.join(this_directory, color_control_filename)

    lines = ['#----- This is ' + color_control_filename + ' for directory:',
             '#-----    ' + this_directory,
             '#',
             '[Ini Template]',
             'Filename = ' + defaults_dict['color control template filename'],
             '',
             '[Selection]',
             '# Omit Comps & Omit Obs: values may extend to multiple lines if necessary.',
             'Omit Comps = ',
             'Omit Obs = ',
             '# Omit Images: give filename, (with or) without .fts at end.',
             'Omit Images = ',
             'Max MP Obs Mag Uncertainty = ' + str(defaults_dict['max mp obs mag uncertainty']),
             'Max Comp Obs Uncertainty = ' + str(defaults_dict['max comp obs mag uncertainty']),
             'Min SR Mag = ' + str(defaults_dict['min sr mag']),
             'Max SR Mag = ' + str(defaults_dict['max sr mag']),
             'Max Catalog dSR mmag = ' + str(defaults_dict['max catalog dsr mmag']),
             'Min Sloan RI Color = ' + str(defaults_dict['min sloan ri color']),
             'Max Sloan RI Color = ' + str(defaults_dict['max sloan ri color']),
             '',
             '[Regression]',
             '# Transform = Filter Passband CI_passband1 CI_passband2  -or-',
             '# Transform = Filter Passband Command -where- Command: Fit or Use -0.03',
             '# One per line only. First order only for do_color() [2020-10-21].',
             'Transform = ',
             '# Extinction = Filter Command -where- Command: Fit -or- Use +0.16.',
             '# One per line only. Recommend \'Use\' if at all possible.',
             'Extinction = ',
             'Fit Vignette = Yes',
             'Fit XY = No',
             '# Strongly recommend Fit JD = No for do_color().',
             'Fit JD = No'
             ]
    lines = [line + '\n' for line in lines]
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(lines)


def make_color_control_dict(this_directory, defaults_dict):
    color_control_ini = astropak.ini.IniFile(os.path.join(this_directory,
                                                          defaults_dict['color control filename']))
    color_control_dict = color_control_ini.value_dict

    # Parse and overwrite 'omit comps', 'omit obs', and 'omit images':
    comps_to_omit = color_control_dict['omit comps'].replace(',', ' ').split()
    color_control_dict['omit comps'] = tuple(comps_to_omit)
    obs_to_omit = color_control_dict['omit obs'].replace(',', ' ').split()
    color_control_dict['omit obs'] = tuple(obs_to_omit)
    color_control_dict['omit images'] = tuple(color_control_dict['omit images'])

    # Parse and overwrite 'transforms':
    transform_dict = dict()
    transform_lines = [line.strip() for line in color_control_dict['transforms']]
    for line in transform_lines:
        items = line.replace(',', ' ').split()
        if len(items) in [5, 6]:
            key = tuple(items[:4])
            command = items[4].lower().strip()
            if command == 'fit':
                values = tuple(['fit'])
            elif command == 'use' and len(items) >= 6:
                values = tuple(['use', mpc.ini.float_or_warn(items[5], 'Transforms ' + line)])
            else:
                values = None
            transform_dict[key] = values
        else:
            print(' >>>>> ERROR:', defaults_dict['color control filename'], 'bad transform line:', line)
    color_control_dict['transforms'] = transform_dict

    # Parse and overwrite 'extinctions':
    extinction_dict = dict()
    extinction_lines = [line.strip() for line in color_control_dict['extinctions']]
    for line in extinction_lines:
        items = line.replace(',', ' ').split()
        if len(items) in [2, 3]:
            key = items[0]
            command = items[1].lower().strip()
            value = None
            if command == 'fit':
                value = tuple(['fit'])
            elif command == 'use':
                if len(items) >= 3:
                    value = tuple(['use', mpc.ini.float_or_warn(items[2], 'Extinctions ' + line)])
                else:
                    print(' >>>>> ERROR:', defaults_dict['color control filename'],
                          'bad extinction line:', line)
            else:
                print(' >>>>> ERROR:', defaults_dict['color control filename'],
                      'bad extinction line:', line)
            extinction_dict[key] = value
        else:
            print(' >>>>> ERROR:', defaults_dict['color control filename'],
                  'bad extinction line:', line)
    color_control_dict['extinctions'] = extinction_dict
    return color_control_dict




