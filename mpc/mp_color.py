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
# ********** NO PHOTRIX IMPORTS ALLOWED ************* in this module mp_color.py.
# from photrix.image import Image, FITS

# From EVD package astropak (preferred):
import astropak.ini
import astropak.stats
import astropak.util
import astropak.image

# MPC_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BOOT_INI_FILENAME = 'defaults.ini'

# For color handling:
AVAILABLE_CATALOG_COLOR_INDEX_PASSBANDS = ('SG', 'SR', 'SI')
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

MP_COLOR_TOP_DIRECTORY = 'C:/Astro/MP Color/'


_____INITIALIZE_COLOR_WORKFLOW______________________________ = 0
# Largely implemented as calls to functions in module 'mp_phot'.


def start(color_top_directory=MP_COLOR_TOP_DIRECTORY, mp_number=None, an_string=None):
    """ Preliminaries to begin MP COLOR workflow.
        Wrapper to call mp_phot.start() with correct color parms.
    :param mp_number: number of target MP, e.g., 1602 for Indiana. [integer or string].
    :param an_string: Astronight string representation, e.g., '20191106' [string].
    :return: [None]
    """
    defaults_dict = mpc.ini.make_defaults_dict()
    color_log_filename = defaults_dict['color log filename']
    mpc.mp_phot.start(color_top_directory, mp_number, an_string, color_log_filename)


def resume(color_top_directory=MP_COLOR_TOP_DIRECTORY, mp_number=None, an_string=None):
    """ Restart a workflow in its correct working directory; keep log file--DO NOT overwrite it.
        Implemented as: wrapper to call mp_phot.resume() with correct color parms.
        Parameters as for start().
    :return: [None]
    """
    mpc.mp_phot.resume(color_top_directory, mp_number, an_string)


def assess():
    """ Assess FITS files in target directory, then write color control ini file and its template.
        Implemented as: wrapper around mp_phot.assess() + control file writing.
    :return: [None]
    """
    defaults_dict = mpc.ini.make_defaults_dict()
    color_log_filename = defaults_dict['color log filename']
    mpc.mp_phot.assess(color_log_filename, write_mp_phot_control_stub=False)

    # Write color control files (ini and its template):
    context = mpc.mp_phot.get_context(color_log_filename)
    if context is None:
        return
    this_directory, mp_string, an_string = context
    log_file = open(color_log_filename, mode='a')  # set up append to log file.

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


def make_dfs():
    """ For one MP on one night (color): make the 3 required dataframes.
        Implemented as wrapper around mp_phot.make_dfs()
    :return: [None]
    """
    defaults_dict = mpc.ini.make_defaults_dict()
    color_log_filename = defaults_dict['color log filename']
    context = mpc.mp_phot.get_context(color_log_filename)
    if context is None:
        return
    this_directory, mp_string, an_string = context
    color_control_dict = make_color_control_dict(this_directory, defaults_dict)
    mpc.mp_phot.make_dfs(color_control_dict)


_____COLOR_INDEX_PHOTOMETRY________________________________________________ = 0


# def do_color_SRI():
#     """ Convenience wrapper function. """
#     do_one_color(filters=('SR', 'SI'), result_color_index=('SR', 'SI'), catalog_color_index=('SR', 'SI'))
#
#
# def do_color_SRI_from_johnson():
#     """ Convenience wrapper function. """
#     do_one_color(filters=('R', 'I'), result_color_index=('SR', 'SI'), catalog_color_index=('SR', 'SI'))


def do_one_color(definition=None, catalog_color_index=None):
    """ Estimate ONE MP Color Index from images in TWO FILTERS.
        First run the 3 initial mp_phot functions (start(), assess(), make_dfs()) to make the dataframes.
        Then run this to get color index.
        ALL PARAMETERS MANDATORY.
    :param definition: a 4-tuple (filter_a, filter_b, passband_a, passband_b), e.g., ('R', 'I', 'SR', 'SI').
        Filters must have usable images in df_all.
        Passbands must have transforms to one of the filters, using the catalog_color_index, that is,
            in the above example definition, the transforms (R->SR, catalog_color_index) and
            (I->SI, catalog_color_index) must exist in the defaults_dict.
    :param catalog_color_index: specify the two passbands defining the color index, e.g., ('SR', 'SI') to
        define color in Sloan SR and SI (typical case). In any case, the two passbands must be
        represented in the ATLAS refcat2 catalog (prob. as 'r' and 'i') and must be included
        in the local catalog object. [2-tuple of strings] MANDATORY: no default.
    :return: None. Only writes to screen and to log file, makes diagnostic plots.
    """
    # ===== Get context, write log file header and color control stub:
    defaults_dict = mpc.ini.make_defaults_dict()
    color_log_filename = defaults_dict['color log filename']
    context = mpc.mp_phot.get_context(color_log_filename)
    if context is None:
        return
    this_directory, mp_string, an_string = context
    color_control_dict = make_color_control_dict(this_directory, defaults_dict)
    instrument_dict = mpc.ini.make_instrument_dict(defaults_dict)
    log_file = open(defaults_dict['color log filename'], mode='a')  # set up append to log file.
    log_file.write('\n===== do_color()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    # ===== Verify parms; verify that required filters and transforms exist:
    if not _verify_input_parms(definition, catalog_color_index, instrument_dict, color_control_dict):
        return
    fa, fb, result_a, result_b = definition     # convenience variables.
    catalog_a, catalog_b = catalog_color_index  # "
    print(' >>>>> do_one_color(): to get result color', result_a + '-' + result_b,
          'from filters', fa, '&', fb, 'using catalog color', catalog_a + '-' + catalog_b)

    # ===== Load and prepare session's master dataframe, and rename passband columns, e.g. 'r' -> 'SR':
    df_all = mpc.mp_phot.make_df_all(filters_to_include=(fa, fb), comps_only=False,
                                     require_mp_obs_each_image=True)
    df_all = df_all.rename(columns={'g': 'SG',   'r': 'SR',   'i': 'SI',   'z': 'SZ',
                                    'dg': 'dSG', 'dr': 'dSR', 'di': 'dSI', 'dz': 'dSZ'})

    # Verify before all obs screenings:
    if not _verify_images_available(df_all, definition):
        return
    if not _verify_catalog_color_available(df_all, definition):
        return

    df_screened_obs = _remove_images_on_user_request(df_all, color_control_dict['omit images'])
    df_screened_obs = _remove_comps_on_user_request(df_screened_obs, color_control_dict['omit comps'])
    df_screened_obs = _remove_obs_with_unusuable_instmag_instmagsigma(df_screened_obs)
    df_screened_obs = _keep_images_with_one_mp_obs(df_screened_obs)
    df_screened_obs = _keep_images_with_low_mp_obs_uncertainty(df_screened_obs,
                                                        color_control_dict['max mp obs mag uncertainty'])
    df_screened_obs = _remove_comps_obs_with_high_obs_uncertainty(df_screened_obs,
                                                        color_control_dict['max comp obs mag uncertainty'])
    df_screened_obs = _apply_selections_to_comp_obs(df_screened_obs, color_control_dict)
    df_screened_obs = _remove_images_with_few_comps(df_screened_obs)
    df_screened_obs = _remove_comps_absent_from_any_image(df_screened_obs)
    df_screened_obs = _remove_comp_obs_on_user_request(df_screened_obs, color_control_dict['omit obs'])
    df_screened_obs = _remove_images_with_too_few_comps(df_screened_obs,
                                                        color_control_dict['min valid comps per image'])
    df_screened_obs = df_screened_obs.sort_values(by=['JD_mid', 'Type', 'SourceID'])

    # Verify final df_screened_obs:
    if not _verify_images_available(df_all, definition):
        return
    if not _verify_catalog_color_available(df_all, definition):
        return

    # For each filter, build and run model, then get no-CI MP mag for each image:
    df_results = None # accumulator for final mag results, both filters.
    for (f, pb) in [(fa, result_a), (fb, result_b)]:
        df_model_this_filter = df_screened_obs.loc[df_screened_obs['Filter'] == f, :].copy()
        # Here, count MP obs in this filter, error if not at least one.
        model = SessionModel_OneFilter(df_model_this_filter, f, pb, catalog_color_index, color_control_dict,
                                       defaults_dict)
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
    # print(output + '\n   Write this to ' + MP_PHOT_CONTROL_FILENAME + ': #MP_RI_COLOR  ' +
    #       '{0:.3f}'.format(mp_color))
    # log_file.write(output)
    # log_file.close()


class SessionModel_OneFilter:
    """ Applies to *one* filter in one session.
        NOTE 20201021: For now we use only 1st-order (linear in Color Index) transforms.
        [2nd-order (quadratic in CI) will be too hard to backsolve, and probably unstable.]
    """
    def __init__(self, df_model, filter, passband, color_index_passbands, color_control_dict,
                 defaults_dict):
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
        self.defaults_dict = defaults_dict.copy()

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
            print(' >>>>> ERROR: SessionModel_OneFilter(', filter, passband, 'was passed no images.')
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
        color_log_filename = self.defaults_dict['color log filename']
        mpc.mp_phot.write_text_file(filename,
                                    'Mixed-model Regression for directory ' +
                                    mpc.mp_phot.get_context(color_log_filename)[0] + '\n\n' +
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


def _verify_input_parms(definition, catalog_color_index, instrument_dict, color_control_dict):
    """ Verify that parms passed in are correct in: number, form, and valid content.
        PASSES TESTS 2020-11-12 written for *ONE* COLOR ONLY.
    :param definition: a tuple of 4-tuples (filter_a, filter_b, passband_a, passband_b),
               e.g., (('R', 'I', 'SR', 'SI'), ...).
               A bare 4-tuple of strings for one color is OK too.
    :param catalog_color_index: passbands defining the color index (of comps). [tuple of strings]
    :param instrument_dict: instrument dict appropriate to this color session. [py dict]
    :param color_control_dict: color control dictionary for this color session. [py dict]
    :return: True iff parms entirely OK, else False. [boolean]
    """
    # TODO: extend testing to TWO COLOR definition.
    # Quick screen for types:
    if not all([isinstance(definition, tuple),
                isinstance(catalog_color_index, tuple),
                isinstance(color_control_dict, dict),
                isinstance(instrument_dict, dict),
                (len(catalog_color_index) == 2)]):
        print(' >>>>> ERROR: _verify_input_parms() found at least one parm with wrong type.')
        return False

    # Wrap definition in outer tuple if a bare 4-tuple of strings (for one color):
    if len(definition) == 4:
        if all([isinstance(x, str) for x in definition]):
            definition = (definition,)  # definition must be tuple of tuples (nested).

    # Parm 'definition': verify syntax:
    definition_ok = True  # to be falsified by any error.
    if not len(definition) in (1, 2):  # only 1 or 2 result colors supported.
        definition_ok = False
    for d in definition:
        if not len(d) == 4:
            definition_ok = False
            break
        if not all([isinstance(dd, str) for dd in d]):
            definition_ok = False
            break
    if not definition_ok:
        print(' >>>>> ERROR: definition has invalid syntax.')
        return False

    # Parm 'definition': verify filters available in instrument_dict:
    absent_filters = set()
    for d in definition:
        for f in d[:2]:
            if f not in instrument_dict['available filters']:
                absent_filters.add(f)
    if not len(absent_filters) == 0:
        print(' >>>>> ERROR: filters absent from instrument_dict:', str(list(absent_filters)))
        return False

    # Parm 'definition': verify all data available for transforms in color control file:
    definition_ok = True
    for f_a, f_b, pb_a, pb_b in definition:
        if instrument_dict['transforms'].get((f_a, pb_a,
                                              catalog_color_index[0], catalog_color_index[1])) is None:
            definition_ok = False
        if instrument_dict['transforms'].get((f_b, pb_b,
                                              catalog_color_index[0], catalog_color_index[1])) is None:
            definition_ok = False
            break
    if not definition_ok:
        print(' >>>>> ERROR: definition requires a transform that is missing from color control file.')
        return False

    # Parm 'definition': verify extinctions available for filters in color control file:
    extinction_ok = True
    for f_a, f_b, pb_a, pb_b in definition:
        for f in (f_a, f_b):
            extinction = color_control_dict['extinctions'].get(f)
            if extinction is None:
                extinction_ok = False
                break
    if not extinction_ok:
        print(' >>>>> ERROR: definition requires an extinction that is missing from color control file.')
        return False

    # Parm 'catalog_color_index': verify available in catalog:
    cat_ci_ok = (len(catalog_color_index) == 2) and\
        all([pb in AVAILABLE_CATALOG_COLOR_INDEX_PASSBANDS for pb in catalog_color_index])
    if not cat_ci_ok:
        print(' >>>>> ERROR: catalog CI passbands invalid (must be 2, both in catalog).')
        return False

    return True


def _verify_images_available(df_in, definition):
    """ Return True iff at least one credible image available for every filter in this color session.
    :param df_in: master dataframe of all comp and MP observations. [pandas dataframe]
    :param definition: a 4-tuple (filter_a, filter_b, passband_a, passband_b), e.g., ('R', 'I', 'SR', 'SI').
    :return: True iff >1 image in each filter available in this color session. [boolean]
    """
    f_a, f_b, _, _ = definition
    all_present = True  # falsify on absence detected.
    for f in [f_a, f_b]:
        mp_present = any((df_in['Filter'] == f) & (df_in['Type'] == 'MP'))
        comp_present = any((df_in['Filter'] == f) & (df_in['Type'] == 'Comp'))
        if not (mp_present and comp_present):
            all_present = False
            print(' >>>>> ERROR: no image with filter', f + '.')
    return all_present


def _verify_catalog_color_available(df_in, definition):
    """ Return True iff df has columns comprising definition's required color (e.g., columns 'SR' & 'SI'),
        that is, both data for both passbands comprising the color defined in definition tuple.
    :param df_in: master dataframe of all comp and MP observations. [pandas dataframe]
    :param definition: a 4-tuple (filter_a, filter_b, passband_a, passband_b), e.g., ('R', 'I', 'SR', 'SI').
    :return: True iff >1 both color index columns exist and not all values are NA/None. [boolean]
    """
    _, _, pb_a, pb_b = definition
    all_present = True  # falsify on absence detected.
    for pb in [pb_a, pb_b]:
        if not (pb in df_in.columns):
            all_present = False
            print(' >>>>> ERROR: column', pb, 'absent from master dataframe.')
        else:
            count_data = sum([not np.isnan(x) for x in df_in.loc[:, pb]])
            if count_data < len(df_in) / 2:
                all_present = False
                print(' >>>>> ERROR: most data invalid for column', pb, 'in master dataframe.')
    return all_present


_____SCREENING_FUNCTIONS____________________________________ = 0


def _remove_images_on_user_request(df_in, images_to_omit):
    """ Remove all observations from dataframe that have FITSfile that control file instructs to omit.
    :param df_in: master dataframe of observations. [pandas dataframe]
    :param images_to_omit: images [FITS file names] for which all observations are to be removed. [tuple]
    :return: updated dataframe. [pandas dataframe]
    """
    if len(images_to_omit) <= 0:
        return df_in
    df_screened_obs = df_in.copy()
    obs_to_remove = (df_screened_obs['FITSfile'].isin(images_to_omit))
    if any(obs_to_remove):
        obs_to_keep = (~ obs_to_remove).to_list()
        df_screened_obs = df_screened_obs.loc[obs_to_keep, :]
    return df_screened_obs


def _remove_comps_on_user_request(df_in, comps_to_omit):
    """ Remove all comp observations from dataframe that have comp IDs that control file instructs to omit.
    :param df_in: screened master dataframe of observations. [pandas dataframe]
    :param comps_to_omit: comp IDs for which all observations are to be removed. [tuple]
    :return: updated dataframe. [pandas dataframe]
    """
    if len(comps_to_omit) <= 0:
        return df_in
    df_screened_obs = df_in.copy()
    obs_to_remove = df_screened_obs['SourceID'].isin(comps_to_omit)
    is_comp = (df_screened_obs['Type'] == 'Comp')
    obs_to_remove = (obs_to_remove & is_comp)
    if any(obs_to_remove):
        obs_to_keep = (~ obs_to_remove).to_list()
        df_screened_obs = df_screened_obs.loc[obs_to_keep, :]
    return df_screened_obs


def _remove_obs_with_unusuable_instmag_instmagsigma(df_in):
    """ Remove from dataframe any observations, comp or MP, with None, NaN, etc as instrument magnitude
            or inst. mag. sigma.
    :param df_in: screened master dataframe of observations. [pandas dataframe]
    :return: updated dataframe. [pandas dataframe]
    """
    return df_in.dropna(subset=['InstMag', 'InstMagSigma'])


def _keep_images_with_one_mp_obs(df_in):
    """ Retain in dataframe only the observations of those images with exactly one MP observation.
    :param df_in: screened master dataframe of observations. [pandas dataframe]
    :return: updated dataframe. [pandas dataframe]
    """
    mp_obs_count_by_filename = df_in.loc[df_in['Type'] == 'MP', 'FITSfile'].value_counts()
    filenames_with_one_mp = [fn for fn in mp_obs_count_by_filename.index
                             if mp_obs_count_by_filename[fn] == 1]
    obs_to_keep = df_in['FITSfile'].isin(filenames_with_one_mp).to_list()
    if all(obs_to_keep):
        return df_in
    return df_in.loc[obs_to_keep, :].copy()


def _keep_images_with_low_mp_obs_uncertainty(df_in, max_mp_obs_mag_uncertainty):
    """ Retain in dataframe only the observations whose MP observation uncertainty is sufficiently low.
    :param df_in: screened master dataframe of observations. [pandas dataframe]
    :param max_mp_obs_mag_uncertainty: upper allowed limit of MP observation magnitude uncertainty. [float]
    :return: updated dataframe. [pandas dataframe]
    """
    is_mp_obs = (df_in['Type'] == 'MP')
    obs_uncert_ok = (df_in['InstMagSigma'] <= max_mp_obs_mag_uncertainty)
    row_is_good_mp = (is_mp_obs & obs_uncert_ok).to_list()
    filenames_to_keep = df_in.loc[row_is_good_mp, 'FITSfile']
    obs_to_keep = df_in['FITSfile'].isin(filenames_to_keep).to_list()
    if all(obs_to_keep):
        return df_in
    return df_in.loc[obs_to_keep, :].copy()


def _remove_comps_obs_with_high_obs_uncertainty(df_in, max_comp_obs_mag_uncertainty):
    """ Retain in dataframe only the comp observations whose magnitude uncertainty is sufficiently low.
    :param df_in: screened master dataframe of observations. [pandas dataframe]
    :param max_comp_obs_mag_uncertainty: upper allowed limit of comp obs magnitude uncertainty. [float]
    :return: updated dataframe. [pandas dataframe]
    """
    # TODO: ? make max uncert relative to instrumental mag (to more accurately remove true outliers) ?
    is_comp_obs = df_in['Type'] == 'Comp'
    uncert_too_high = df_in['InstMagSigma'] > max_comp_obs_mag_uncertainty
    obs_to_remove = (is_comp_obs & uncert_too_high)
    if any(obs_to_remove):
        obs_to_keep = (~ obs_to_remove).to_list()
        return df_in.loc[obs_to_keep, :].copy()
    return df_in


def _apply_selections_to_comp_obs(df_in, color_control_dict):
    """ Split df_in to separate comp-obs and MP-obs dataframes; apply color_control selections to the comps,
        recombine and return the recombined dataframe of MPs and selected comp obs.
        All selection criteria handled *except* 'omit obs' which is handled below (later).
    :param df_in: screened master dataframe of observations. [pandas dataframe]
    :param color_control_dict: user options from color control ini file. [python dict]
    :return: updated dataframe. [pandas dataframe]
    """
    df_in_comp_obs = df_in.loc[df_in['Type'] == 'Comp', :].copy()
    df_in_mp_obs = df_in.loc[df_in['Type'] == 'MP', :].copy()

    # Apply selection criteria to comp obs only (MP obs don't have catalog values):
    remove_by_min_mag = (df_in_comp_obs['SR'] < color_control_dict['min sr mag'])
    remove_by_max_mag = (df_in_comp_obs['SR'] > color_control_dict['max sr mag'])
    remove_by_cat_mag_uncert = (df_in_comp_obs['dSR'] > color_control_dict['max catalog dsr mmag'])
    color = (df_in_comp_obs['SR'] - df_in_comp_obs['SI'])
    remove_by_min_color = (color < color_control_dict['min sloan ri color'])
    remove_by_max_color = (color > color_control_dict['max sloan ri color'])
    obs_to_remove = (remove_by_min_mag | remove_by_max_mag | remove_by_cat_mag_uncert |
                     remove_by_min_color | remove_by_max_color)
    if any(obs_to_remove):
        obs_to_keep = (~ obs_to_remove).to_list()
        df_screened_comp_obs = df_in_comp_obs.loc[obs_to_keep, :]
        df_screened = df_screened_comp_obs.append(df_in_mp_obs)  # if any comp obs were removed.
        return df_screened.sort_values(by=['JD_mid', 'Type'])
    return df_in  # if no comp obs were removed.


def _remove_images_with_few_comps(df_in):
    """ Remove all observations from images which have fewer comps than half the maximum number of comps.
        (Remove obviously bad images.)
    :param df_in:screened master dataframe of observations. [pandas dataframe]
    :return:updated dataframe. [pandas dataframe]
    """
    comp_counts = df_in.loc[df_in['Type'] == 'Comp', 'FITSfile'].value_counts()
    max_comp_count = comp_counts.max()
    filenames_too_few_comps = [fn for fn in comp_counts.index if comp_counts[fn] < max_comp_count / 2.0]
    if len(filenames_too_few_comps) >= 1:
        remove_too_few_comps = df_in['FITSfile'].isin(filenames_too_few_comps)
        obs_to_keep = (~ remove_too_few_comps).to_list()
        return df_in.loc[obs_to_keep, :].copy()
    return df_in


def _remove_comps_absent_from_any_image(df_in):
    """ Remove all observations of any comp star that is not present in *every* image.
        After running this, all images will have obs from *exactly* the same comps (though the user's
        'omit obs' selection criteria may later remove some, and this has no effect on that.)
    :param df_in: screened master dataframe of observations. [pandas dataframe]
    :return: updated dataframe. [pandas dataframe]
    """
    image_count = len(df_in.loc[df_in['Type'] == 'Comp', 'FITSfile'].drop_duplicates())
    image_count_per_comp = df_in.loc[df_in['Type'] == 'Comp', 'SourceID'].value_counts()
    comps_not_in_every_image = [id for id in image_count_per_comp.index
                                if image_count_per_comp[id] < image_count]
    if len(comps_not_in_every_image) >= 1:
        obs_to_remove = df_in['SourceID'].isin(comps_not_in_every_image)
        obs_to_keep = (~ obs_to_remove).to_list()
        return df_in.loc[obs_to_keep, :].copy()
    return df_in


def _remove_comp_obs_on_user_request(df_in, omit_obs_criteria):
    """ Remove (comp) observations as specified by 'omit obs' element of user's color control ini file.
    :param df_in: screened master dataframe of observations. [pandas dataframe]
    :param omit_obs_criteria: observation IDs to remove, as specified by user. [tuple of strings]
    :return: updated dataframe. [pandas dataframe]
    """
    remove_by_obs_id = df_in['Serial'].isin(omit_obs_criteria)
    if any(remove_by_obs_id):
        obs_to_keep = (~ remove_by_obs_id).to_list()
        return df_in.loc[obs_to_keep, :].copy()
    return df_in


def _remove_images_with_too_few_comps(df_in, min_valid_comps_per_image):
    """ Remove (with console warning) all obs from any images with fewer than a minimum number of comps.
        Rarely changes df, especially after all the above screens unless there are very many
        observations removed by the user's 'omit obs' selection criteria.
    :param df_in: screened master dataframe of observations. [pandas dataframe]
    :param min_valid_comps_per_image: the minimum number of valid comp obs per image. [int]
    :return: updated dataframe. [pandas dataframe]
    """
    comp_count_per_image = df_in.loc[df_in['Type'] == 'Comp', 'FITSfile'].value_counts()
    images_too_few_comps = [fn for fn in comp_count_per_image.index
                            if comp_count_per_image[fn] < min_valid_comps_per_image]
    if len(images_too_few_comps) >= 1:
        remove_too_few_comps = df_in['FITSfile'].isin(images_too_few_comps)
        obs_to_keep = (~ remove_too_few_comps).to_list()
        return df_in.loc[obs_to_keep, :].copy()
    return df_in


_____CONTROL_INI_DICT_FUNCTIONS_____________________________ = 0


def _write_color_control_ini_template(this_directory, defaults_dict):
    """ Write color control ini template to current session directory.
    :param this_directory: fullpath of current session directory. [string]
    :param defaults_dict: dict of defaults [py dict]
    :return: [None]
    """
    color_control_template_filename = defaults_dict['color control template filename']
    fullpath = os.path.join(this_directory, color_control_template_filename)

    lines = ['#----- This is template file ' + color_control_template_filename,
             '#----- in directory ' + this_directory,
             '#',
             '[MP Location]',
             'MP Location Early = string -> mp location early',
             'MP Location Late  = string -> mp location late',
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
    :return:
    """
    color_control_filename = defaults_dict['color control filename']
    fullpath = os.path.join(this_directory, color_control_filename)

    fits_filenames = mpc.mp_phot.get_fits_filenames(this_directory)  # TODO: move this fn to util.py.
    utc_mid_list = [astropak.image.FITS(this_directory, '', fn).utc_mid for fn in fits_filenames]
    mp_filename_earliest = fits_filenames[np.argmin(utc_mid_list)]
    mp_filename_latest = fits_filenames[np.argmax(utc_mid_list)]

    lines = ['#----- This is ' + color_control_filename + ' for directory:',
             '#-----    ' + this_directory,
             '#',
             '[Ini Template]',
             'Filename = ' + defaults_dict['color control template filename'],
             '',
             '[MP Location]',
             '# for each, give: FITS_filename  x_pixel  y_pixel.',
             'MP Location Early = ' + mp_filename_earliest,
             'MP Location Late  = ' + mp_filename_latest,
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

    # Parse and overwrite MP Location items:
    early_items = color_control_dict['mp location early'].split()
    late_items = color_control_dict['mp location late'].split()
    color_control_dict['mp location filenames'] = (early_items[0], late_items[0])
    color_control_dict['x pixels'] = (float(early_items[1]), float(late_items[1]))
    color_control_dict['y pixels'] = (float(early_items[2]), float(late_items[2]))
    del color_control_dict['mp location early'], color_control_dict['mp location late']

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




