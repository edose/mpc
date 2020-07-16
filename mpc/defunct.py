__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"# _____TRANSFORM_DETERMINATION_______________________________________________ = 0
#
# def calc_transform(f='Clear', pbf='r', pb1='r', pb2='i'):
#     """ From one image in filter f, get transform for filter f-->catalog passband pbf,
#             subject to color index=cat_pb1 - cat_pb2.
#         Must have already run start() or resume() to set working directory.
#         Must have already run through make_dfs() for this directory.
#         Will use all FITS file images taken in filter
#     :param f: name of filter in which FITS file image was taken, and defining the transform. [string]
#     :param pbf: name of catalog passband to which to transform magnitudes
#         (associated with filter). [string]
#     :param pb1: first catalog passband of color index. [string]
#     :param pb2: second catalog passband of color index. [string]
#     :return: None. Writes results to console and summary to log file.
#     """
#     context = get_context()
#     if context is None:
#         return
#     this_directory, mp_string, an_string = context
#     log_file = open(LOG_FILENAME, mode='a')  # set up append to log file.
#     mp_int = int(mp_string)  # put this in try/catch block.
#     mp_string = str(mp_int)
#     state = get_session_state()  # for extinction and transform values.
#
#     # Set up required data:
#     df_transform = make_df_all(filters_to_include=f, comps_only=True, require_mp_obs_each_image=False)
#     user_selections = read_selection_criteria(TRANSFORM_CONTROL_FILENAME,
#         TRANSFORM_COMP_SELECTION_DEFAULTS)
#     apply_calc_transform_selections(df_transform, user_selections)  # adds boolean column 'UseInModel'.
#     options_dict = read_regression_options(TRANSFORM_CONTROL_FILENAME)
#
#     # Perform mixed-model regression:
#     model = TransformModel(df_transform, f, pbf, pb1, pb2, state, options_dict)
#
#     # Make diagnostic plots:
#     make_transform_diagnostic_plots(model, df_transform, f, pbf, pb1, pb2, state, user_selections)
#
#     # Write results to log file and to console:
#     print('Transform=', '{0:.4f}'.format(model.mm_fit.df_fixed_effects.loc['CI', 'Value']),
#           '  tr_sigma=', '{0:.4f}'.format(model.mm_fit.df_fixed_effects.loc['CI', 'Stdev']),
#           '    mag_sigma=', '{0:.1f}'.format(1000.0 * model.mm_fit.sigma), 'mMag.')
#
#
# class TransformModel:
#     def __init__(self, df_transform, f, pbf, pb1, pb2, state, options_dict):
#         """  Makes and holds color-transform model via mixed-model regression.
#              Requires data from at least 3 images in f.
#         :param df_transform: table of data from which to draw data for regression. [pandas Dataframe]
#         :param f: name of filter in which FITS file image was taken, and defining the transform. [string]
#         :param pbf: name of catalog passband to which to transform mags (associated with filter). [string]
#         :param pb1: first catalog passband of color index. [string]
#         :param pb2: second catalog passband of color index. [string]
#         :param state:
#         :param options_dict: holds options for making comp fit. [pandas dict object]
#         """
#         self.df_used = df_transform.copy().loc[df_transform['UseInModel'], :]  # only obs used in model.
#         n_images = len(self.df_used['FITSfile'].drop_duplicates())
#         self.enough_images = (n_images >= 3)
#         if not self.enough_images:
#             return
#         self.f = f
#         self.pbf, self.pb1, self.pb2 = pbf, pb1, pb2
#         self.state = state
#         self.fit_extinction = options_dict.get('fit_extinction', False)
#         self.fit_vignette = options_dict.get('fit_vignette', True)
#         self.fit_xy = options_dict.get('fit_xy', False)
#         self.fit_jd = options_dict.get('fit_jd', True)
#
#         self.dep_var_name = 'InstMag_with_offsets'
#         self.mm_fit = None      # placeholder for the fit result [photrix MixedModelFit object].
#         self.extinction = None  # "
#         self.vignette = None    # "
#         self.x = None           # "
#         self.y = None           # "
#         self.jd1 = None         # "
#         self.transform = None
#         self.transform_sigma = None
#
#         self._prep_and_do_regression()
#
#     def _prep_and_do_regression(self):
#         """ Using photrix.util.MixedModelFit class (which wraps statsmodels.MixedLM.from_formula() etc).
#             This function uses comp data only (no minor planet data).
#         :return: [None] Puts model into self.mm_fit.
#         """
#         if not self.enough_images:
#             return
#         self.df_used['CI'] = self.df_used[self.pb1] - self.df_used[self.pb2]
#
#         # Initiate dependent-variable offset, which will aggregate all such offset terms:
#         dep_var_offset = self.df_used[self.pbf].copy()  # *copy* CatMag, or it will be damaged
#
#         # Build fixed-effect (x) variable list and construct dep-var offset:
#         fixed_effect_var_list = ['CI']
#         if self.fit_extinction:
#             fixed_effect_var_list.append('Airmass')
#         else:
#             extinction = self.state['extinction']['Clear']
#             dep_var_offset += extinction * self.df_used['Airmass']
#             print(' Extinction (Airmass) not fit: value fixed at',
#                   '{0:.3f}'.format(extinction))
#         if self.fit_vignette:
#             fixed_effect_var_list.append('Vignette')
#         if self.fit_xy:
#             fixed_effect_var_list.extend(['X1024', 'Y1024'])
#         if self.fit_jd:
#             fixed_effect_var_list.append('JD_fract')
#
#         # Build 'random-effect' variable:
#         random_effect_var_name = 'FITSfile'  # cirrus effect is per-image
#
#         # Build dependent (y) variable:
#         self.df_used[self.dep_var_name] = self.df_used['InstMag'] - dep_var_offset
#
#         # Execute regression:
#         import warnings
#         from statsmodels.tools.sm_exceptions import ConvergenceWarning
#         warnings.simplefilter('ignore', ConvergenceWarning)
#         self.mm_fit = MixedModelFit(data=self.df_used,
#                                     dep_var=self.dep_var_name,
#                                     fixed_vars=fixed_effect_var_list,
#                                     group_var=random_effect_var_name)
#         if not self.mm_fit.converged:
#             print(' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.')
#             print(self.mm_fit.statsmodels_object.summary())
#             print(' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.')
#         else:
#             # TODO: fix these
#             self.transform = 9999.
#             self.transform_sigma = 9999.
#
#
# def make_transform_diagnostic_plots(model, df_model, f, pbf, pb1, pb2, state, user_selections):
#     """  Display and write to file several diagnostic plots, to help decide which obs, comps, images
#          might need removal by editing control file.
#     :param model: mixed model summary object. [photrix.MixedModelFit object]
#     :param df_model: dataframe of all data including UseInModel
#         (user selection) column. [pandas DataFrame]
#     :param f: name of filter in which observations were made. [string]
#     :param pbf: name of target passband (associated with filter) in which mags to be reported. [string]
#     :param pb1: name of first of two catalog passbands in color index. [string]
#     :param pb2: name of second of two catalog passbands in color index. [string]
#     :param state: session state for this observing session [dict]
#     :param user_selections: comp selection criteria, used for drawing limits on plots [python dict]
#     :return: [None] Writes image files e.g., Transform_Image1_QQ_comps.png
#     """
#     this_directory, mp_string, an_string = get_context()
#
#     # Delete any previous transform image files from current directory:
#     image_filenames = [fn for fn in os.listdir('.')
#                        if fn.startswith(TRANSFORM_IMAGE_PREFIX) and fn.endswith('.png')]
#     for fn in image_filenames:
#         os.remove(fn)
#
#     # Wrangle needed data into convenient forms:
#     df_plot = pd.merge(left=df_model.loc[df_model['UseInModel'], :].copy(),
#                        right=model.mm_fit.df_observations,
#                        how='left', left_index=True, right_index=True, sort=False)  # add col 'Residuals'.
#     is_comp_obs = (df_plot['Type'] == 'Comp')
#     df_plot_comp_obs = df_plot.loc[is_comp_obs, :]
#     df_plot_mp_obs = df_plot.loc[(~ is_comp_obs), :]
#     df_image_effect = model.mm_fit.df_random_effects
#     df_image_effect.rename(columns={"GroupName": "FITSfile", "Group": "ImageEffect"}, inplace=True)
#     intercept = model.mm_fit.df_fixed_effects.loc['Intercept', 'Value']
#     # jd_slope = model.mm_fit.df_fixed_effects.loc['JD_fract', 'Value']  # undefined if FIT_JD is False.
#     sigma = model.mm_fit.sigma
#     if 'Airmass' in model.mm_fit.df_fixed_effects.index:
#         extinction = model.mm_fit.df_fixed_effects.loc['Airmass', 'Value']  # if fit in model
#     else:
#         extinction = state['extinction']['Clear']  # default if not fit in model (normal case)
#     # if 'CI' in model.mm_fit.df_fixed_effects.index:
#     #     transform = model.mm_fit.df_fixed_effects.loc['CI', 'Value']  # if fit in model
#     # else:
#     #     transform = TRANSFORM_CLEAR_SR_SR_SI  # default if not fit in model (normal case)
#     if model.fit_jd:
#         jd_coefficient = model.mm_fit.df_fixed_effects.loc['JD_fract', 'Value']
#     else:
#         jd_coefficient = 0.0
#     comp_ids = df_plot_comp_obs['SourceID'].drop_duplicates()
#     n_comps = len(comp_ids)
#     comp_color, mp_color = 'dimgray', 'orangered'
#     obs_colors = [comp_color if i is True else mp_color for i in is_comp_obs]
#     jd_floor = floor(min(df_model['JD_mid']))
#     obs_jd_fract = df_plot['JD_mid'] - jd_floor
#     xlabel_jd = 'JD(mid)-' + str(jd_floor)
#     transform_string = f + ARROW_CHARACTER + pbf + ' (' + pb1 + '-' + pb2 + ')'
#
#     # ################ TRANSFORM FIGURE 1: Q-Q plot of mean comp effects
#     #    (1 pt per comp star used in model), code heavily adapted from photrix.process.SkyModel.plots():
#     window_title = 'Transform Q-Q by comp :: ' + transform_string +\
#         ' :: ' + '\\'.join(this_directory.split('\\')[-2:])
#     page_title = window_title
#     plot_annotation = str(n_comps) + ' comps used in model.\n(tags: comp SourceID)'
#     df_y = df_plot_comp_obs.loc[:, ['SourceID', 'Residual']].groupby(['SourceID']).mean()
#     df_y = df_y.sort_values(by='Residual')
#     y_data = df_y['Residual'] * 1000.0  # for millimags
#     y_labels = df_y.index.values
#     make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
#                           TRANSFORM_IMAGE_PREFIX + '1_QQ_comps.png')
#
#     # ################ TRANSFORM FIGURE 2: Q-Q plot of comp residuals
#     #    (1 pt per comp obs used in model), code heavily adapted from photrix.process.SkyModel.plots():
#     window_title = 'Transform Q-Q by obs :: ' + transform_string +\
#         ' :: ' + '\\'.join(this_directory.split('\\')[-2:])
#     page_title = window_title
#     plot_annotation = str(len(df_plot_comp_obs)) + ' observations of ' + \
#         str(n_comps) + ' comps used in model.\n (tags: observation Serial numbers)'
#     df_y = df_plot_comp_obs.loc[:, ['Serial', 'Residual']]
#     df_y = df_y.sort_values(by='Residual')
#     y_data = df_y['Residual'] * 1000.0  # for millimags
#     y_labels = df_y['Serial'].values
#     make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
#                           TRANSFORM_IMAGE_PREFIX + '2_QQ_obs.png')
#
#     # ################ TRANSFORM FIGURE 3: Catalog and Time plots:
#     fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(11, 8.5))  # (width, height) in "inches", was 15,9
#     fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
#     fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
#     main_title = 'Transform Catalog+Time :: ' + transform_string +\
#         ' :: ' + '\\'.join(this_directory.split('\\')[-2:])
#     window_title = main_title
#     fig.suptitle(main_title, color='darkblue', fontsize=20)
#     fig.canvas.set_window_title(window_title)
#     subplot_text = 'rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
#     fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=12, color='dimgray')
#
#     # Catalog mag uncertainty plot (comps only, one point per comp, x=cat r mag, y=cat r uncertainty):
#     ax = axes[0, 0]
#     make_9_subplot(ax, 'Catalog Mag Uncertainty (dr)', 'Catalog Mag (r)', 'mMag', '', False,
#                    x_data=df_plot_comp_obs['r'], y_data=df_plot_comp_obs['dr'])
#     # Catalog color plot (comps only, one point per comp, x=cat r mag, y=cat color (r-i)):
#     ax = axes[0, 1]
#     make_9_subplot(ax, 'Catalog Color Index', 'Catalog Mag (r)', 'CI Mag', '', zero_line=False,
#                    x_data=df_plot_comp_obs['r'], y_data=(df_plot_comp_obs['r'] - df_plot_comp_obs['i']))
#     # Inst Mag plot (comps only, one point per obs, x=cat r mag, y=InstMagSigma):
#     ax = axes[0, 2]
#     make_9_subplot(ax, 'Instrument Magnitude Uncertainty', 'Catalog Mag (r)', 'mMag', '', True,
#                    x_data=df_plot_comp_obs['r'], y_data=df_plot_comp_obs['InstMagSigma'])
#     # Cirrus plot (comps only, one point per image, x=JD_fract, y=Image Effect):
#     ax = axes[1, 0]
#     df_this_plot = pd.merge(df_image_effect, df_plot_comp_obs.loc[:, ['FITSfile', 'JD_fract']],
#                             how='left', on='FITSfile', sort=False).drop_duplicates()
#     make_9_subplot(ax, 'Image effect (cirrus plot)', xlabel_jd, 'mMag', '', False,
#                    x_data=df_this_plot['JD_fract'], y_data=1000.0 * df_this_plot['ImageEffect'],
#                    alpha=1.0, jd_locators=True)
#     # SkyADU plot (comps only, one point per obs: x=JD_fract, y=SkyADU):
#     ax = axes[1, 1]
#     make_9_subplot(ax, 'SkyADU vs time', xlabel_jd, 'ADU', '', False,
#                    x_data=df_plot_comp_obs['JD_fract'], y_data=df_plot_comp_obs['SkyADU'],
#                    jd_locators=True)
#     # FWHM plot (comps only, one point per obs: x=JD_fract, y=FWHM):
#     ax = axes[1, 2]
#     make_9_subplot(ax, 'FWHM vs time', xlabel_jd, 'FWHM (pixels)', '', False,
#                    x_data=df_plot_comp_obs['JD_fract'], y_data=df_plot_comp_obs['FWHM'],
#                    jd_locators=True)
#     # InstMagSigma plot (comps only, one point per obs; x=JD_fract, y=InstMagSigma):
#     ax = axes[2, 0]
#     make_9_subplot(ax, 'Inst Mag Sigma vs time', xlabel_jd, 'mMag', '', False,
#                    x_data=df_plot_comp_obs['JD_fract'], y_data=1000.0 * df_plot_comp_obs['InstMagSigma'],
#                    jd_locators=True)
#     # Airmass plot (comps only, one point per obs; x=JD_fract, y=Airmass):
#     ax = axes[2, 1]
#     make_9_subplot(ax, 'Airmass vs time', xlabel_jd, 'Airmass', '', False,
#                    x_data=df_plot_comp_obs['JD_fract'], y_data=df_plot_comp_obs['Airmass'],
#                    jd_locators=True)
#     axes[2, 2].remove()  # clear this space (no lightcurve subplot for transform).
#     plt.show()
#     fig.savefig(TRANSFORM_IMAGE_PREFIX + '3_Catalog_and_Time.png')
#
#     # ################ TRANSFORM FIGURE 4: Residual plots:
#     fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(11, 8.5))
#         (width, height) in "inches", was 15, 9
#     fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
#     fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
#     main_title = 'Transform Residuals :: ' + transform_string +\
#         ' :: ' + '\\'.join(this_directory.split('\\')[-2:])
#     window_title = main_title
#     fig.suptitle(main_title, color='darkblue', fontsize=20)
#     fig.canvas.set_window_title(window_title)
#     subplot_text = str(len(df_plot_comp_obs)) + ' obs   ' +\
#         str(n_comps) + ' comps    ' +\
#         'sigma=' + '{0:.0f}'.format(1000.0 * sigma) + ' mMag' +\
#         (12 * ' ') + ' rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
#     fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=12, color='dimgray')
#
#     # Comp residual plot (comps only, one point per obs: x=catalog r mag, y=model residual):
#     ax = axes[0, 0]
#     make_9_subplot(ax, 'Model residual vs r (catalog)', 'Catalog Mag (r)', 'mMag', '', True,
#                    x_data=df_plot_comp_obs['r'], y_data=1000.0 * df_plot_comp_obs['Residual'])
#     draw_x_line(ax, user_selections['min_r_mag'])
#     draw_x_line(ax, user_selections['max_r_mag'])
#
#     # Comp residual plot (comps only, one point per obs: x=raw Instrument Mag, y=model residual):
#     ax = axes[0, 1]
#     make_9_subplot(ax, 'Model residual vs raw Instrument Mag', 'Raw instrument mag', 'mMag', '', True,
#                    x_data=df_plot_comp_obs['InstMag'], y_data=1000.0 * df_plot_comp_obs['Residual'])
#
#     # Comp residual plot (comps only, one point per obs: x=catalog r-i color, y=model residual):
#     ax = axes[0, 2]
#     make_9_subplot(ax, 'Model residual vs Color Index (cat)', 'Catalog Color (r-i)', 'mMag', '', True,
#                    x_data=df_plot_comp_obs['r'] - df_plot_comp_obs['i'],
#                    y_data=1000.0 * df_plot_comp_obs['Residual'])
#
#     # Comp residual plot (comps only, one point per obs: x=Julian Date fraction, y=model residual):
#     ax = axes[1, 0]
#     make_9_subplot(ax, 'Model residual vs JD', xlabel_jd, 'mMag', '', True,
#                    x_data=df_plot_comp_obs['JD_fract'], y_data=1000.0 * df_plot_comp_obs['Residual'],
#                    jd_locators=True)
#
#     # Comp residual plot (comps only, one point per obs: x=Airmass, y=model residual):
#     ax = axes[1, 1]
#     make_9_subplot(ax, 'Model residual vs Airmass', 'Airmass', 'mMag', '', True,
#                    x_data=df_plot_comp_obs['Airmass'], y_data=1000.0 * df_plot_comp_obs['Residual'])
#
#     # Comp residual plot (comps only, one point per obs: x=Sky Flux (ADUs), y=model residual):
#     ax = axes[1, 2]
#     make_9_subplot(ax, 'Model residual vs Sky Flux', 'Sky Flux (ADU)', 'mMag', '', True,
#                    x_data=df_plot_comp_obs['SkyADU'], y_data=1000.0 * df_plot_comp_obs['Residual'])
#
#     # Comp residual plot (comps only, one point per obs: x=X in images, y=model residual):
#     ax = axes[2, 0]
#     make_9_subplot(ax, 'Model residual vs X in image', 'X from center (pixels)', 'mMag', '', True,
#                    x_data=df_plot_comp_obs['X1024'] * 1024.0,
#                    y_data=1000.0 * df_plot_comp_obs['Residual'])
#     draw_x_line(ax, 0.0)
#
#     # Comp residual plot (comps only, one point per obs: x=Y in images, y=model residual):
#     ax = axes[2, 1]
#     make_9_subplot(ax, 'Model residual vs Y in image', 'Y from center (pixels)', 'mMag', '', True,
#                    x_data=df_plot_comp_obs['Y1024'] * 1024.0,
#                    y_data=1000.0 * df_plot_comp_obs['Residual'])
#     draw_x_line(ax, 0.0)
#
#     # Comp residual plot (comps only, one point per obs: x=vignette (dist from center), y=model residual):
#     ax = axes[2, 2]
#     make_9_subplot(ax, 'Model residual vs distance from center', 'dist from center (pixels)', 'mMag',
#                    '', True,
#                    x_data=1024*np.sqrt(df_plot_comp_obs['Vignette']),
#                    y_data= 1000.0 * df_plot_comp_obs['Residual'])
#
#     plt.show()
#     fig.savefig(TRANSFORM_IMAGE_PREFIX + '4_Residuals.png')
#
#     # ################ TRANSFORM FIGURE(S) 5: Variability plots:
#     # Several comps on a subplot, vs JD, normalized by (minus) the mean of all other comps' responses.
#     # Make df_offsets (one row per obs, at first with only raw offsets):
#     trial_transform = model.mm_fit.df_fixed_effects.loc['CI', 'Value']
#     transform = trial_transform if abs(trial_transform < 0.2) else 0.0  # no crazy values.
#     make_comp_variability_plots(df_plot_comp_obs, xlabel_jd, transform, sigma,
#                                 image_prefix=TRANSFORM_IMAGE_PREFIX)



# def write_transform_control_txt_stub(this_directory, log_file):
#     defaults = TRANSFORM_COMP_SELECTION_DEFAULTS
#     lines = [';----- This is ' + TRANSFORM_CONTROL_FILENAME + ' for directory:\n;      ' + this_directory,
#              ';',
#              ';===== SELECTION CRITERIA BLOCK =====================================',
#              ';      Selection criteria for comp stars, observations, images:',
#              ';#COMP  nnnn nn,   nnn        ; to omit comp(s) by comp ID',
#              ';#OBS nnn,nnnn nnnn   nn      ; to omit observation(s) by Serial number',
#              ';#IMAGE  MP_mmmm-00nn-Clear   ; to omit one FITS image (.fts at end optional)',
#              (';#MIN_R_MAG ' + str(defaults['min_catalog_r_mag'])).ljust(30) +
#              '; default=' + str(defaults['min_catalog_r_mag']),
#              (';#MAX_R_MAG ' + str(defaults['max_catalog_r_mag'])).ljust(30) +
#              '; default=' + str(defaults['max_catalog_r_mag']),
#              (';#MAX_CATALOG_DR_MMAG ' + str(defaults['max_catalog_dr_mag'])).ljust(30) +
#              '; default=' + str(defaults['max_catalog_dr_mag']),
#              (';#MIN_SLOAN_RI_COLOR ' + str(defaults['min_catalog_ri_color'])).ljust(30) +
#              '; default=' + str(defaults['min_catalog_ri_color']),
#              (';#MAX_SLOAN_RI_COLOR ' + str(defaults['max_catalog_ri_color'])).ljust(30) +
#              '; default=' + str(defaults['max_catalog_ri_color']),
#              ';',
#              ';===== REGRESSION OPTIONS BLOCK =====================================',
#              ';----- OPTIONS for regression model, rarely used:',
#              (';#FIT_EXTINCTION ' + str(DEFAULT_MODEL_OPTIONS['fit_extinction'])).ljust(30) + '; default='
#              + str(DEFAULT_MODEL_OPTIONS['fit_extinction']) + ' or yes, False, No  (case-insensitive)',
#              (';#FIT_VIGNETTE ' + str(DEFAULT_MODEL_OPTIONS['fit_vignette'])).ljust(30) + '; default='
#              + str(DEFAULT_MODEL_OPTIONS['fit_vignette']) + ' or yes, False, No  (case-insensitive)',
#              (';#FIT_XY ' + str(DEFAULT_MODEL_OPTIONS['fit_xy'])).ljust(30) + '; default='
#              + str(DEFAULT_MODEL_OPTIONS['fit_xy']) + ' or yes, False, No  (case-insensitive)',
#              (';#FIT_JD ' + str(DEFAULT_MODEL_OPTIONS['fit_jd'])).ljust(30) + '; default='
#              + str(DEFAULT_MODEL_OPTIONS['fit_jd']) + ' or yes, False, No  (case-insensitive)',
#              ';'
#              ]
#     lines = [line + '\n' for line in lines]
#     fullpath = os.path.join(this_directory, TRANSFORM_CONTROL_FILENAME)
#     if not os.path.exists(fullpath):
#         with open(fullpath, 'w') as f:
#             f.writelines(lines)
#             log_file.write('New ' + TRANSFORM_CONTROL_FILENAME + ' file written.\n')


# def apply_calc_transform_selections(df_transform, user_selections):
#     """ Applies user selections to set desired df_transform's 'UseInModel' row(s) to False, in-place.
#     :param df_transform: observations dataframe. [pandas DataFrame]
#     :param user_selections: dict of lists of items to remove from df_obs before modeling. [dict of lists]
#     :return df_obs: [None] (both dataframes are modified in-place.)
#     """
#     # TODO: are we lucky enough for this to work as is?
#     apply_do_phot_selections(df_transform, user_selections)



# def make_diagnostics(df_model):
#     """ Use mixed-model regression to separate image effects ("random effect")
#             from comp effects (comp-averaged residual). Present data ~ready for plotting etc.
#     :param df_model: merged (obs, comps, images) dataframe of observation data [pandas DataFrame]
#     :return: df_model(updated with image effect etc), df_image_diagnostics,
#              df_comp_diagnostics. [yes, 3 pandas DataFrames]
#     """
#     # Extract comps & their statistics, assemble other data:
#     is_comp_id = (df_model['Type'] == 'Comp')
#     is_phot_filter = (df_model['Filter'] == MP_PHOTOMETRY_FILTER)
#     to_keep = is_comp_id & is_phot_filter
#     df = df_model.loc[to_keep, :].copy()  # (maybe cut down list of columns, later)
#     # comps = df['SourceID'].drop_duplicates().copy()
#     # images = df['FITSfile'].drop_duplicates().copy()
#     transform = TRANSFORM_CLEAR_SR_SR_SI
#     state = get_session_state()
#     extinction = state['extinction']['Clear']
#
#     # Make dep. variable from InstMags adjusted for terms: r mag, transform, extinction:
#     r_mag_term = df['r']
#     transform_term = transform * df['Airmass']
#     extinction_term = extinction * (df['r'] - df['i'])
#     df['DepVar'] = df['InstMag'] - r_mag_term - transform_term - extinction_term
#
#     # Do mixed-model regression with
#     fit = MixedModelFit(data=df, dep_var='DepVar', fixed_vars=['JD_fract'], group_var='FITSfile')
#
#     # Make df_image_diagnostics & merge onto df_model:
#     df_image_diagnostics = fit.df_random_effects.sort_values(by='GroupName').copy()
#     df_image_diagnostics = df_image_diagnostics.rename(columns={"GroupName": "FITSfile",
#                                                                 "Group": "ImageEffect"})
#     df_model = pd.merge(df_model, df_image_diagnostics, how='left', on='FITSfile', sort=False)
#
#     # Make df_comp_diagnostics; merge residual and CompEffect onto df_model:
#     df_comp_obs = fit.df_observations.copy().drop(columns=['FittedValue'])
#     df_comp_obs['Serial'] = list(df_comp_obs.index)
#     df = pd.merge(df, df_comp_obs, how='left', on='Serial', sort=False)
#     df_comp_effect = df.loc[:, ['SourceID', 'Residual']].groupby(['SourceID']).mean()  # excludes NaNs ++ !
#     df_comp_effect = df_comp_effect.rename(columns={'Residual': 'CompEffect'})
#     df = pd.merge(df, df_comp_effect, how='left', on='SourceID', sort=False)
#
#     # Make new df_model columns:
#     df['InstMagAdjusted'] = df['InstMag'] - transform_term - extinction_term
#
#     return df_model


# def make_comp_diagnostics(df_model):
#     """ Return for each comp: offset from InstMag expected, mean offset, offset metric,
#             where "expected" IM is calc from mixed-model fit on all *other* comps.
#     :param df_model: merged (obs, comps, images) dataframe of observation data [pandas DataFrame]
#     :return: df of one row/comp, columns=CompID, offsets (list), mean_offset, offset_metric;
#                   suitable for use in dignostic plotting. [pandas DataFrame]
#     """
#     # Extract comps & their statistics:
#     is_comp_id = (df_model['Type'] == 'Comp')
#     is_phot_filter = (df_model['Filter'] == MP_PHOTOMETRY_FILTER)
#     to_keep = is_comp_id & is_phot_filter
#     df = df_model.loc[to_keep, :].copy()  # (maybe cut down list of columns, later)
#     comps = df['SourceID'].drop_duplicates().copy()
#     images = df['FITSfile'].drop_duplicates().copy()
#
#     transform = TRANSFORM_CLEAR_SR_SR_SI
#     state = get_session_state()
#     extinction = state['extinction']['Clear']
#
#     dict_list = []
#     # Loop over comps:
#     for comp in comps:
#         df_other = df.loc[df['SourceID'] != comp, :].copy()  # (maybe cut down list of columns, later)
#
#         # Get Z (estimated nightly zero-point) from *other* comps
#         mean_z = (df_other['InstMag'] - df_other['r'] -
#                   transform * (df_other['r'] - df_other['i']) - extinction * df_other['Airmass']).mean()
#
#         # Get mean random effect (per-image general variation) form *other* comps:
#         offsets = []
#         for i, image in enumerate(images):
#             df_image = df_other.loc[df_other['FITSfile'] == image, :]
#             image_effect = (df_image['InstMag'] - mean_z - df_image['r'] -
#                             transform * (df_image['r'] - df_image['i']) -
#                             extinction * df_image['Airmass']).mean()
#
#             # Get offset for this comp, this image:
#             is_this_obs = (df['SourceID'] == comp) & (df['FITSfile'] == image)
#             inst_mag = df.loc[is_this_obs, 'InstMag']
#             r_catmag = df.loc[is_this_obs, 'r']
#             i_catmag = df.loc[is_this_obs, 'i']
#             airmass = df.loc[is_this_obs, 'Airmass']
#             offset = inst_mag - mean_z - r_catmag - transform * (r_catmag - i_catmag) \
#                 - extinction * airmass - image_effect
#             offsets.append(offset.iloc[0])
#         this_dict = {'CompID': comp, 'Offsets': offsets, 'MeanOffset': mean(offsets)}
#         dict_list.append(this_dict)
#     df_comp_diagnostics = pd.DataFrame(data=dict_list)
#     df_comp_diagnostics.index = list(df_comp_diagnostics['CompID'])
#     return df_comp_diagnostics

# ----- Ultimately, the code modeled in ml_2_groups won't do.
# Statsmodels "variance components" are just variance pools,
#    and effects per group (per-image, or per-comp) CANNOT be extracted, just don't exist.
# Choices, then are: (1) settle for one random effect (prob. image), or
#    (2) use R+lmer via rpy2 or maybe pymer4 package. Groan.
# def ml_2_groups():
#     # This converges to right values, but fails to yield random-effect values. Sigh.
#     from random import seed, randint, random, shuffle
#     import statsmodels.api as sm
#     # Make data:
#     n = 100
#     n1 = int(n/2)
#     n2 = int(n) - n1
#     # seed(2465)
#     a = pd.Series([i + 0.1 * random() for i in range(n)])      # indep fixed var
#     b = pd.Series([0.5 * random() for i in range(n)])   # "
#     gp1 = n1 * ['gp1_a'] + n2 * ['gp1_b']
#     gp2 = n1 * ['gp2_a'] + n2 * ['gp2_b']
#     shuffle(gp1)  # works in place
#     shuffle(gp2)  # "
#     val_gp1 = pd.Series([-2 if g.endswith('a') else 2 for g in gp1])
#     val_gp2 = pd.Series([-0.5 if g.endswith('a') else 0.5 for g in gp2])
#     y_random_error = pd.Series([0.1 * random() for i in range(n)])
#     intercept = pd.Series(n * [123.000])
#     y = intercept + a + b + val_gp1 + val_gp2 + y_random_error
#     df_x = pd.DataFrame({'Y': y, 'A': a, 'B': b, 'GP1': gp1, 'GP2': gp2})
#     # df_x['Intercept'] = 1
#     df_x['group'] = 'No group'  # because statsmodels cannot handle >1 group, but uses variance components.
#
#     # Make model with 2 crossed random variables:
#     model_formula = 'Y ~ A + B'
#     # variance_component_formula = {'Group1': '0 + C(GP1)',
#     #                               'Group2': '0 + C(GP2)'}
#     variance_component_formula = {'Group1': '0 + GP1',
#                                   'Group2': '0 + GP2'}
#     random_effect_formula = '0'
#     model = sm.MixedLM.from_formula(model_formula,
#                                     data=df_x,
#                                     groups='group',
#                                     vc_formula=variance_component_formula,
#                                     re_formula=random_effect_formula)
#     result = model.fit()
#     print(result.summary())
#     return result


# CATALOG_TESTS________________________________________________ = 0

# def get_transforms_landolt_r_mags(fits_directory):
# # This as comparison catalog (to check ATLAS refcat2) gives unreasonably high errors,
# #      wheres APASS10 works just fine. Hmm.
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

# #################################################################################################
# Keep the following 2 (commented out) "Canopus plots" for full-screen plots as demos for SAS talk.

# # "CANOPUS plot" (comps only, one point per obs:
# #     x=catalog r mag, y=obs InstMag(r) adjusted for extinction and transform):
# ax = axes[0, 0]
# make_labels_9_subplots(ax, 'Adjusted CANOPUS (all images)',
#                        'Catalog Mag (r)', 'Image-adjusted InstMag (r)', zero_line=False)
# df_canopus = df_plot_comp_obs.loc[:,
#              ['SourceID', 'Airmass', 'r', 'i', 'FITSfile', 'JD_fract', 'InstMag']]
# df_canopus['CI'] = df_canopus['r'] - df_canopus['i']
# df_canopus = pd.merge(df_canopus, df_image_effect, how='left', on='FITSfile', sort=False)
# extinction_adjustments = extinction * df_canopus['Airmass']
# transform_adjustments = transform * df_canopus['CI']
# image_adjustments = df_canopus['ImageEffect']
# jd_adjustments = jd_coefficient * df_canopus['JD_fract']
# sum_adjustments = extinction_adjustments + transform_adjustments + image_adjustments + jd_adjustments
# adjusted_instmags = df_canopus['InstMag'] - sum_adjustments
# df_canopus['AdjInstMag'] = adjusted_instmags
# # ax.scatter(x=df_canopus['r'], y=adjusted_instmags, alpha=0.6, color='darkblue')
# ax.scatter(x=df_canopus['r'], y=adjusted_instmags, alpha=0.6, color=comp_color)
# # first_comp_id = df_canopus.iloc[0, 0]
# # df_first_comp = df_canopus.loc[df_canopus['SourceID'] == first_comp_id, :]
# draw_x_line(ax, user_selections['min_r_mag'])
# draw_x_line(ax, user_selections['min_r_mag'])
#
# # "CANOPUS plot" (comps only, one point per obs:
# #     x=catalog r mag adjusted for extinction and transform, y=obs InstMag(r)):
# ax = axes[0, 1]
# make_labels_9_subplots(ax, 'Adjusted CANOPUS DIFF plot (all images)',
#                        'Catalog Mag (r)', 'Adjusted InstMag - r(cat)', zero_line=False)
# # Using data from previous plot:
# ax.scatter(x=df_canopus['r'], y=(adjusted_instmags - df_canopus['r']), alpha=0.6, color=comp_color)
# # ax.scatter(x=df_canopus['r'], y=adjusted_instmags, alpha=0.6, color='darkblue')
# draw_x_line(ax, user_selections['min_r_mag'])
# draw_x_line(ax, user_selections['min_r_mag'])
# #################################################################################################

# For spanning one subplot across more than one subplot tile:
# gs = axes[1, 0].get_gridspec()
# for row in [1, 2]:
#     for col in range(3):
#         axes[row, col].remove()
# axbig = fig.add_subplot(gs[1:, 1:])
# axbig.set_title('MP Target Lightcurve', loc='center', pad=-3)  # pad in points
# ax.set_xlabel(xlabel_jd, labelpad=-29)  # labelpad in points
# ax.set_ylabel('Mag (r)', labelpad=-5)  # "



# def do_ci_plot(result, mag_diff, mp_color, sigma_color, mp_string, filter_string, color_string):
#     """ Construct, display, and save as PNG a plot of the color index regression.
#     :param result: results from OLS regression. [object]
#     :param mag_diff: inst. magnitude difference (typically R-I) for MP. [float]
#     :param mp_color: color index Sloan (r-i) for MP. [float]
#     :param sigma_color: uncertainty in mp_color. [float]
#     :param mp_string: MP designation, e.g., 'MP 2415' or '2415 Ganesa'. [string]
#     :param filter_string: filters used in observations, e.g., 'R-I'. [string]
#     :param color_string: passbands in which color is expressed, e.g., 'Sloan r-i'. [string]
#     :return: None
#     """
#     fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(9, 6))  # (width, height) in "inches"
#     ax = axes  # not subscripted if just one subplot in Figure
#     page_title = 'Color index regression for ' + mp_string
#     ax.set_title(page_title, color='darkblue', fontsize=20, pad=30)
#     x_values = [x[1] for x in result.model.exog]  # unpack from 2-d array.
#     y_values = result.model.endog
#     ax.set_xlabel('Diff(instrumental magnitudes): ' + filter_string)
#     ax.set_ylabel('Color index: ' + color_string)
#     ax.grid(True, color='lightgray', zorder=-1000)
#     ax.scatter(x=x_values, y=y_values, alpha=0.7, color='black', zorder=+1000)
#     x_mp = [mag_diff]
#     y_mp = [mp_color]
#     ax.scatter(x=x_mp, y=y_mp, alpha=1, color='orangered', zorder=+1001)
#
#     plt.show()
#     filename = 'Image_ColorIndex.png'
#     fig.savefig(filename)