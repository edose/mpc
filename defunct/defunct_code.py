__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"# *********************************************************************************************
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
#                        'BminusV', 'e_BminusV', 'Vmag', 'e_Vmag']
#     df = df_apass[columns_to_keep]
#     df = df[df['e_RAJ2000'] < 2.0]
#     df = df[df['e_DEJ2000'] < 2.0]
#     df = df[~pd.isnull(df['BminusV'])]
#     df = df[~pd.isnull(df['e_BminusV'])]
#     df = df[~pd.isnull(df['Vmag'])]
#     df = df[~pd.isnull(df['e_Vmag'])]
#     df['R_estimate'] = df['Vmag'] - 0.5 * df['BminusV']
#     df['e_R_estimate'] = np.sqrt(df['e_Vmag'] ** 2 + 0.25 * df['e_BminusV'] ** 2)  # error in quadrature.
#     if r_min is not None:
#         df = df[df['R_estimate'] >= r_min]
#     if r_max is not None:
#         df = df[df['R_estimate'] <= r_max]
#     df = df[df['e_R_estimate'] <= 0.04]
#     return df


# def try_reg():
#     """  This was used to get R_estimate coeffs, using catalog data to predict experimental Best_R_mag.
#     :return: [None]
#     """
#     df_comps_and_mp = get_df_comps_and_mp()
#     dfc = df_comps_and_mp[df_comps_and_mp['Type'] == 'Comp']
#     dfc = dfc[dfc['InAllImages']]

# from sklearn.linear_model import LinearRegression
# x = [[bv, v] for (bv, v) in zip(dfc['BminusV'], dfc['Vmag'])]
# y = list(dfc['Best_R_mag'])
# reg = LinearRegression(fit_intercept=True)
# reg.fit(x, y)
# print('\nsklearn: ', reg.coef_, reg.intercept_)
#
# xx = dfc[['BminusV', 'Vmag']]
# yy = dfc['Best_R_mag']
# reg.fit(xx, yy)
# print('\nsklearn2: ', reg.coef_, reg.intercept_)

# # statsmodel w/ formula api (R-style formulas) (fussy about column names):
# import statsmodels.formula.api as sm
# dfc['BV'] = dfc['BminusV']  # column name BminusV doesn't work in formula.
# result = sm.ols(formula='Best_R_mag ~ BV + Vmag', data=dfc).fit()
# print('\n' + 'sm.ols:')
# print(result.summary())

# statsmodel w/ dataframe-column api:
# import statsmodels.api as sm
# # make column BV as above
# # result = sm.OLS(dfc['Best_R_mag'], dfc[['BV', 'Vmag']]).fit()  # <--- without constant term
# result = sm.OLS(dfc['Best_R_mag'], sm.add_constant(dfc[['BminusV', 'Vmag']])).fit()
# print('\n' + 'sm.ols:')
# print(result.summary())
#
# # statsmodel w/ dataframe-column api:
# import statsmodels.api as sm
# # make column BV as above
# # result = sm.OLS(dfc['Best_R_mag'], dfc[['BV', 'Vmag']]).fit()  # <--- without constant term
# result = sm.OLS(dfc['Best_R_mag'], sm.add_constant(dfc[['R_estimate']])).fit()
# print('\n' + 'sm.ols:')
# print(result.summary())
# # also available: result.params, .pvalues, .rsquared

# FR

__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# FROM old Canopus.py:

# import os
#
# MP_TOP_DIRECTORY = 'C:/Astro/MP Photometry/'
#
#
# def canopus(mp_top_directory=MP_TOP_DIRECTORY, rel_directory=None):
#     """ Read all FITS in mp_directory, rotate right, bin 2x2, invalidating plate solution.
#     Intended for making images suitable (North Up, smaller) for photometric reduction in Canopus 10.
#     Tests OK ~20191101.
#     :param mp_top_directory: top path for FITS files [string]
#     :param rel_directory: rest of path to FITS files, e.g., 'MP_768/AN20191020' [string]
#     : return: None
#     """
#     this_directory = os.path.join(mp_top_directory, rel_directory)
#     # clean_subdirectory(this_directory, 'Canopus')
#     # output_directory = os.path.join(this_directory, 'Canopus')
#     output_directory = this_directory
#     import win32com.client
#     app = win32com.client.Dispatch('MaxIm.Application')
#     count = 0
#     for entry in os.scandir(this_directory):
#         if entry.is_file():
#             fullpath = os.path.join(this_directory, entry.name)
#             doc = win32com.client.Dispatch('MaxIm.Document')
#             doc.Openfile(fullpath)
#             doc.RotateRight()  # Canopus requires North-up.
#             doc.Bin(2)  # to fit into Canopus image viewer.
#             doc.StretchMode = 2  # = High, the better to see MP.
#             output_filename, output_ext = os.path.splitext(entry.name)
#             output_fullpath = os.path.join(output_directory, output_filename + '_Canopus' + output_ext)
#             doc.SaveFile(output_fullpath, 3, False, 3, False)  # FITS, no stretch, floats, no compression.
#             doc.Close  # no parentheses is actually correct. (weird MaxIm API)
#             count += 1
#             print('*', end='', flush=True)
#     print('\n' + str(count), 'converted FITS now in', output_directory)


# _____TRANSFORM_DETERMINATION_______________________________________________ = 0
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
