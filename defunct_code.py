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
