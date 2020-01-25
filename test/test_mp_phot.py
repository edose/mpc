from random import gauss, seed, uniform

import pandas as pd
import pytest

from mpc import mp_phot  # <--- test target

__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"


def test_make_comp_diagnostics():
    seed(1234)
    # Make test df_model:
    n_comps, n_images = 8, 5
    comps = ['comp' + str(i) for i in range(n_comps)]
    catmag_r = [uniform(11, 15) for i in range(n_comps)]
    catmag_i = [r + 0.2 + gauss(0, 0.01) for r in catmag_r]
    offset = [0, 0.03, 0.07, -0.10] + ((n_comps - 4) * [0])
    images = ['image' + str(i) for i in range(n_images)]
    random_effect = [gauss(0, 0.05) for i in range(n_images)]  # image effect
    airmass = [1.2 + 0.03 * i for i in range(n_images)]
    transform = mp_phot.TRANSFORM_CLEAR_SR_SR_SI
    state = mp_phot.get_session_state()
    extinction = state['extinction']['Clear']
    zero_point = -20.0

    dict_list = []
    for i_comp, comp in enumerate(comps):
        for i_image, image in enumerate(images):
            inst_mag = zero_point \
                       + catmag_r[i_comp] \
                       + transform * (catmag_r[i_comp] - catmag_i[i_comp]) \
                       + extinction * airmass[i_image] \
                       + offset[i_comp] \
                       + random_effect[i_image] \
                       + gauss(0, 0.01)
            this_dict = {'SourceID': comp, 'FITSfile': image, 'Airmass': airmass[i_image],
                         'r': catmag_r[i_comp], 'i': catmag_i[i_comp], 'InstMag': inst_mag}
            dict_list.append(this_dict)
    df_model = pd.DataFrame(data=dict_list)
    df_model['Type'] = 'Comp'
    df_model['Filter'] = 'Clear'
    df_model['Serial'] = range(len(df_model))
    df_model.index = list(df_model['Serial'])

    # Call test fn to produce df_comp_diagnostics
    df_comp_diagnostics = mp_phot.make_comp_diagnostics(df_model)
    assert len(df_comp_diagnostics) == n_comps

