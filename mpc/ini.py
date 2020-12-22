__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
import configparser

import astropak.ini

MPC_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_FILE_SUBDIRECTORY = 'ini'
BOOT_INI_FILENAME = 'defaults.ini'


def make_defaults_dict(root_dir=MPC_ROOT_DIRECTORY, ini_subdir=INI_FILE_SUBDIRECTORY,
                       filename=BOOT_INI_FILENAME):
    """ Reads .ini file, returns defaults_dict.
        See defaults.template for value types and key names.
        :param root_dir: root directory; the mpc source directory except when testing. [string]
        :param ini_subdir: the subdirectory under root where defaults ini file is found. [string]
        :param filename: defaults ini filename, typically 'defaults.ini'. [string]
    :return: the defaults_dict. [python dict object; All keys and values are strings]

    """
    fullpath = os.path.join(root_dir, ini_subdir, filename)
    defaults_ini = astropak.ini.IniFile(fullpath)
    return defaults_ini.value_dict


def make_instrument_dict(defaults_dict):
    """ Reads .ini file, returns instrument_dict.
        See instrument.template for value types and key names.
    :return: instrument_dict. [python dict object, some items nested dicts]
    """
    filename = defaults_dict['instrument ini']
    fullpath = os.path.join(MPC_ROOT_DIRECTORY, INI_FILE_SUBDIRECTORY, 'instrument', filename)
    instrument_ini = astropak.ini.IniFile(fullpath)
    instrument_dict = instrument_ini.value_dict

    # Parse and overwrite 'mag exposures':
    mag_exposure_dict = dict()
    mag_exposure_lines = [line.strip() for line in instrument_dict['mag exposures']]
    for line in mag_exposure_lines:
        mag_exposure_list = []
        filter_name, raw_value = tuple(line.split(maxsplit=1))
        pairs = raw_value.split(',')
        for pair in pairs:
            items = pair.split()
            if len(items) == 2:
                mag = float_or_warn(items[0], filename + ' Mag Exposures ' + filter_name)
                exp = float_or_warn(items[1], filename + ' Mag Exposures ' + filter_name)
                mag_exposure_list.append((mag, exp))
            elif len(items) != 0:
                print(' >>>>> ERROR: ' + filename + ' Mag Exposure not in pairs: ' + line)
                return None
        mag_exposure_dict[filter_name] = tuple(mag_exposure_list)
    instrument_dict['mag exposures'] = mag_exposure_dict

    # Parse and overwrite 'transforms':
    transform_dict = dict()
    transform_lines = [line.strip() for line in instrument_dict['transforms']]
    for line in transform_lines:
        items = line.replace(',', ' ').split()
        if len(items) in [5, 6]:
            key = tuple(items[:4])
            values = tuple([float_or_warn(item, 'Transforms ' + line) for item in items[4:]])
            transform_dict[key] = values
        else:
            print(' >>>>> ERROR:', filename, 'bad transform line:', line)
    instrument_dict['transforms'] = transform_dict

    # Parse and overwrite 'available filters', 'default color filters', 'default color index':
    instrument_dict['available filters'] = tuple(instrument_dict['available filters'].split())
    instrument_dict['default color filters'] = tuple(instrument_dict['default color filters'].split())
    instrument_dict['default color index'] = \
        tuple([s.strip() for s in instrument_dict['default color index'].split('-')])
    return instrument_dict


def make_observer_dict(defaults_dict):
    """ Reads .ini file, returns observer_dict.
        Used mostly for ALCDEF file generation.
    :return: observer_dict. [python dict object, all keys and values are strings]
    See observer.template for value types and key names.
    """
    filename = defaults_dict['observer ini']
    fullpath = os.path.join(MPC_ROOT_DIRECTORY, INI_FILE_SUBDIRECTORY, 'observer', filename)
    observer_ini = astropak.ini.IniFile(fullpath)
    observer_dict = observer_ini.value_dict
    return observer_dict


def make_site_dict(defaults_dict):
    """ Reads .ini file, returns site_dict.
    :return: site_dict. [python dict object, some values are nested dicts.]
    See site.template for value types and key names.
    """
    filename = defaults_dict['site ini']
    fullpath = os.path.join(MPC_ROOT_DIRECTORY, INI_FILE_SUBDIRECTORY, 'site', filename)
    site_ini = astropak.ini.IniFile(fullpath)
    site_dict = site_ini.value_dict

    # Parse and overwrite 'extinctions':
    extinction_dict = dict()
    for line in site_dict['extinctions']:
        items = line.replace(',', ' ').split()
        if len(items) == 3:
            filter_name = items[0]
            summer_extinction = float_or_warn(items[1], filename + '[Climate][Extinctions]')
            winter_extinction = float_or_warn(items[2], filename + '[Climate][Extinctions]')
            extinction_dict[filter_name] = tuple([summer_extinction, winter_extinction])
        else:
            print(' >>>>> ERROR:', filename, 'bad extinctions line:', line)
    site_dict['extinctions'] = extinction_dict
    return site_dict


def read_color_control_from_ini(filename):
    """ Read ini file in session directory, return dict of comp selection criteria.
        Generally used in do_phot() or do_color().
    :param filename: name of control or color control ini file within session directory.
    :return: selection criteria. [python dict object]
    """
    config = configparser.ConfigParser()
    config.read(filename)
    comp_list, obs_list, image_list = [], [], []

    value = config.get('Selection', 'Omit Comps')
    comp_list = multiline_ini_value_to_items(value)
    value = config.get('Selection', 'Omit Obs')
    obs_serial_list = multiline_ini_value_to_items(value)
    value = config.get('Selection', 'Omit Images')
    image_list = multiline_ini_value_to_items(value)
    for comp in comp_list:
        warn_if_not_positive_int(comp, 'Comp: ' + comp)
    for obs in obs_serial_list:
        warn_if_not_positive_int(obs, 'Obs: ' + obs)

    min_sr_mag = config.get('Selection', 'Min SR Mag')
    max_sr_mag = config.get('Selection', 'Max SR Mag')
    max_catalog_dsr_mmag = config.get('Selection', 'Max Catalog dSR mmag')
    min_sri_color = config.get('Selection', 'Min Sloan RI Color')
    max_sri_color = config.get('Selection', 'Max Sloan RI Color')
    warn_if_not_float(min_sr_mag, 'Min SR Mag')
    warn_if_not_float(max_sr_mag, 'Max SR Mag')
    warn_if_not_float(max_catalog_dsr_mmag, 'Max Catalog dSR mmag')
    warn_if_not_float(min_sri_color, 'Min Sloan RI Color')
    warn_if_not_float(max_sri_color, 'Max Sloan RI Color')

    return {'comps': comp_list, 'obs': obs_serial_list, 'images': image_list,
            'min_sr_mag': min_sr_mag, 'max_sr_mag': max_sr_mag,
            'max_catalog_dsr_mmag': max_catalog_dsr_mmag,
            'min_sri_color': min_sri_color, 'max_sri_color': max_sri_color}


def read_regression_options_from_ini(filename):
    """ Reads regression options from ini file (typically color_control.ini or control.ini),
        returns dict of options controlling a linear regression to come.
    :param filename: name of ini file; assumes we are in correct working directory. [string]
    :return: option_dict [python dict].
    """
    config = configparser.ConfigParser()
    config.read(filename)
    option_dict = dict()

    value = config.get('Regression', 'MP Color Index SRI')
    option_dict['mp color index sri'] = float_or_warn(value, filename + ' MP Color Index SRI')
    value = config.get('Regression', 'Transform')
    lines = multiline_ini_value_to_lines(value)
    transform_dict = dict()
    for line in lines:
        items = line.replace(',', ' ').split()
        filter = items[0]
        passband = items[1]
        if items[2].lower() in ['fit=1', 'fit=2']:
            transform_dict[(filter, passband)] = items[2]
        elif items[2].lower() == 'use' and len(items) in [4, 5]:
            transform_dict[(filter, passband)] = tuple(items[3:])
            for item in items[3:]:
                warn_if_not_float(item, 'Transform value ' + item)
        else:
            print(' >>>>> WARNING: Transform line', line, 'in', filename, 'not understood.')

    value = config.get('Regression', 'Extinction')
    lines = multiline_ini_value_to_lines(value)
    extinction_dict = dict()
    for line in lines:
        items = line.replace(',', ' ').split()
        filter = items[0]
        if items[1].lower() == 'fit':
            extinction_dict[filter] = items[1]
        elif items[1].lower() == 'use' and len(items) == 3:
            extinction_dict[filter] = items[2]
            warn_if_not_float(items[2], 'Extinction value ' + items[2])
        else:
            print(' >>>>> WARNING: Extinction line', line, 'in', filename, 'not understood.')

    fit_vignette = config.getboolean('Regression', 'Fit Vignette')
    fit_xy = config.getboolean('Regression', 'Fit XY')
    fit_jd = config.getboolean('Regression', 'Fit JD')
    return {'transform': transform_dict, 'extinction': extinction_dict,
            'fit_vignette': fit_vignette, 'fit_xy': fit_xy, 'fit_jd': fit_jd}


SUPPORT_FUNCTIONS__________________________________________ = 0


def multiline_ini_value_to_lines(value):
    lines = list(filter(None, (x.strip() for x in value.splitlines())))
    lines = [line.replace(',', ' ').strip() for line in lines]  # replace commas with spaces
    return lines


def multiline_ini_value_to_items(value):
    lines = multiline_ini_value_to_lines(value)
    value_list = []
    _ = [value_list.extend(line.split()) for line in lines]
    return value_list


def float_or_warn(value, string_for_warning):
    try:
        return float(value)
    except ValueError:
        print(' >>>>> WARNING:', string_for_warning, 'cannot be parsed as float.')
        return None


def warn_if_not_float(value, string_for_warning):
    try:
        _ = float(value)
    except ValueError:
        print(' >>>>> WARNING:', string_for_warning, 'cannot be parsed as float.')


def warn_if_not_positive_int(value, string_for_warning):
    try:
        i = int(value)
        if not i >= 1:
            print(' >>>>> WARNING:', string_for_warning, 'is integer but negative.')
    except ValueError:
        print(' >>>>> WARNING:', string_for_warning, 'cannot be parsed as int.')


