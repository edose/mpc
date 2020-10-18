__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
import configparser

MPC_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_FILE_SUBDIRECTORY = 'ini'
BOOT_INI_FILENAME = 'defaults.ini'


def read_defaults_ini():
    fullpath = os.path.join(MPC_ROOT_DIRECTORY, INI_FILE_SUBDIRECTORY, BOOT_INI_FILENAME)
    config = configparser.ConfigParser()
    config.read(fullpath)
    defaults_dict = dict()
    defaults_dict['instrument ini'] = config.get('Other ini files', 'Instrument')
    defaults_dict['site ini'] = config.get('Other ini files', 'Site')
    defaults_dict['observer ini'] = config.get('Other ini files', 'Observer')
    for key in config['Session Files']:
        defaults_dict[key] = config.get('Session Files', key)
    return defaults_dict


def read_instrument_ini(defaults_dict):
    """ We'll read just what we (think we) need to."""
    filename = defaults_dict['instrument ini']
    fullpath = os.path.join(MPC_ROOT_DIRECTORY, INI_FILE_SUBDIRECTORY, 'instrument', filename)
    config = configparser.ConfigParser()
    config.read(fullpath)
    instrument_dict = dict()

    for key in config['Limits']:
        float_value = float_or_warn(config.get('Limits', key), filename + ' [Limits]')
        instrument_dict[key.lower()] = float_value

    for key in config['Camera']:
        raw_value = config.get('Camera', key)
        if key in ['x pixels', 'y pixels', 'saturation adu', 'max vignetting pct at corner']:
            value = float_or_warn(raw_value, filename + ' [Limits]')
        else:
            value = raw_value
        instrument_dict[key.lower()] = value

    # [Filters] section:
    instrument_dict['available filters'] = multiline_ini_value_to_items(config.get('Filters', 'Available'))
    mag_exposure_dict = dict()
    value = config.get('Filters', 'Mag Exposures')
    lines = list(filter(None, (x.strip() for x in value.splitlines())))
    mag_exposure_lines = [line.strip() for line in lines]  # replace commas with spaces
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
        mag_exposure_dict[filter_name] = mag_exposure_list
    instrument_dict['mag exposures'] = mag_exposure_dict
    transform_dict = dict()
    transform_lines = multiline_ini_value_to_lines(config.get('Filters', 'Transforms'))
    for line in transform_lines:
        items = line.split()
        if len(items) in [5, 6]:
            key = tuple(items[:4])
            values = [float_or_warn(item, 'Transforms ' + line) for item in items[4:]]
            transform_dict[key] = values
        else:
            print(' >>>>> ERROR:', filename, 'bad transform line:', line)
    instrument_dict['transforms'] = transform_dict
    items = config.get('Filters', 'Default Color Filters').split()
    if len(items) == 2:
        instrument_dict['default_color_filters'] = tuple(items)
    else:
        print(' >>>>> ERROR:', filename, 'bad default color filters.')
    items = config.get('Filters', 'Default Color Index').replace('-', ' ').split()
    if len(items) == 2:
        instrument_dict['default_color_index'] = tuple(items)
    else:
        print(' >>>>> ERROR:', filename, 'bad default color index.')

    for key in config['Scale']:
        float_value = float_or_warn(config.get('Scale', key), filename + ' [Scale]')
        instrument_dict[key.lower()] = float_value

    for key in config['Timing']:
        float_value = float_or_warn(config.get('Timing', key), filename + ' [Timing]')
        instrument_dict[key.lower()] = float_value

    return instrument_dict


def read_observer_ini(defaults_dict):
    filename = defaults_dict['observer ini']
    fullpath = os.path.join(MPC_ROOT_DIRECTORY, INI_FILE_SUBDIRECTORY, 'observer', filename)
    config = configparser.ConfigParser()
    config.read(fullpath)
    observer_dict = dict()
    for section in config.sections():
        for key in config[section]:
            observer_dict[key.lower()] = config.get(section, key)
    return observer_dict


def read_site_ini(defaults_dict):
    filename = defaults_dict['site ini']
    fullpath = os.path.join(MPC_ROOT_DIRECTORY, INI_FILE_SUBDIRECTORY, 'site', filename)
    config = configparser.ConfigParser()
    config.read(fullpath)
    site_dict = dict()
    site_dict['site_name'] = config.get('Site', 'Name')
    site_dict['mpc_code'] = config.get('Site', 'MPC Code')
    for key in config['Location']:
        site_dict[key] = float_or_warn(config.get('Location', key), filename + ' Location:' + key)
    site_dict['coldest_date'] = config.get('Climate', 'Coldest Date')
    extinction_lines = multiline_ini_value_to_lines(config.get('Climate', 'Extinctions'))
    extinction_dict = dict()
    for line in extinction_lines:
        items = line.split()
        if len(items) == 3:
            filter_name = items[0]
            summer_extinction = float_or_warn(items[1], filename + '[Climate][Extinctions]')
            winter_extinction = float_or_warn(items[2], filename + '[Climate][Extinctions]')
            extinction_dict[filter_name] = tuple([summer_extinction, winter_extinction])
        else:
            print(' >>>>> ERROR:', filename, 'bad extinctions line:', line)
    site_dict['extinctions'] = extinction_dict
    return site_dict


def read_selection_criteria_from_ini(filename):
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


