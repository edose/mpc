__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
import os
from datetime import datetime, timezone, timedelta
from math import ceil, floor
from collections import Counter
from enum import Enum
from typing import Union, List, Dict, Tuple

# External packages:
import numpy as np
import pandas as pd
# from astroquery.mpc import MPC
from bs4 import BeautifulSoup
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, get_sun

# import matplotlib
# matplotlib.use('Agg')  # Place before importing matplotlib.pyplot or pylab.
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches

# Author's packages:
from photrix.user import Site, Astronight
from photrix.util import RaDec, datetime_utc_from_jd, jd_from_datetime_utc, \
    hhmm_from_datetime_utc
from mpc.mp_astrometry import calc_exp_time, PAYLOAD_DICT_TEMPLATE, \
    get_one_html_from_list
from mpc.ini import make_defaults_dict, make_site_dict

# Astropak (author's) imports (a mistake to include, but live with it for now):
import astropak.web as web
from astropak.util import ra_as_degrees, dec_as_degrees, dec_as_hex, ra_as_hours, \
    degrees_as_hex, make_directory_if_not_exists, next_date_utc

MOON_CHARACTER = '\u263D'
# MOON_CHARACTER = '\U0001F319'  # Drat, matplotlib: 'glyph missing from current font'.
HTTP_OK_CODE = 200  # "OK. The request has succeeded."
CALL_TARGET_COLUMNS = ['LCDB', 'Eph', 'CN', 'CS', 'Favorable', 'Num', 'Name',
                       'OppDate', 'OppMag',
                       'MinDistDate', 'MDist', 'BrtDate', 'BrtMag', 'BrtDec',
                       'PFlag', 'P', 'AmplMin', 'AmplMax', 'U', 'Diam']

# MP PHOTOMETRY PLANNING:
MIN_MP_ALTITUDE = 29  # degrees
MIN_MOON_DISTANCE = 40  # degrees (default value)
MIN_HOURS_OBSERVABLE = 2  # (default value) less than this -> MP omitted from planning.
# DSW = ('254.34647d', '35.11861269964489d', '2220m')
# DSNM = ('251.10288d', '31.748657576406853d', '1372m')
# NSM_DOME = ('254.471022d', '32.903156d', '2180m')

# Next is: (v_mag, exp_time in sec), for photometry only. Targets S/N >~200.
EXP_TIME_CLEAR_TABLE_PHOTOMETRY = [(13, 90), (14, 150), (15, 300), (16, 600),
                                   (17, 870), (17.5, 900)]
EXP_TIME_BB_TABLE_PHOTOMETRY = [(13, 90), (14, 165), (15, 330), (16, 660),
                                (17, 900), (17.5, 900)]
EXP_OVERHEAD = 24  # Nominal exposure overhead, in seconds.
COV_RESOLUTION_MINUTES = 5  # min. coverage plot resolution, in minutes.
MAX_V_MAGNITUDE_DEFAULT = 18.4  # so that too-faint MPs don't get into planning & plots.
MAX_EXP_TIME_NO_GUIDING = 119

MPFILE_DIRECTORY = 'C:/Dev/Photometry/MPfile'
MP_LIGHTCURVE_DIRECTORY = 'C:/Astro/MP Photometry'
ACP_PLANNING_TOP_DIRECTORY = 'C:/Astro/ACP'
# MP_PHOTOMETRY_PLANNING_DIRECTORY = 'C:/Astro/MP Photometry/$Planning'
CURRENT_MPFILE_VERSION = '1.1'
# MPfile version 1.1 = added #BRIGHTEST & #FAMILY; #EPH_RANGE rather than #UTC_RANGE.

# COLOR PLANNING:
MAX_COLOR_MANDATORY_MP_NUMBER = 1000  # usually overridden by user
MIN_COLOR_V_MAGNITUDE_DEFAULT = 10.5
MAX_COLOR_V_MAGNITUDE_DEFAULT = 16
MIN_COLOR_MP_ALTITUDE = 35
MIN_COLOR_MOON_DISTANCE = 50
MAX_COLOR_SUN_ALT = -10
MIN_COLOR_PHASE_ANGLE = 2.0  # MPs below this phase angle are omitted from Color roster.
PHASE_ANGLE_TO_FLAG = 3.0
COLOR_PRIORITY_LIST_FULLPATH = 'C:/Astro/MP Color/MP Color priority list.txt'
COLOR_PREVIOUSLY_OBSERVED_FULLPATH = \
    'C:/Astro/MP Color/MP Color previously observed list.txt'
COLOR_ROSTER_TXT_FILENAME = 'MP Color Roster.txt'
COLOR_ROSTER_CSV_FILENAME = 'MP Color Roster.csv'

MIN_COLOR_HOURS_OBSERVABLE = 0.5
MAX_COLOR_MOTION = 1.5  # in arcseconds/minute
DELAY_BETWEEN_MPES_CALLS = 5  # seconds between successive calls to MP eph service
EXP_TIME_FACTOR_V14_TABLE_COLOR = [(11, 0.25), (12, 0.3), (13, 0.5), (14, 1),
                                   (15, 2.5), (16, 5)]
N_COLOR_OBSERVATIONS_SUFFICIENT = 2
MP_FLAG_CURRENT_LC = '!'
MP_FLAG_POTENTIAL_LC = '?'
MP_FLAG_ONE_REQUIRED = '*'
MP_FLAG_OMIT = 'OMIT'  # only this flag can be longer than one character.


class Sort(Enum):
    START = 'UTC_earliest'
    BEST = 'UTC_highest'
    LAST = 'UTC_latest'
    NUMBER = 'Number_int'
    MAG = 'V_mag'


class DuplicateMPFileError(Exception):
    pass


class NoObservableMPsError(Exception):
    pass


_____FOR_COLOR_PLANNING_____________________________________ = 0


def make_color_roster(an: Union[str, int],
                      site_name: str = 'NMS_Dome',
                      min_moon_dist: float = MIN_COLOR_MOON_DISTANCE,
                      min_hours: float = MIN_COLOR_HOURS_OBSERVABLE,
                      min_vmag: float = MIN_COLOR_V_MAGNITUDE_DEFAULT,
                      max_vmag: float = MAX_COLOR_V_MAGNITUDE_DEFAULT,
                      max_mandatory_mp_number: int = MAX_COLOR_MANDATORY_MP_NUMBER,
                      sort: Sort = Sort.BEST):
    """ Main COLOR INDEX planning function for MP photometry.
    :param an Astronight, e.g., 20200201.
    :param site_name: name of site for Site object.
    :param min_moon_dist: min dist from min (degrees) to consider MP observable.
    :param min_hours: min hours of observing time to include an MP. [float]
    :param min_vmag: minimum estimated V mag allowed for MP to be kept in table & plots.
    :param max_vmag: maximum estimated V mag allowed for MP to be kept in table & plots.
    :param max_mandatory_mp_number: max MP number to query as block from MPES.
    :param sort: property on which to sort rows of output table, from Sort enum.
    :return:
    Typical usage:
        make_color_roster(an=20210131, max_mandatory_mp_number=500)
    """
    # ===== Make required lists of MP numbers (as strings).
    # Make list of low-number MPs (base MP list):
    low_number_list = [str(mp + 1) for mp in range(max_mandatory_mp_number)]

    # Get user's PRIORITY MPs from Priority file:
    priority_dict = read_mp_control_file(COLOR_PRIORITY_LIST_FULLPATH,
                                         may_specify_flags=True)

    # Get POTENTIAL LIGHTCURVE MPs from MPfile directory:
    mpfile_dict = make_mpfile_dict()
    potential_lc_list = [str(mpfile.number) for mpfile in mpfile_dict.values()]
    potential_lc_dict = {mp: MP_FLAG_POTENTIAL_LC for mp in potential_lc_list}

    # Get CURRENT LIGHTCURVE MPs from MP lightcurve directory:
    current_lc_list = [fname[3:] for fname in os.listdir(MP_LIGHTCURVE_DIRECTORY)
                       if os.path.isdir(os.path.join(MP_LIGHTCURVE_DIRECTORY, fname))
                       and (fname.startswith("MP_"))]
    current_lc_dict = {mp: MP_FLAG_CURRENT_LC for mp in current_lc_list}

    # Get PREVIOUSLY OBSERVED MPs & obs counts:
    mps_completed_list, mps_in_progress_dict = \
        read_previously_observed_file(COLOR_PREVIOUSLY_OBSERVED_FULLPATH)

    # Make dict for roster list, including mps in progress:
    all_mps_dict = {mp: ' ' for mp in low_number_list}
    all_mps_dict.update(priority_dict)
    all_mps_dict.update(potential_lc_dict)
    all_mps_dict.update(current_lc_dict)
    all_mps_dict.update({mp: f'need {N_COLOR_OBSERVATIONS_SUFFICIENT - count}'
                         for mp, count in mps_in_progress_dict.items()})
    all_mps_dict = {mp: flag for mp, flag in all_mps_dict.items()
                    if mp not in mps_completed_list}  # remove completed MPs.

    # Make 3-column starter dataframe df_flag:
    dict_list = [{'MP': mp, 'Flag': flag}
                 for mp, flag in all_mps_dict.items() if mp.strip() != '']
    df_flag = pd.DataFrame(data=dict_list)
    df_flag['PreviousCount'] = [int(flag[5:]) if flag.startswith('need ')
                                else 0
                                for flag in df_flag['Flag'].values]
    df_flag.index = list(df_flag['MP'].astype(int))
    df_flag = df_flag.sort_index()

    # Make df_mpes dataframes for all MPs including those in progress:
    all_mps_list = df_flag['MP'].values.tolist()
    defaults_dict = make_defaults_dict()
    site_dict = make_site_dict(defaults_dict)
    an_string = str(an)
    year, month, day = int(an_string[:4]), int(an_string[4:6]), int(an_string[6:8])
    utc_start = next_date_utc(datetime(year, month, day, 0, 0, 0)
                              .replace(tzinfo=timezone.utc))
    df_mpes = web.make_df_mpes(mp_list=all_mps_list, site_dict=site_dict,
                               utc_start=utc_start, hours=13)

    # Screen df_mpes for user criteria incl: altitude, V mag, motion max, moon distance:
    df_mpes['V_mag'] = [float(v) for v in df_mpes['V_mag']]
    v_mag_ok = (df_mpes['V_mag'] >= min_vmag) & \
               (df_mpes['V_mag'] <= max_vmag)
    altitude_ok = (df_mpes['Alt'] >= MIN_COLOR_MP_ALTITUDE)
    motion_ok = (abs(df_mpes['Motion']) < MAX_COLOR_MOTION)
    moon_dist_ok = (df_mpes['Moon_dist'] >= min_moon_dist) | (df_mpes['Moon_alt'] < 0)
    sun_alt_ok = (df_mpes['Sun_alt'] <= MAX_COLOR_SUN_ALT)
    phase_angle_ok = (df_mpes['Phase_angle']) >= MIN_COLOR_PHASE_ANGLE
    row_ok = v_mag_ok & altitude_ok & motion_ok & moon_dist_ok & \
        sun_alt_ok & phase_angle_ok
    df_mpes = df_mpes.loc[row_ok, :].sort_values(by=['Number', 'UTC'])
    df_mpes.index = [i for i in range(len(df_mpes))]
    roster_mps = df_mpes['Number'].drop_duplicates()

    # Convert df_mpes (1 row per MPES time) to df_mp (1 row per MP):
    all_mps_dict_list = []
    for mp in roster_mps:
        df_mp = df_mpes.loc[df_mpes['Number'] == mp, :]
        n = len(df_mp)
        all_mps_dict = {'Number': mp,
                        'Number_int': int(mp),
                        'Name': df_mp['Name'].iloc[0],
                        'V_mag': max(df_mp['V_mag']),
                        'RA_deg': sum([ra_as_degrees(ra)
                                       for ra in df_mp['RA']]) / n,
                        'Dec_deg': sum([dec_as_degrees(dec)
                                        for dec in df_mp['Dec']]) / n,
                        'Moon_dist': min(df_mp['Moon_dist']),
                        'Motion': max(df_mp['Motion']),
                        'Phase_angle': sum(df_mp['Phase_angle']) / n
                        }
        all_mps_dict_list.append(all_mps_dict)
    df_mp = pd.DataFrame(data=all_mps_dict_list)
    df_mp.index = list(df_mp['Number_int'])
    df_mp = pd.merge(how='left', left=df_mp, right=df_flag,
                     left_index=True, right_index=True)

    # Add columns to df_mp (additional data):
    this_an = Astronight(an_string, site_name)
    radecs = [RaDec(ra, dec) for (ra, dec) in zip(df_mp['RA_deg'], df_mp['Dec_deg'])]
    df_mp['Gal_lat'] = [SkyCoord(ra, dec, unit='deg', frame='icrs').galactic.b.value
                        for (ra, dec) in zip(df_mp['RA_deg'], df_mp['Dec_deg'])]
    ts_observables = [this_an.ts_observable(radec, min_alt=MIN_COLOR_MP_ALTITUDE,
                                            min_moon_dist=MIN_COLOR_MOON_DISTANCE)
                      for radec in radecs]
    ts_transits = [this_an.transit(radec) for radec in radecs]
    df_mp['UTC_earliest'] = [ts.start for ts in ts_observables]
    df_mp['UTC_latest'] = [ts.end for ts in ts_observables]
    df_mp['UTC_highest'] = [min(max(tr, ob.start), ob.end)
                            for (tr, ob) in zip(ts_transits, ts_observables)]
    df_mp['Hours_observable'] = [ts.seconds / 3600 for ts in ts_observables]
    observable_long_enough = (df_mp['Hours_observable'] >= min_hours)
    if sort in Sort:
        sort_columns = [sort.value, 'UTC_latest', 'UTC_earliest', 'Number_int']
    else:
        print(' >>>>> WARNING: sort choice unknown. '
              'Use: Sort.START, .BEST, .LAST, .NUMBER, or .MAG')
        sort_columns = ['UTC_earliest', 'UTC_latest', 'Number_int']
    df_mp = df_mp.loc[observable_long_enough, :].sort_values(by=sort_columns)

    # Make lines from df_mp:
    in_progress_table_lines, main_table_lines, csv_lines = [], [], []
    number_length = max([len(number) for number in df_mp['Number']])
    max_name_length = min(30, max([len(name) for name in df_mp['Name']]))
    for mp in df_mp.index:
        mp_number_string = str(df_mp.loc[mp, 'Number']).rjust(number_length)
        mp_name_string = df_mp.loc[mp, 'Name'][:max_name_length].ljust(max_name_length)
        utc_string = hhmm_from_datetime_utc(df_mp.loc[mp, 'UTC_earliest']) + '-' + \
            hhmm_from_datetime_utc(df_mp.loc[mp, 'UTC_highest']) + '-' + \
            hhmm_from_datetime_utc(df_mp.loc[mp, 'UTC_latest']) + '  '
        ra_string = ra_as_hours(df_mp.loc[mp, 'RA_deg'], seconds_decimal_places=0)
        dec_string = dec_as_hex(df_mp.loc[mp, 'Dec_deg'], arcseconds_decimal_places=0)
        exposure_factor_string = calc_exp_time(df_mp.loc[mp, 'V_mag'],
                                               EXP_TIME_FACTOR_V14_TABLE_COLOR)
        acp_string = ' '.join(['COLOR MP_' +
                               str(df_mp.loc[mp, 'Number']).ljust(number_length),
                               '{0:4.2f}x'.format(exposure_factor_string),
                               '@', ra_string, dec_string])
        table_line = mp_number_string + \
            ' ' + df_mp.loc[mp, 'Flag'] + ' ' + \
            mp_name_string + ' ' + \
            utc_string + \
            '{0:5.1f}'.format(df_mp.loc[mp, 'V_mag']) + '  ' + \
            '{0:3d}'.format(round(df_mp.loc[mp, 'Moon_dist'])) + '  ' + \
            '{0:5.2f}'.format(df_mp.loc[mp, 'Motion']) + ' ' + \
            '{0:4d}'.format(int(round(df_mp.loc[mp, 'Gal_lat']))) + \
            ('*' if abs(df_mp.loc[mp, 'Gal_lat']) < 16.0 else ' ') + \
            '{0:6.1f}'.format(df_mp.loc[mp, 'Phase_angle']) + \
            (('*' if abs(df_mp.loc[mp, 'Phase_angle']) < PHASE_ANGLE_TO_FLAG
              else ' ') + '    ' +
             acp_string)
        # Write table line to either in-progress table or main table:
        if df_mp.loc[mp, 'PreviousCount'] >= 1:
            in_progress_table_lines.append(table_line)
        else:
            main_table_lines.append(table_line)

        # Write CSV file line, whether MP is in-progress or not:
        csv_line = '\t'.join([str(df_mp.loc[mp, 'Number']),
                              df_mp.loc[mp, 'Flag'],
                              df_mp.loc[mp, 'Name'],
                              hhmm_from_datetime_utc(df_mp.loc[mp, 'UTC_earliest']),
                              hhmm_from_datetime_utc(df_mp.loc[mp, 'UTC_highest']),
                              hhmm_from_datetime_utc(df_mp.loc[mp, 'UTC_latest']),
                              '{0:5.1f}'.format(df_mp.loc[mp, 'V_mag']),
                              '{0:3d}'.format(round(df_mp.loc[mp, 'Moon_dist'])),
                              '{0:5.2f}'.format(df_mp.loc[mp, 'Motion']),
                              '{0:4d}'.format(int(round(df_mp.loc[mp, 'Gal_lat']))) +
                              ('*' if abs(df_mp.loc[mp, 'Gal_lat']) < 16.0 else ' '),
                              '{0:6.1f}'.format(df_mp.loc[mp, 'Phase_angle']) +
                              ('*' if abs(df_mp.loc[mp, 'Phase_angle']) <
                               PHASE_ANGLE_TO_FLAG else ' '),
                              acp_string + '  ;  ' +
                              hhmm_from_datetime_utc(df_mp.loc[mp, 'UTC_earliest']) +
                              '-' +
                              hhmm_from_datetime_utc(df_mp.loc[mp, 'UTC_latest'])])
        csv_lines.append(csv_line)
    in_progress_mp_count = len(in_progress_table_lines)
    if in_progress_mp_count == 0:
        in_progress_table_lines = ['(No in-progress MPs are available for this night.)']


    # Build text File:
    # Make header lines, then assemble all lines:
    day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                   'Saturday', 'Sunday'][datetime(year, month, day).weekday()]
    total_mp_count = len(in_progress_table_lines) + len(main_table_lines)
    header_lines = ['MP COLOR Candidates for AN' + this_an.an_date_string + '   ' +
                    day_of_week.upper(),
                    '    as generated by make_color_roster() at ' +
                    '{:%Y-%m-%d %H:%M  UTC}'.format(datetime.now(timezone.utc)),
                    '    min.alt = ' + '{:.1f}'.format(MIN_COLOR_MP_ALTITUDE) +
                    u'\N{DEGREE SIGN}' +
                    '    V mag = ' + '{0:4.1f}'.format(min_vmag) +
                    ' to ' + '{0:4.1f}'.format(max_vmag) +
                    '    min. phase angle = ' +
                    '{:.1f}'.format(MIN_COLOR_PHASE_ANGLE) + u'\N{DEGREE SIGN}',
                    '    min. moon distance = ' +
                    '{:.1f}'.format(MIN_COLOR_MOON_DISTANCE) +
                    u'\N{DEGREE SIGN}' + ' when moon is up',
                    '    all MP numbers to ' + str(max_mandatory_mp_number) +
                    ' considered,\n        except those already observed at least ' +
                    str(N_COLOR_OBSERVATIONS_SUFFICIENT) + ' times,',
                    '    yielding ' +
                    str(len(in_progress_table_lines)) + ' in-progress MPs and ' +
                    str(len(main_table_lines)) +
                    ' other MP Candidates in the table below.',
                    '',
                    this_an.acp_header_string()]

    if in_progress_mp_count == 0:
        in_progress_table_header = ['', '========== IN-PROGRESS MPs (high priority):']
    else:
        in_progress_table_header = ['', '',
                                    '========== IN-PROGRESS MPs (high priority):',
                                    '  1st-best-last'.rjust(max_name_length + 21) +
                                    'V mag'.rjust(7) + 'Moon'.rjust(5) +
                                    '"/min'.rjust(7) + ' GalB' + '  Phase']
    main_table_pre_header = ['', '', '========== Other available MPs:',
                             ''.rjust(number_length) + 'Flag: ' +
                             MP_FLAG_CURRENT_LC + ' is current lightcurve, ' +
                             MP_FLAG_POTENTIAL_LC + ' is potential lightcurve, ' +
                             'and other flags as from priority file.']
    main_table_header = ['|'.rjust(number_length + 2) +
                         '  1st-best-last'.rjust(max_name_length + 16) +
                         'V mag'.rjust(7) + 'Moon'.rjust(5) +
                         '"/min'.rjust(7) + ' GalB' + '  Phase']
    all_lines = header_lines + \
        in_progress_table_header + in_progress_table_lines +\
        main_table_pre_header + main_table_header + main_table_lines + main_table_header

    # Write Text file:
    roster_directory = os.path.join(ACP_PLANNING_TOP_DIRECTORY, 'AN' + an_string)
    make_directory_if_not_exists(roster_directory)
    roster_txt_fullpath = os.path.join(roster_directory, COLOR_ROSTER_TXT_FILENAME)
    with open(roster_txt_fullpath, 'w') as this_file:
        this_file.write('\n'.join(all_lines))
    print(f'Text file: {total_mp_count} MPs written to {roster_txt_fullpath}.')

    # Write CSV file:
    csv_legend_line = ['\t'.join(['MP', 'Flag', 'Name', '1st', 'best', 'last',
                                  'V mag', 'Moon', 'as/min', 'GalB', 'Phase',
                                  'ACP string  ;  1st-last'])]
    roster_csv_fullpath = os.path.join(roster_directory, COLOR_ROSTER_CSV_FILENAME)
    with open(roster_csv_fullpath, 'w') as this_file:
        this_file.write('\n'.join(csv_legend_line + csv_lines))
    print(f'CSV file: {len(csv_lines)} MPs written to {roster_csv_fullpath}.')


def read_mp_control_file(fullpath, may_specify_flags):
    """ Read MP numbers from a priority file, return dictionary with MP numbers as keys,
            flags as values.
        Directive line begins with #FLAG; if following string does not exist,
            flag is ' ';  otherwise use first character of following string.
        Topmost FLAG dominates for a given MP number, so put most important flags
            at top of file.
    :param fullpath: full path of file to read. [string]
    :param may_specify_flags: True iff #FLAG directives are allowed.
        Error if False and #FLAG is present [boolean].
    :return: dict of MP numbers: flag as read from file. [dict of string:char]
    """
    bad_mp_strings = []
    mp_dict = {}
    current_flag = '?'
    if os.path.exists(fullpath) and os.path.isfile(fullpath):
        with open(fullpath, 'r') as mpfile:
            lines = mpfile.readlines()
    for line in lines:
        # Remove comments at end, get element strings:
        strings = line.strip().split(';', maxsplit=1)[0].replace(',', ' ').split()
        strings = [s.strip() for s in strings if s != '']
        if len(strings) >= 1:
            if strings[0].upper() == '#FLAG':
                if may_specify_flags is False:
                    raise SyntaxError('#FLAG directive not allowed in file ' + fullpath)
                if len(strings) <= 1:
                    current_flag = ' '
                else:
                    current_flag = (strings[1])[0]
                continue
            for mp_string in strings:
                try:
                    _ = int(mp_string)
                except ValueError:
                    bad_mp_strings.append(mp_string)
                else:
                    if mp_string not in mp_dict.keys():
                        mp_dict[mp_string] = current_flag
        if len(bad_mp_strings) > 0:
            error_text = str(len(bad_mp_strings)) + ' bad MP string(s) in ' + \
                         fullpath + ':\n' + '\n'.join(bad_mp_strings)
            raise ValueError(error_text)
    return mp_dict


# def read_all_mps_from_file(fullpath):
#     """ Read all MP numbers from a file, return list of MP numbers.
#     Does NOT remove duplicates.
#     :param fullpath: full path of file to read. [string]
#     :return: list of MP numbers from file. [list of string]
#     """
#     mp_list = []
#     if os.path.exists(fullpath) and os.path.isfile(fullpath):
#         with open(fullpath, 'r') as mpfile:
#             lines = mpfile.readlines()
#         strings = []
#         for line in lines:
#             # Remove comments at end, get element strings:
#             strings = line.strip().split(';', maxsplit=1)[0].replace(',', ' ').split()
#             strings = [s.strip() for s in strings if s != '']
#             mp_list.extend(strings)
#         bad_mp_strings = []
#         for mp_string in mp_list:
#             try:
#                 _ = int(mp_string)
#             except ValueError:
#                 bad_mp_strings.append(mp_string)
#         if len(bad_mp_strings) > 0:
#             error_text = str(len(bad_mp_strings)) + ' bad MP string(s) in ' +
#             fullpath + ':\n' + \
#                          '\n'.join(bad_mp_strings)
#             raise ValueError(error_text)
#     return mp_list


def read_previously_observed_file(fullpath: str = COLOR_PREVIOUSLY_OBSERVED_FULLPATH) \
        -> Tuple[List[str], Dict[str, int]]:
    raw_mp_dict = read_mp_control_file(fullpath=fullpath, may_specify_flags=False)
    counter = Counter(list(raw_mp_dict.keys()))
    mps_completed_list = [mp for mp, count in counter.items()
                          if count >= N_COLOR_OBSERVATIONS_SUFFICIENT]
    mps_in_progress_dict = {mp: count for mp, count in counter.items()
                             if count < N_COLOR_OBSERVATIONS_SUFFICIENT}
    return mps_completed_list, mps_in_progress_dict


# def get_mp_lists_from_file(fullpath, codes=[]):
#     """ Read MP list file, get MP numbers user-designated for each category in list file.
#     First character of a line may indicate a category by one-character code (not a numerical digit).
#     If first character of a line is numeric digit, put that line's MP numbers into default list (code=' ').
#     Blank lines ignored.
#     Comments (whole- or partial-line) marked with ';' anywhere in line are ignored.
#     :param fullpath: full path of file to read. [string]
#     :param codes: list of one-character (not digit) codes, e.g. ['+']. [list of one-character codes]
#     :return: dict of lists, where each key=code [character], list=MP numbers. [list of strings]
#     """
#     mp_lists = OrderedDict([(' ', [])])
#     for code in codes:
#         mp_lists[code] = []
#     if os.path.exists(fullpath) and os.path.isfile(fullpath):
#         with open(fullpath, 'r') as mpfile:
#             lines = mpfile.readlines()
#         for line in lines:
#             # Remove comments at end, get element strings:
#             strings = line.strip().split(';', maxsplit=1)[0].replace(',', ' ').split()
#             strings = [s for s in strings if s != '']
#             if len(strings) >= 1:
#                 if strings[0][0] in codes or strings[0][0].isdigit():
#                     code = strings[0][0]
#                     if code.isdigit():
#                         code = ' '
#                     else:
#                         strings[0] = strings[0][1:]
#                     strings = [s for s in strings if s != '']
#                     mp_lists[code].extend(strings)
#     return mp_lists


_____FOR_LIGHTCURVE_PLANNING________________________________ = 0


def make_mp_roster(an_string, site_name='NMS_Dome', min_moon_dist=MIN_MOON_DISTANCE,
                   min_hours=MIN_HOURS_OBSERVABLE, max_vmag=MAX_V_MAGNITUDE_DEFAULT,
                   plots_to_console=False, forced_include=None):
    """ Main LIGHTCURVE planning function for MP photometry.
    :param an_string: Astronight, e.g. 20200201 [string or int]
    :param site_name: name of site for Site object. [string]
    :param min_moon_dist: min dist from min (degrees) to consider MP observable. [float]
    :param min_hours: min hours of observing time to include an MP. [float]
    :param max_vmag: maximum estimated V mag allowed for MP to be kept in table & plots. [float]
    :param plots_to_console: [NOT IMPLEMENTED] will control whether plots are drawn to console;
        (plots are always written to files). [boolean]
    :param forced_include: list of MP numbers to include in any case. [list or tuple of ints or strs]
    :return: [None]
    """
    if forced_include is None:
        forced_include = []
    else:
        if isinstance(forced_include, str) or isinstance(forced_include, int):
            forced_include = [forced_include]
        forced_include = [str(mp) for mp in forced_include]

    # Make and print table of values, 1 line/MP, sorted by earliest observable UTC:
    df_an_table = make_df_an_table(an_string, site_name=site_name,
                                   min_moon_dist=min_moon_dist, min_hours=min_hours,
                                   forced_include=forced_include)

    # Write warning lines for MPs that are no longer observable:
    gone_west_lines = []
    for i in df_an_table.index:
        if df_an_table.loc[i, 'Status'].lower() == 'too late':
            gone_west_lines.append(' >>>>> WARNING: MP ' + df_an_table.loc[i, 'MPnumber'] +
                                   ' ' + df_an_table.loc[i, 'MPname'] +
                                   ' has gone low in the west, probably should archive the MPfile.')

    # Warn of, then remove too-faint MPs:
    bright_enough = [vmag <= max_vmag for vmag in df_an_table['V_mag']]
    in_forced_list = [mp in forced_include for mp in df_an_table['MPnumber']]
    mps_to_keep = [bright or forced for (bright, forced) in zip(bright_enough, in_forced_list)]
    mps_too_faint = df_an_table.loc[[not be for be in bright_enough], :]
    too_faint_lines = []
    for i in mps_too_faint.index:
        transit_utc = df_an_table.loc[i, 'TransitUTC']
        if (not np.isnan(transit_utc.day)) and (df_an_table.loc[i, 'MPnumber'] not in forced_include):
            too_faint_lines.append('      ' + 'V=' + '{:5.2f}'.format(df_an_table.loc[i, 'V_mag']) +
                                   '  transit=' + hhmm_from_datetime_utc(transit_utc) +
                                   df_an_table.loc[i, 'MPnumber'].rjust(7) + ' ' +
                                   df_an_table.loc[i, 'MPname'])
    if len(too_faint_lines) > 0:
        print(' >>>>> WARNING: MPs fainter than V', str(max_vmag), 'and thus excluded:')
        for line in too_faint_lines:
            print(line)

    df_an_table = df_an_table.loc[mps_to_keep, :]
    if df_an_table is None:
        print('No MPs observable for AN', an_string + '.')
        return

    df = df_an_table.copy()
    an = Astronight(an_string, site_name)
    an_year = int(an.an_date_string[0:4])
    an_month = int(an.an_date_string[4:6])
    an_day = int(an.an_date_string[6:8])
    day_of_week_string = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                          'Friday', 'Saturday', 'Sunday']\
        [datetime(an_year, an_month, an_day).weekday()]
    roster_header = ['MP Roster for AN ' + an_string + '  ' + day_of_week_string.upper() +
                     '     (site = ' + site_name + ')',
                     an.acp_header_string(), ''.join(100*['-'])]
    table_header_line_1 = ''.ljust(16) + '                       Exp Duty Mot.         '
    table_header_line_2 = ''.ljust(16) + 'Start Tran  End   V    (s)  %  "/exp   P/hr'
    roster_header.extend([table_header_line_1, table_header_line_2])
    for i in df.index:
        if df.loc[i, 'Status'].lower() not in ['ok', 'too late']:
            continue  # sentinel
        short_mp_name = df.loc[i, 'MPname'] if len(df.loc[i, 'MPname']) <= 8 \
            else df.loc[i, 'MPname'][:8] + '\u2026'
        duty_cycle = df.loc[i, 'DutyCyclePct']
        duty_cycle_string = ' --' \
            if (duty_cycle is None or np.isnan(duty_cycle) == True) \
            else str(int(round(duty_cycle))).rjust(3)
        motion = (df.loc[i, 'ExpTime'] / 60) * df.loc[i, 'MotionRate']
        motion_string = '{0:4.1f}'.format(motion) + ('*' if motion > 9 else ' ')
        period = df.loc[i, 'Period']
        period_string = '    ? ' if (period is None or np.isnan(period) == True) \
            else '{0:6.2f}'.format(period)
        table_line_elements = [df.loc[i, 'MPnumber'].rjust(6),
                               short_mp_name.ljust(9),
                               hhmm_from_datetime_utc(df.loc[i, 'StartUTC']),
                               hhmm_from_datetime_utc(df.loc[i, 'TransitUTC']),
                               hhmm_from_datetime_utc(df.loc[i, 'EndUTC']),
                               '{0:4.1f}'.format(df.loc[i, 'V_mag']),
                               str(int(round(df.loc[i, 'ExpTime']))).rjust(5),
                               duty_cycle_string,
                               motion_string,
                               period_string,
                               ' ' + df.loc[i, 'PhotrixPlanning']]
        # if df.loc[i, 'ExpTime'] <= MAX_EXP_TIME_NO_GUIDING:
        #     table_line_elements.append(' AG+')
        roster_header.append(' '.join(table_line_elements))
    print('\n'.join(roster_header + [table_header_line_1] + [table_header_line_2]))
    if len(gone_west_lines) >= 1:
        print('\n\n' + '\n'.join(gone_west_lines))

    # Make ACP AN directory if it doesn't exist:
    text_file_directory = os.path.join(ACP_PLANNING_TOP_DIRECTORY, 'AN' + an_string)
    os.makedirs(text_file_directory, exist_ok=True)

    # Delete previous plot and text files, if any:
    image_filenames = [f for f in os.listdir(text_file_directory) if f.endswith('.png')]
    for f in image_filenames:
        os.remove(os.path.join(text_file_directory, f))
    table_filenames = [f for f in os.listdir(text_file_directory) if f.startswith('MP_table_')]
    for f in table_filenames:
        os.remove(os.path.join(text_file_directory, f))

    # Write text file:
    text_filename = 'MP_roster_' + an_string + '.txt'
    text_file_fullpath = os.path.join(text_file_directory, text_filename)
    with open(text_file_fullpath, 'w') as this_file:
        this_file.write('\n'.join(roster_header))
        if len(gone_west_lines) >= 1:
            this_file.write('\n\n' + '\n'.join(gone_west_lines))

    # Display plots; also write to PNG files:
    is_to_plot = [status.lower() == 'ok' for status in df_an_table['Status']]
    df_for_plots = df_an_table.loc[is_to_plot]
    make_coverage_plots(an_string, site_name, df_for_plots, plots_to_console)


def make_roster_one_class(month_string='202007', mp_family='MC'):
    """ From CALL website, get targets for one month, one MP family (e.g., 'MC' for Mars Crosser).
    :param month_string: month designator, as 'yyyymm'. [string]
    :param mp_family: official MP family Code. [string]
    :return: dataframe resembling CALL target HTML table, capable of being appended to other MP family
        tables. [pandas DataFrame]
    """
    year_month = month_string.strip()
    if len(year_month) == 6:
        target_year, target_month = year_month[0:4], year_month[4:6]
    else:
        print(' >>>>> ERROR: month_string not valid.')
        return

    import requests
    # url = "http://www.minorplanet.info/PHP/call_OppLCDBQuery.php"  # old URL (-2021)
    url = "https://www.minorplanet.info/php/callopplcdbquery.php"
    payload = {"OppData_NumberLow": "1",
               "OppData_NumberHigh": "999999",
               "OppDataNameOptions[]": "Any",
               "OppData_NameSearch": "",
               "OppDataYearOptions[]": target_year,
               "OppDataMonthOptions[]": target_month,
               "Family[]": mp_family,
               "OppDataFavorableOptions[]": "All",
               "OppDataCALLOptions[]": "All",
               "OppDataLCDBOptions[]": "All",
               "OppData_MinMag": "11",
               "OppData_MaxMag": "16",
               "OppData_MinDec": "-22",
               "OppData_MaxDec": "90",
               "OppData_MaxDia": "5000",
               "submit": "Submit"}
    r = requests.post(url, data=payload)

    mp_list = []
    if r.status_code == HTTP_OK_CODE:
        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table')
        mp_table = tables[2]
        mp_lines = mp_table.find_all('tr')
        for line in mp_lines[1:]:
            cells = line.find_all('td')
            cell_strings = [cell.text if cell.text != '\xa0' else '' for cell in cells]  # clean weird HTML.
            mp_list.append(cell_strings)
    mp_dict_list = [dict(zip(CALL_TARGET_COLUMNS, mp)) for mp in mp_list]
    df = pd.DataFrame(data=mp_dict_list).drop(columns=CALL_TARGET_COLUMNS[0:2])
    opp_date_valid = [not s.startswith('99') for s in df['OppDate']]
    df = df[opp_date_valid]
    df['YearMonth'] = month_string
    df['Family'] = mp_family
    index_values = [number + '_' + family for (number, family) in zip(df['Num'], df['Family'])]
    df.index = index_values
    return df


def backlog(months=6):
    days_forward = int(round(months * (365.25 / 12)))  # good enough for now.
    latest_utc_brightest = datetime.now().replace(tzinfo=timezone.utc) + timedelta(days=days_forward)
    mpfile_dict = make_mpfile_dict()
    backlog_dict_list = []
    for mp in mpfile_dict.keys():
        mpfile = mpfile_dict[mp]
        if mpfile.brightest_utc <= latest_utc_brightest:
            backlog_dict = {'Number': mpfile.number, 'Name': mpfile.name,
                            'Brightest': mpfile.brightest_utc, 'Motive': mpfile.motive}
            backlog_dict_list.append(backlog_dict)
    df = pd.DataFrame(data=backlog_dict_list).sort_values(by='Brightest')
    df.index = list(df['Number'])
    return df  # temporary for testing.


def make_df_an_table(an_string, site_name, min_moon_dist=MIN_MOON_DISTANCE,
                     min_hours=MIN_HOURS_OBSERVABLE, forced_include=None):
    """  Make dataframe of one night's MP photometry planning data, one row per MP.
         USAGE: df = make_df_an_table('20200201')
    :param an_string: Astronight, e.g. 20200201 [string or int]
    :param site_name: name of site for Site object. [string]
    :param min_moon_dist: min dist from min (degrees) to consider MP observable [float].
    :param min_hours: min hours of observing time to include an MP. [float]
    :param forced_include: list of MP numbers to include in any case. [list of strs]
    :return: table of planning data, one row per current MP, many columns including one for
                           coverage list of dataframes. [DataFrame]
    """
    an_string = str(an_string)  # (precaution in case int passed in)
    an_object = Astronight(an_string, site_name)
    # dark_start, dark_end = an_object.ts_dark.start, an_object.ts_dark.end
    mid_dark = an_object.local_middark_utc
    # dark_no_moon_start, dark_no_moon_end = an_object.ts_dark_no_moon.start, an_object.ts_dark_no_moon.end
    mpfile_dict = make_mpfile_dict()
    if forced_include is None:
        forced_include = []

    an_dict_list = []  # results to be deposited here, to make a dataframe later.
    for mp in mpfile_dict.keys():
        mpfile = mpfile_dict[mp]
        # an_dict doesn't need to include defaults for case before or after mpfile ephemeris,
        #    because making the dataframe should put in NANs for missing keys anyway (check this later):
        an_dict = {'MPnumber': mpfile.number, 'MPname': mpfile.name, 'Motive': mpfile.motive,
                   'Priority': mpfile.priority, 'Period': mpfile.period}
        # Interpolate within ephemeris (because MP is moving in sky); 2 iterations s/be enough:
        data, status, ts_observable, mp_radec = None, None, None, None  # keep stupid IDE happy.
        best_utc = mid_dark  # best_utc will = mid-observable time at converged RA,Dec.

        # Converge on best RA, Dec, observable timespan (they interact, as MP is moving):
        hours_observable = 0.0  # default to keep IDE happy.
        for i in range(2):
            data = mpfile.eph_from_utc(best_utc)
            if data is None:
                print(' >>>>> ERROR: could not read this AN date from MPFile', mp)
                if mpfile.eph_range[1] < an_object.ts_dark.start:
                    status = 'too late'
                else:
                    status = 'too early'
                break
            status = 'ok'
            mp_radec = RaDec(data['RA'], data['Dec'])
            ts_observable = an_object.ts_observable(mp_radec,
                                                    min_alt=MIN_MP_ALTITUDE,
                                                    min_moon_dist=min_moon_dist)  # Timespan object
            hours_observable = ts_observable.seconds / 3600.0
            mid_observable = ts_observable.midpoint  # for loop exit
            best_utc = mid_observable  # update for loop continuation.

        # Mark valid MPs that are observable too briefly:
        if status.lower() == 'ok':
            if hours_observable < min_hours:
                status = 'too brief'

        # Override status if MP is forced and observable:
        if an_dict['MPnumber'] in forced_include and hours_observable > 0.0:
            status = 'ok'

        # For MPs observable this night, add one line to table:
        # print(mpfile.name, status)
        an_dict['Status'] = status
        if status.lower() == 'ok':
            an_dict['RA'] = data['RA']
            an_dict['Dec'] = data['Dec']
            an_dict['StartUTC'] = ts_observable.start
            an_dict['EndUTC'] = ts_observable.end
            an_dict['TransitUTC'] = an_object.transit(mp_radec)
            an_dict['MoonDist'] = mp_radec.degrees_from(an_object.moon_radec)
            an_dict['PhaseAngle'] = data['Phase']
            an_dict['V_mag'] = data['V_mag']
            an_dict['ExpTime'] = float(round(float(
                calc_exp_time(an_dict['V_mag'], EXP_TIME_BB_TABLE_PHOTOMETRY))))
            if an_dict['Period'] is not None:
                # Duty cycle is % of time spent observing this MP if one exposure per 1/60 of period.
                an_dict['DutyCyclePct'] = 100.0 * ((an_dict['ExpTime'] + EXP_OVERHEAD) / 60.0) / \
                                          an_dict['Period']
            else:
                an_dict['DutyCyclePct'] = None
            an_dict['MotionAngle'] = data['MotionAngle']
            an_dict['MotionRate'] = data['MotionRate']
        if status.lower() == 'ok':
            an_dict['PhotrixPlanning'] = 'IMAGE MP_' + mpfile.number + \
                                         ' BB=' + str(round(an_dict['ExpTime'])) + 's(*) ' + \
                                         ra_as_hours(an_dict['RA'], seconds_decimal_places=1) + ' ' + \
                                         degrees_as_hex(an_dict['Dec'], arcseconds_decimal_places=0)
            if an_dict['Period'] is not None:
                an_dict['Coverage'] = make_df_coverage(an_dict['Period'],
                                                       mpfile.obs_jd_ranges,
                                                       (jd_from_datetime_utc(an_dict['StartUTC']),
                                                        jd_from_datetime_utc(an_dict['EndUTC'])))
                an_dict['PhaseCoverage'] = make_df_phase_coverage(an_dict['Period'],
                                                                  mpfile.obs_jd_ranges)
            else:
                an_dict['Coverage'] = None
        an_dict_list.append(an_dict)
    if len(an_dict_list) == 0:
        return None
    df_an_table = pd.DataFrame(data=an_dict_list)
    df_an_table.index = df_an_table['MPnumber'].values
    mp_numbers = df_an_table['MPnumber'].values
    duplicate_mp_numbers = [mp_number for mp_number, count in Counter(mp_numbers).items() if count > 1]
    if duplicate_mp_numbers:
        raise DuplicateMPFileError(' '.join(duplicate_mp_numbers))
    if 'TransitUTC' not in df_an_table.columns:
        raise NoObservableMPsError('none available for AN ' + an_string)
    df_an_table = df_an_table.sort_values(by='TransitUTC')
    return df_an_table


def make_coverage_plots(an_string, site_name, df_an_table, plots_to_console):
    """ Make Nobs-vs-UTC plots, one per MP, i.e., plots of phase coverage by previous nights' observations.
    :param an_string: Astronight, e.g. 20200201 [string or int]
    :param site_name: name of site for Site object. [string]
    :param df_an_table: the master planning table for one Astronight [pandas DataFrame].
    :param plots_to_console: True iff plots desired to be sent to console, False for file safe only. [bool]
    :return: [None] (makes plots 3 x 3 per Figure/page).
    """
    # Nested functions:
    def make_labels_9_subplots(ax, title, xlabel, ylabel, text='', zero_line=True):
        ax.set_title(title, loc='center',  fontsize=10, pad=-3)  # pad in points
        ax.set_xlabel(xlabel, labelpad=-29)  # labelpad in points
        ax.set_ylabel(ylabel, labelpad=-5)  # "
        ax.text(x=0.5, y=0.95, s=text,
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if zero_line is True:
            ax.axhline(y=0, color='lightgray', linewidth=1, zorder=-100)

    # Prepare some data:
    df = df_an_table.copy()
    an_object = Astronight(an_string, site_name)
    dark_start, dark_end = an_object.ts_dark.start, an_object.ts_dark.end
    utc_zero = datetime(year=dark_start.year, month=dark_start.month,
                        day=dark_start.day).replace(tzinfo=timezone.utc)
    hours_dark_start = (dark_start - utc_zero).total_seconds() / 3600.0
    hours_dark_end = (dark_end - utc_zero).total_seconds() / 3600.0

    # Define plot structure (for both hourly coverage and phase coverage):
    mps_to_plot = [name for (name, cov) in zip(df['MPnumber'], df['Coverage']) if cov is not None]
    n_plots = len(mps_to_plot)  # count of individual MP plots.
    n_cols, n_rows = 3, 3
    n_plots_per_figure = n_cols * n_rows
    n_figures = ceil(n_plots / n_plots_per_figure)  # count of pages of plots.

    for i_figure in range(n_figures):
        n_plots_remaining = n_plots - (i_figure * n_plots_per_figure)
        n_plots_this_figure = min(n_plots_remaining, n_plots_per_figure)
        if n_plots_this_figure >= 1:
            # Start new Figure for HOURLY coverage:
            fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(11, 8))
            fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
            fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
            fig.suptitle('MP Hourly Coverage for ' + an_string + '     ::      Page ' +
                         str(i_figure + 1) + ' of ' + str(n_figures),
                         color='darkblue', fontsize=16)
            fig.canvas.set_window_title('MP hourly coverage for AN ' + an_string)
            subplot_text = 'rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
            fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=11,
                     color='dimgray')

            # Start new Figure for PHASE coverage:
            fig_p, axes_p = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(11, 8))
            fig_p.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
            fig_p.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
            fig_p.suptitle('MP Phase Coverage for ' + an_string + '     ::      Page ' +
                           str(i_figure + 1) + ' of ' + str(n_figures),
                           color='darkblue', fontsize=16)
            fig_p.canvas.set_window_title('MP phase coverage for AN ' + an_string)
            fig_p.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=11,
                       color='dimgray')

            # Loop through subplots for BOTH Figure pages (HOURLY & PHASE covereage):
            i_first = i_figure * n_plots_per_figure
            for i_plot in range(0, n_plots_this_figure):
                this_mp = mps_to_plot[i_first + i_plot]
                i_col = i_plot % n_cols
                i_row = int(floor(i_plot / n_cols))
                ax = axes[i_row, i_col]
                ax_p = axes_p[i_row, i_col]
                subplot_title = 'MP ' + this_mp +\
                                '    {0:.3f} h'.format(df.loc[this_mp, 'Period']) +\
                                '    {0:d} s'.format(int(round(df.loc[this_mp, 'ExpTime']))) +\
                                '    {0:d}%'.format(int(round(df.loc[this_mp, 'DutyCyclePct'])))
                make_labels_9_subplots(ax, subplot_title, '', '', '', zero_line=False)
                make_labels_9_subplots(ax_p, subplot_title, '', '', '', zero_line=False)

                # Plot HOURLY coverage curve:
                datetime_values = (df.loc[this_mp, 'Coverage'])['DateTimeUTC']
                x = [(dt - utc_zero).total_seconds() / 3600.0 for dt in datetime_values]  # UTC hour.
                y = (df.loc[this_mp, 'Coverage'])['Coverage']  # count of prev obs (this apparition).
                ax.plot(x, y, linewidth=3, alpha=1, color='darkblue', zorder=+50)
                ax.fill_between(x, 0, y, facecolor=(0.80, 0.83, 0.88), zorder=+49)
                max_y_hourly = y.max()

                # Plot PHASE coverage curve:
                x = (df.loc[this_mp, 'PhaseCoverage'])['Phase']
                y = (df.loc[this_mp, 'PhaseCoverage'])['PhaseCoverage']
                ax_p.plot(x, y, linewidth=3, alpha=1, color='darkgreen', zorder=+50)
                ax_p.fill_between(x, 0, y, facecolor=(0.83, 0.87, 0.83), zorder=+49)
                max_y_phase = y.max()

                # Set max y value for both coverage plots:
                max_nobs_to_plot = max(5, 1 + max(max_y_hourly, max_y_phase))

                # HOURLY coverage: Make left box if any unavailable timespan before available timespan:
                left_box_start = hours_dark_start
                left_box_end = (df.loc[this_mp, 'StartUTC'] - utc_zero).total_seconds() / 3600.0
                if left_box_end > left_box_start:
                    ax.add_patch(patches.Rectangle((left_box_start, 0),  # (x,y)bottom left, width, height
                                                   left_box_end - left_box_start,
                                                   max_nobs_to_plot,
                                                   linewidth=1, alpha=1, zorder=+100,
                                                   edgecolor='black', facecolor='darkgray'))

                # HOURLY coverage: Make right box if any unavailable timespan after available timespan:
                right_box_start = (df.loc[this_mp, 'EndUTC'] - utc_zero).total_seconds() / 3600.0
                right_box_end = hours_dark_end
                if right_box_end > right_box_start:
                    ax.add_patch(patches.Rectangle((right_box_start, 0),  # (x,y)bottom left, width, height
                                                   right_box_end - right_box_start,
                                                   max_nobs_to_plot,
                                                   linewidth=1, alpha=1, zorder=+100,
                                                   edgecolor='black', facecolor='darkgray'))

                # HOURLY coverage: add info box.
                max_len_infobox_mp_name = 12
                infobox_mp_name = df.loc[this_mp, 'MPname']
                if len(infobox_mp_name) > max_len_infobox_mp_name:
                    infobox_mp_name = infobox_mp_name[0:max_len_infobox_mp_name] + '...'
                infobox_text = infobox_mp_name + '  ' + MOON_CHARACTER + ' ' +\
                    str(int(round(df.loc[this_mp, 'MoonDist']))) + u'\N{DEGREE SIGN}' +\
                    '   ' + df.loc[this_mp, 'Motive']  # remove alpha angle 2021-01-23.
                # infobox_text = infobox_mp_name + '  ' + MOON_CHARACTER + ' ' +\
                #     str(int(round(df.loc[this_mp, 'MoonDist']))) + u'\N{DEGREE SIGN}' +\
                #     '   ' + 'α=' + '{0:.0f}'.format(df.loc[this_mp, 'PhaseAngle']) + u'\N{DEGREE SIGN}' +\
                #     '   ' + df.loc[this_mp, 'Motive']
                hours_dark = hours_dark_end - hours_dark_start
                n_box_top = max_nobs_to_plot
                n_box_bottom = 0.86 * max_nobs_to_plot
                box_height = n_box_top - n_box_bottom
                n_text_base = 0.92 * max_nobs_to_plot
                ax.add_patch(patches.Rectangle((hours_dark_start, n_box_bottom), hours_dark, box_height,
                                               linewidth=1, alpha=0.75, zorder=+200,
                                               fill=True, edgecolor='lightgray', facecolor='whitesmoke'))
                ax.text(x=hours_dark_start + 0.02 * hours_dark, y=n_text_base, s=infobox_text,
                        verticalalignment='center', fontsize=8, color='dimgray', zorder=+201)

                # PHASE coverage: add info box.
                ax_p.add_patch(patches.Rectangle((0, n_box_bottom), 1.0, box_height,
                                                 linewidth=1, alpha=0.75, zorder=+200,
                                                 fill=True, edgecolor='lightgray', facecolor='whitesmoke'))
                ax_p.text(x=0.02, y=n_text_base, s=infobox_text,
                          verticalalignment='center', fontsize=8, color='dimgray', zorder=+201)

                # Complete HOURLY coverage plot:
                ax.grid(b=True, which='major', axis='x', color='lightgray',
                        linestyle='dotted', zorder=-1000)
                ax.set_xlim(hours_dark_start, hours_dark_end)
                ax.set_ylim(0, max_nobs_to_plot)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0 / 6.0))
                x_transit = ((df.loc[this_mp, 'TransitUTC']) - utc_zero).total_seconds() / 3600.0
                ax.axvline(x=x_transit, color='lightblue', zorder=+40)

                # PHASE coverage: add vertical line for phase at StartUTC for this MP on this AN:
                JD_ref = ((df.loc[this_mp, 'Coverage'])['JD'])[0]
                phase_ref = ((df.loc[this_mp, 'Coverage'])['Phase'])[0]
                JD_start = jd_from_datetime_utc(df.loc[this_mp, 'StartUTC'])
                period_days = df.loc[this_mp, 'Period'] / 24.0
                phase_start = (phase_ref + (JD_start - JD_ref) / period_days) % 1.0
                # print(' >>>>>', this_mp, str(phase_start))
                ax_p.axvline(x=phase_start, color='darkgreen', linewidth=2, zorder=+40)

                # Complete PHASE coverage plot:
                ax_p.grid(b=True, which='major', axis='x', color='lightgray',
                          linestyle='dotted', zorder=-1000)
                ax_p.set_xlim(0.0, 1.0)
                ax_p.set_ylim(0, max_nobs_to_plot)
                ax_p.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                # ax_p.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

            # Remove any empty subplots (if this is the last Figure):
            for i_plot_to_remove in range(n_plots_this_figure, n_plots_per_figure):
                i_col = i_plot_to_remove % n_cols
                i_row = int(floor(i_plot_to_remove / n_cols))
                ax = axes[i_row, i_col]
                ax.remove()
                ax_p = axes_p[i_row, i_col]
                ax_p.remove()
                # plt.show()

            # Save HOURLY coverage plots:
            filename = 'MP_hourly_coverage_' + an_string + '_{0:02d}'.format(i_figure + 1) + '.png'
            # mp_photometry_planning_fullpath = os.path.join(MP_PHOTOMETRY_PLANNING_DIRECTORY, filename)
            # # print('Saving hourly coverage to', mp_photometry_planning_fullpath)
            # fig.savefig(mp_photometry_planning_fullpath)
            acp_planning_fullpath = os.path.join(ACP_PLANNING_TOP_DIRECTORY, 'AN' + an_string, filename)
            # print('Saving hourly coverage to', acp_planning_fullpath)
            fig.savefig(acp_planning_fullpath)

            # Save PHASE coverage plots:
            filename = 'MP_phase_coverage_' + an_string + '_{0:02d}'.format(i_figure + 1) + '.png'
            # mp_photometry_planning_fullpath = os.path.join(MP_PHOTOMETRY_PLANNING_DIRECTORY, filename)
            # # print('Saving phase coverage to', mp_photometry_planning_fullpath)
            # fig_p.savefig(mp_photometry_planning_fullpath)
            acp_planning_fullpath = os.path.join(ACP_PLANNING_TOP_DIRECTORY, 'AN' + an_string, filename)
            # print('Saving phase coverage to', acp_planning_fullpath)
            fig_p.savefig(acp_planning_fullpath)


# def photometry_exp_time_clear_from_v_mag(v_mag):
#     """  Given V mag, return *Clear* filter exposure time suited to lightcurve photometry.
#     :param v_mag: target V magnitude [float]
#     :return: suitable exposure time in Clear filter suited to lightcurve photometry. [float]
#     """
#     return calc_exp_time(v_mag, EXP_TIME_BB_TABLE_PHOTOMETRY)


def make_df_coverage(period, obs_jd_ranges, target_jd_ranges, resolution_minutes=COV_RESOLUTION_MINUTES):
    """ Construct high-resolution array describing how well tonight's phases have previously been observed.
    :param period: MP lightcurve period, in hours. Required, else this function can't work. [float]
    :param obs_jd_ranges: start,end pairs of Julian Dates for previous obs, this MP.
        Typically obtained from an updated MPfile for that MP. [list of 2-tuples of floats]
    :param target_jd_ranges: start,end pair of JDs of proposed new observations.
        Presumably tonight's available observation timespan. [2-tuple or list of floats]
    :param resolution_minutes: approximate time resolution of output dataframe, in minutes. [float]
    :return: 1 row / timepoint in new obs window, columns = JD, DateTimeUTC, Phase, Nobs. [pandas DataFrame]
    """
    # Construct array of JDs covering target time span, and coverage count array of same length:
    if period is None:
        return None
    if period <= 0.0:
        return None
    # Set up target JD array and matching empty coverage array:
    resolution_days = resolution_minutes / 24 / 60
    n_target_jds = ceil((target_jd_ranges[1] - target_jd_ranges[0]) / resolution_days) + 1
    actual_resolution_days = (target_jd_ranges[1] - target_jd_ranges[0]) / (n_target_jds - 1)
    # Target JDs will form x of coverage plot:
    target_jds = [target_jd_ranges[0] + i * actual_resolution_days for i in range(n_target_jds)]
    # coverage is an accumulator array that will form y of plot:
    coverage = len(target_jds) * [0]

    # Build coverage array:
    period_days = period / 24.0
    # Phase zero defined at JD of earliest (previous) observation (same as in Canopus);
    # if there is no previous obs, use the target start JD:
    if len(obs_jd_ranges) >= 1:
        jd_at_phase_zero = min([float(obs_jd[0]) for obs_jd in obs_jd_ranges])
    else:
        jd_at_phase_zero = target_jd_ranges[0]
    for i, jd in enumerate(target_jds):
        for obs_jd_range in obs_jd_ranges:
            obs_jd_start, obs_jd_end = obs_jd_range
            diff_cycles_first_obs = int(ceil((jd - obs_jd_start) / period_days))  # larger
            diff_cycles_last_obs = int(floor((jd - obs_jd_end) / period_days))  # smaller
            for n in range(diff_cycles_last_obs, diff_cycles_first_obs + 1):
                obs_jd_candidate = jd - n * period_days
                if obs_jd_start <= obs_jd_candidate <= obs_jd_end:
                    coverage[i] += 1

    # Make dataframe:
    target_phase_array = [((jd - jd_at_phase_zero) / period_days) % 1 for jd in target_jds]
    dt_array = [datetime_utc_from_jd(jd) for jd in target_jds]
    df_coverage = pd.DataFrame({'JD': target_jds, 'DateTimeUTC': dt_array, 'Phase': target_phase_array,
                                'Coverage': coverage})
    return df_coverage


# def make_df_phase_coverage(period, obs_jd_ranges, phase_entries=100):
def make_df_phase_coverage(period, obs_jd_ranges, phase_entries=500):
    """ Construct high-res array for 1 MP describing how well all phases have previously been observed.
    :param period: MP lightcurve period, in hours. Required, else this function can't work. [float]
    :param obs_jd_ranges: start,end pairs of Julian Dates for previous obs, this MP.
        Typically obtained from an updated MPfile for that MP. [list of 2-tuples of floats]
    :param phase_entries: how many equally-spaced entries in phase to be computed. [int]
    :return: 1 row / timepoint in new obs window, columns = JD, DateTimeUTC, Phase, Nobs. [pandas DataFrame]
    """
    if period is None:
        return None
    if period <= 0.0:
        return None
    phase_coverage = (phase_entries + 1) * [0]  # accumulator array.

    # Build coverage array:
    period_days = period / 24.0
    # Phase zero defined at JD of earliest (previous) observation (same as in Canopus);
    # If there is no previous obs, then phase coverages are automatically zero anyway.
    if len(obs_jd_ranges) >= 1:
        jd_at_phase_zero = min([float(obs_jd[0]) for obs_jd in obs_jd_ranges])
        for obs_jd_range in obs_jd_ranges:
            obs_jd_start, obs_jd_end = obs_jd_range
            diff_cycles_first_obs = (obs_jd_start - jd_at_phase_zero) / period_days
            duration_cycles = (obs_jd_end - obs_jd_start) / period_days
            first_entry = int(round((diff_cycles_first_obs * float(phase_entries)) % float(phase_entries)))
            if first_entry >= phase_entries:
                first_entry -= phase_entries
            n_entries = round(duration_cycles * float(phase_entries))
            for i in range(first_entry, first_entry + n_entries):
                phase_coverage[i % phase_entries] += 1

    # Make dataframe:
    edge_phase_coverage = max(phase_coverage[0], phase_coverage[phase_entries])
    phase_coverage[0] = edge_phase_coverage
    phase_coverage[phase_entries] = edge_phase_coverage
    phase_values = [float(i) / phase_entries for i in range(len(phase_coverage))]
    df_phase_coverage = pd.DataFrame({'Phase': phase_values, 'PhaseCoverage': phase_coverage})
    return df_phase_coverage


_____PHASE_ANGLE_BISECTOR_______________________________________________________ = 0


def calc_phase_angle_bisector(times, mp_skycoords, deltas, site=None):
    """Calculate and return phase angle bisectors at time for a minor planet.
    :param times: list of times [list of astropy Time objects].
    :param mp_skycoords: list of MP RA,Dec sky locations corresponding to the times.
           [list of SkyCoord objects]
    :param deltas: list of distances site-to-MP, in AU. [list of floats]
    :param site: Site object for observing location. [Site object]
    :return: list of phase angle bisectors. [SkyCoord objects]
    """
    if len(times) != len(mp_skycoords):
        raise ValueError('parameters \'times\' and \'mp_skycoords\' must agree in length.')

    # Make GCRS Cartesian SkyCoords for sun center:
    sun_xyz = [get_sun(t).gcrs.cartesian for t in times]

    # Make GCRS Cartesian SkyCoords for observing site:
    site_loc = EarthLocation.from_geodetic(lon=site.longitude, lat=site.latitude, height=site.elevation)
    site_xyz = [site_loc.get_gcrs_posvel(t)[0] for t in times]

    # Make GCRS Cartesian SkyCoords for MP (NB: Delta is site (not geocenter) to MP):
    mp_xyz_from_geocenter = [SkyCoord(ra=sc.ra, dec=sc.dec,
                                      distance=delta * u.au, frame='gcrs').gcrs.cartesian
                             for (sc, delta) in zip(mp_skycoords, deltas)]
    mp_xyz = [(s.x + m.x, s.y + m.y, s.z + m.z)
              for (s, m) in zip(site_xyz, mp_xyz_from_geocenter)]

    # Make vectors from MP to sun, and to observing site:
    vector_mp_to_sun = [(sun.x - mp[0], sun.y - mp[1], sun.z - mp[2])
                        for (sun, mp) in zip(sun_xyz, mp_xyz)]
    vector_mp_to_site = [(site.x - mp[0], site.y - mp[1], site.z - mp[2])
                         for (site, mp) in zip(site_xyz, mp_xyz)]

    # Make vector sum = PAB vector:
    vector_sum = [(vsun[0] + vsite[0], vsun[1] + vsite[1], vsun[2] + vsite[2])
                  for (vsun, vsite) in zip(vector_mp_to_sun, vector_mp_to_site)]

    sc_list = [SkyCoord(x=-vs[0] / 2, y=-vs[1] / 2, z=-vs[2] / 2, unit='au',
                        frame='gcrs', representation_type='cartesian').geocentricmeanecliptic
               for vs in vector_sum]
    return sc_list


def test_calc_pab():
    times = [Time('2021-10-15 00:00:00', scale='utc'),
             Time('2021-10-16 00:00:00', scale='utc')]
    mp_skycoords = [SkyCoord('09 14 23.6 +18 05 21', unit=(u.hourangle, u.deg)),
                    SkyCoord('09 15 50.5 +18 01 27', unit=(u.hourangle, u.deg))]
    deltas = [2.770, 2.759]
    from photrix.user import Site
    site = Site('NMS_Dome')
    return calc_phase_angle_bisector(times, mp_skycoords, deltas, site)


_____MPFILE_________________________________________________ = 0


def make_mpfile(mp_number, utc_date_brightest=None, days=240, mpfile_directory=MPFILE_DIRECTORY):
    """ Make new MPfile text file for upcoming apparition.
    :param mp_number: MP's number, e.g., 7084. [int or string]
    :param utc_date_brightest: UTC date of MP brightest, e.g. '2020-02-01' or '20200201'. [string]
    :param days: number of days to include in ephemeris. [int]
    :param mpfile_directory: where to write file (almost always use default). [string]
    :return: [None]
    USAGE: make_mpfile(2653, 20200602)
    """
    mp_number = str(mp_number)
    days = max(days, 30)
    s = str(utc_date_brightest).replace('-', '')
    datetime_brightest = datetime(year=int(s[0:4]), month=int(s[4:6]),
                                  day=int(s[6:8])).replace(tzinfo=timezone.utc)
    apparition_year = datetime_brightest.year

    site = Site('NMS_Dome')

    # DO NOT OVERWRITE existing MPfile:
    mpfile_name = 'MP_' + str(mp_number) + '_' + str(apparition_year) + '.txt'
    mpfile_fullpath = os.path.join(mpfile_directory, mpfile_name)
    if os.path.exists(mpfile_fullpath):
        print(' >>>>> ERROR: MPfile for MP', mp_number, 'already exists, will not be overwritten.')
        return

    datetime_start = datetime_brightest - timedelta(days=int(floor(days/2.0)))
    datetime_end = datetime_brightest + timedelta(days=int(floor(days/2.0)))
    print('Ephemeris: from ', '{:%Y-%m-%d}'.format(datetime_start),
          'to about', '{:%Y-%m-%d}'.format(datetime_end))

    # Get strings from MPC (minorplanetcenter.com), making > 1 call if needed for number of days:
    n_days_per_call = 90
    n_calls = ceil(days / n_days_per_call)
    parameter_dict = PAYLOAD_DICT_TEMPLATE.copy()
    parameter_dict['TextArea'] = str(mp_number)
    parameter_dict['i'] = '1'  # interval between lines
    parameter_dict['u'] = 'd'  # units of interval; 'h' for hours, 'd' for days, 'm' for minutes
    parameter_dict['long'] = str(site.longitude).replace("+", "%2B")  # in deg
    parameter_dict['lat'] = str(site.latitude).replace("+", "%2B")    # in deg
    parameter_dict['alt'] = str(site.elevation)  # in meters
    # parameter_dict['long'] = '-105.5'.replace("+", "%2B")   # NMS_Dome longitude in deg
    # parameter_dict['lat'] = '+32.9'.replace("+", "%2B")  # NMS_Dome latitude in deg
    # parameter_dict['alt'] = '2180'    # NMS_Dome elevation (MPC "altitude") in m
    parameter_dict['igd'] = 'n'   # 'n' = don't suppress is sun up
    parameter_dict['ibh'] = 'n'   # 'n' = don't suppress line if MP down
    eph_lines = []
    mp_name = 'MP NAME NOT FOUND.'
    for i_call in range(n_calls):
        dt_this_call = datetime_start + i_call * timedelta(days=90)
        parameter_dict['d'] = '{:%Y %m %d}'.format(dt_this_call)
        parameter_dict['l'] = str(n_days_per_call)
        text = '\n'.join(get_one_html_from_list(mp_list=[mp_number], utc_date_string=parameter_dict['d'],
                                                payload_dict=parameter_dict))
        soup = BeautifulSoup(text, features='html5lib')
        mp_name = str(soup.contents[1].contents[2].contents[12]).split(')')[1].split('<')[0].strip()
        lines = [str(s).strip() for s in soup.find_all('pre')[0].contents]
        this_eph_lines = []
        for line in lines:
            this_eph_lines.extend(line.split('\n'))
        this_eph_lines = [s for s in this_eph_lines[3:-1] if not s.startswith(('/', '<'))]
        eph_lines.extend(this_eph_lines)

    # Parse MPC strings, make new dataframe:
    utc_strings = [s[:11].strip().replace(' ', '-') for s in eph_lines][:days]
    mpc_strings = [s[17:94].strip() for s in eph_lines][:days]
    moon_strings = [s[110:123] for s in eph_lines][:days]

    # Calculate PAB & make strings (to remove calls to minorplanet.info):
    table_times = [Time(us, scale='utc') for us in utc_strings]
    table_mp_sc = [SkyCoord(s[17:40], unit=(u.hourangle, u.deg)) for s in eph_lines[:days]]
    table_deltas = [float(s[40:49]) for s in eph_lines[:days]]
    sc_pab = calc_phase_angle_bisector(table_times, table_mp_sc, table_deltas, site)
    pab_strings = ['{0:6.1f} {1:6.1f}    '.format(sc.lon.degree, sc.lat.degree) for sc in sc_pab]

    # Make galactic_strings (to remove calls to minorplanet.info):
    table_galactics = [sc.galactic for sc in table_mp_sc]
    galactic_strings = ['{0:4.0f} {1:5.0f}   '.format(sc.l.degree, sc.b.degree) for sc in table_galactics]

    # Build table strings:
    table_strings = [date + '  ' + mpc + pab + moon + gal
                     for (date, mpc, pab, moon, gal)
                     in zip(utc_strings, mpc_strings, pab_strings, moon_strings, galactic_strings)]

    # Extract data for MPfile text file:
    utc_start_string = min(utc_strings)
    utc_end_string = max(utc_strings)
    mp_family = '>>> INSERT FAMILY HERE.'

    with open(mpfile_fullpath, 'w') as this_file:
        this_file.write('\n'.join(['; MPfile text file for MP photometry during one apparition.',
                                   '; Generated by mpc.mp_planning.make_mpfile() then edited by user',
                                   '#MP'.ljust(13) + str(mp_number).ljust(24) + '; minor planet number',
                                   '#NAME'.ljust(13) + mp_name.ljust(24) + '; minor planet name',
                                   '#FAMILY'.ljust(13) + mp_family.ljust(24) + '; minor planet family',
                                   '#APPARITION'.ljust(13) + str(apparition_year).ljust(24) + '; year',
                                   '#MOTIVE'.ljust(13) + 'XXX  ;  [pet,shape,period[n,X,??],low-phase]',
                                   '#PERIOD'.ljust(13) + 'nn.nnn  n'.ljust(24) +
                                   '; hours or ? followed by certainty per LCDB (1-3[+-])',
                                   '#AMPLITUDE'.ljust(13) + '0.nn'.ljust(24) + '; magnitudes expected',
                                   '#PRIORITY'.ljust(13) + 'n'.ljust(24) + '; 0-10 (6=normal)',
                                   '#BRIGHTEST'.ljust(13) +
                                   '{:%Y-%m-%d}'.format(datetime_brightest).ljust(24) +
                                   '; MP brightest UTC date as given',
                                   '#EPH_RANGE'.ljust(13) +
                                   (utc_start_string + ' ' + utc_end_string).ljust(24) +
                                   '; date range of ephemeris table below',
                                   '#VERSION'.ljust(13) + CURRENT_MPFILE_VERSION.ljust(24) +
                                   '; MPfile format version',
                                   ';',
                                   '; Record here the JD spans of observations already made of '
                                   'this MP, this opposition (for phase planning):',
                                   '; #OBS'.ljust(7) + '2458881.xxx  2458881.yyy'.ljust(27) +
                                   '; JD_start JD_end',
                                   '; #OBS'.ljust(7) + '2458883.xxx  2458883.yyy'.ljust(27) +
                                   '; JD_start JD_end',
                                   ';',
                                   '#EPHEMERIS',
                                   '================= For MP ' + str(mp_number) +
                                   ', retrieved from web sites ' +
                                   '{:%Y-%m-%d %H:%M  utc}'.format(datetime.now(timezone.utc))]))
        this_file.write('\n' + (71 * ' ') + '__MP Motion__  ____PAB____    ___Moon____   _Galactic_\n' +
                        '    UTC (0h)      RA      Dec.     Delta     R    Elong. Phase   V     "/min   '
                        'Angle  Long.   Lat.   Phase  Dist.  Long. Lat.\n' +
                        (125 * '-'))
        this_file.write('\n' + '\n'.join(table_strings))
        print(mpfile_fullpath, 'written. \n   >>>>> Now please edit: verify name & family, '
                               'enter period & code, amplitude, priority.')


def make_mpfile_dict(mpfile_directory=MPFILE_DIRECTORY):
    """  Returns dict of MPfiles, as: MP number: MPfile object.
    Usage: d = make_mpfile_dict()  --> returns *all* MPfiles. [dict]
    :param mpfile_directory: where the MPfiles reside. [string]
    :return: all MPfiles in a dictionary. [dict of MPfiles objects]
    """
    mpfile_names = all_mpfile_names(mpfile_directory)
    mpfile_dict = {mpfile_name[:-4]: MPfile(mpfile_name, mpfile_directory) for mpfile_name in mpfile_names}
    return mpfile_dict


def all_mpfile_names(mpfile_directory=MPFILE_DIRECTORY):
    """ Returns list of all MPfile names (from filenames in mpfile_directory). """
    mpfile_names = [fname for fname in os.listdir(mpfile_directory)
                    if (fname.endswith(".txt")) and (fname.startswith("MP_"))]
    return mpfile_names


class MPfile:
    """ One object contains all current-apparition data for one MP.
    Fields:
        .format_version [str, currently '1.0']
        .number: MP number [str representing an integer]
        .name: text name of MP, e.g., 'Dido' or '1952 TX'. [str]
        .family: MP family and family code. [str]
        .apparition: identifier (usually year) of this apparition, e.g., '2020'. [str]
        .motive: special reason to do photometry, or 'Pet' if simply a favorite. [str]
        .period: expected rotational period, in hours. [float]
        .period_certainty: LCDB certainty code, e.g., '1' or '2-'. [str]
        .amplitude: expected amplitude, in magnitudes. [float]
        .priority: priority code, 0=no priority, 10=top priority, 6=normal. [int]
        .brightest_utc: given date that MP is brightest, this apparition. [python datetime UTC]
        .eph_range: first & last date within the ephemeris (not observations). [2-tuple of datetime UTC]
        .obs_jd_ranges: list of previous observation UTC ranges. [list of lists of floats]
        .eph_dict_list: One dict per MPC ephemeris time (which are all at 00:00 UTC). [list of dicts]
            dict elements:
                'DateString': UTC date string for this MPC ephemeris line. [str as yyyy-mm-dd]
                'DatetimeUTC': UTC date. [py datetime object]
                'RA': right ascension, in degrees (0-360). [float]
                'Dec': declination, in degrees (-90-+90). [float]
                'Delta': distance Earth (observatory) to MP, in AU. [float]
                'R': distance Sun to MP, in AU. [float]
                'Elong': MP elongation from Sun, in degrees (0-180). [float]
                'Phase': Phase angle Sun-MP-Earth, in degrees. [float]
                'V_mag': Nominal V magnitude. [float]
                'MotionRate': MP speed across sky, in arcsec/minute. [float]
                'MotionDirection': MP direction across sky, in degrees, from North=0 toward East. [float]
                'PAB_longitude': phase angle bisector longitude, in degrees. [float]
                'PAB_latitude': phase angle bisector latitude, in degrees. [float]
                'MoonPhase': -1 to 1, where neg=waxing, 0=full, pos=waning. [float]
                'MoonDistance': Moon-MP distance in sky, in degrees. [float]
                'Galactic_longitude': in degrees. [float]
                'Galactic_latitude': in degrees. [float]
        .df_eph: the same data as in eph_dict_list, with dict keys becoming column names,
                    row index=DateUTC string. [pandas Dataframe]
        .is_valid: True iff all data looks OK. [boolean]
    """
    def __init__(self, mpfile_name, mpfile_directory=MPFILE_DIRECTORY):
        mpfile_fullpath = os.path.join(mpfile_directory, mpfile_name)
        if os.path.exists(mpfile_fullpath) and os.path.isfile(mpfile_fullpath):
            with open(mpfile_fullpath) as mpfile:
                lines = mpfile.readlines()
            self.is_valid = True  # conditional on parsing in rest of __init__()
        else:
            print('>>>>> MP file \'' + mpfile_fullpath + '\' not found. MPfile object invalid.')
            self.is_valid = False
            return
        lines = [line.split(";")[0] for line in lines]  # remove all comments.
        lines = [line.strip() for line in lines]  # remove leading and trailing whitespace.

        # ---------- Header section:
        self.format_version = MPfile._directive_value(lines, '#VERSION')
        if self.format_version != CURRENT_MPFILE_VERSION:
            print(' >>>>> ERROR: ' + mpfile_name + ':  Version Error. MPfile object invalid.')
            self.is_valid = False
            return
        self.number = self._directive_value(lines, '#MP')
        self.name = self._directive_value(lines, '#NAME')
        if self.name is None:
            print(' >>>>> Warning: Name is missing. (MP=' + self.number + ')')
            self.name = None
        self.family = self._directive_value(lines, '#FAMILY')
        self.apparition = self._directive_value(lines, '#APPARITION')
        self.motive = self._directive_value(lines, '#MOTIVE')
        words = self._directive_words(lines, '#PERIOD')
        if words is not None:
            try:
                self.period = float(words[0])
            except ValueError:
                # print(' >>>>> Warning: Period present but non-numeric,'
                # '[None] stored. (MP=' + self.number + ')')
                self.period = None
            if len(words) >= 2:
                self.period_certainty = words[1]
            else:
                self.period_certainty = '?'
        amplitude_string = self._directive_value(lines, '#AMPLITUDE')
        if amplitude_string is None:
            print(' >>>>> Warning: Amplitude is missing. [None] stored. (MP=' + self.number + ')')
            self.amplitude = None
        else:
            try:
                self.amplitude = float(amplitude_string)
            except ValueError:
                # print(' >>>>> Warning: Amplitude present but non-numeric,'
                # '[None] stored. (MP=' + self.number + ')')
                self.amplitude = None
        priority_string = self._directive_value(lines, '#PRIORITY')
        try:
            self.priority = int(priority_string)
        except ValueError:
            print(' >>>>> ERROR: Priority present but incorrect. (MP=' + self.number + ')')
            self.priority = None

        brightest_string = self._directive_value(lines, '#BRIGHTEST')
        try:
            year_str, month_str, day_str = tuple(brightest_string.split('-'))
            self.brightest_utc = datetime(int(year_str), int(month_str),
                                          int(day_str)).replace(tzinfo=timezone.utc)
        except ValueError:
            print(' >>>>> ERROR: Brightest incorrect. (MP=' + self.number + ')')
            self.brightest_utc = None
        eph_range_strs = self._directive_words(lines, '#EPH_RANGE')[:2]
        # self.utc_range = self._directive_words(lines, '#EPH_RANGE')[:2]
        self.eph_range = []
        for utc_str in eph_range_strs[:2]:
            year_str, month_str, day_str = tuple(utc_str.split('-'))
            utc_dt = datetime(int(year_str), int(month_str), int(day_str)).replace(tzinfo=timezone.utc)
            self.eph_range.append(utc_dt)

        # ---------- Observations (already made) section:
        obs_strings = [line[len('#OBS'):].strip() for line in lines if line.upper().startswith('#OBS')]
        obs_jd_range_strs = [value.split() for value in obs_strings]  # nested list of strings (not floats)
        self.obs_jd_ranges = []
        for range in obs_jd_range_strs:
            if len(range) >= 2:
                self.obs_jd_ranges.append([float(range[0]), float(range[1])])
            else:
                print(' >>>>> ERROR: missing #OBS field for MP', self.number, self.name)

        # ---------- Ephemeris section:
        eph_dict_list = []
        i_eph_directive = None
        for i, line in enumerate(lines):
            if line.upper().startswith('#EPHEMERIS'):
                i_eph_directive = i
                break
        if ((not (lines[i_eph_directive + 1].startswith('==========')) or
             (not lines[i_eph_directive + 3].strip().startswith('UTC')) or
             (not lines[i_eph_directive + 4].strip().startswith('----------')))):
            print(' >>>>> ERROR: ' + mpfile_name +
                  ':  MPEC header doesn\'t match expected from minorplanet.info page.')
            self.is_valid = False
            return
        eph_lines = lines[i_eph_directive + 5:]
        for line in eph_lines:
            eph_dict = dict()
            words = line.split()
            eph_dict['DateString'] = words[0]
            date_parts = words[0].split('-')
            eph_dict['DatetimeUTC'] = datetime(year=int(date_parts[0]),
                                               month=int(date_parts[1]),
                                               day=int(date_parts[2])).replace(tzinfo=timezone.utc)
            eph_dict['RA'] = 15.0 * (float(words[1]) + float(words[2]) / 60.0 + float(words[3]) / 3600.0)
            dec_sign = -1 if words[4].startswith('-') else 1.0
            dec_abs_value = abs(float(words[4])) + float(words[5]) / 60.0 + float(words[6]) / 3600.0
            eph_dict['Dec'] = dec_sign * dec_abs_value
            eph_dict['Delta'] = float(words[7])           # earth-MP, in AU
            eph_dict['R'] = float(words[8])               # sun-MP, in AU
            eph_dict['Elong'] = float(words[9])           # from sun, in degrees
            eph_dict['Phase'] = float(words[10])          # phase angle, in degrees
            eph_dict['V_mag'] = float(words[11])
            eph_dict['MotionRate'] = float(words[12])     # MP speed in arcseconds per minute.
            eph_dict['MotionAngle'] = float(words[13])    # MP direction, from North=0 toward East.
            eph_dict['PAB_longitude'] = float(words[14])  # phase angle bisector longitude, in degrees
            eph_dict['PAB_latitude'] = float(words[15])   # phase angle bisector latitude, in degrees
            eph_dict['MoonPhase'] = float(words[16])      # -1 to 1, where neg is waxing, pos is waning.
            eph_dict['MoonDistance'] = float(words[17])   # in degrees from MP
            eph_dict['Galactic_longitude'] = float(words[18])  # in degrees
            eph_dict['Galactic_latitude'] = float(words[19])   # in degrees
            eph_dict_list.append(eph_dict)
        self.eph_dict_list = eph_dict_list
        self.df_eph = pd.DataFrame(data=eph_dict_list)
        self.df_eph.index = self.df_eph['DatetimeUTC'].values
        self.is_valid = True

    @staticmethod
    def _directive_value(lines, directive_string, default_value=None):
        for line in lines:
            if line.upper().startswith(directive_string):
                return line[len(directive_string):].strip()
        return default_value  # if directive absent.

    def _directive_words(self, lines, directive_string):
        value = self._directive_value(lines, directive_string, default_value=None)
        if value is None:
            return None
        return value.split()

    def eph_from_utc(self, datetime_utc):
        """ Interpolate data from mpfile object's ephemeris; return dict, or None if bad datetime input.
            Current code requires that ephemeris line spacing = 1 day.
        :param datetime_utc: target utc date and time. [python datetime object]
        :return: dict of results specific to this MP and datetime, or None if bad datetime input. [dict]
        """
        mpfile_first_date_utc = self.eph_dict_list[0]['DatetimeUTC']
        i = (datetime_utc - mpfile_first_date_utc).total_seconds() / 24 / 3600  # a float.
        if not(0 <= i < len(self.eph_dict_list) - 1):  # i.e., if outside date range of eph table.
            return None
        return_dict = dict()
        i_floor = int(floor(i))
        i_fract = i - i_floor
        for k in self.eph_dict_list[0].keys():
            value_before, value_after = self.eph_dict_list[i_floor][k], self.eph_dict_list[i_floor + 1][k]
            # Add interpolated value if not a string;
            #    (use this calc form, because you can subtract but not add datetime objects):
            if isinstance(value_before, datetime) or isinstance(value_before, float):
                return_dict[k] = value_before + i_fract * (value_after - value_before)  # interpolated val.
        return return_dict


ANCILLARY_only________________________________________________________ = 0


# class KeplerObject:
#     def __init__(self, epoch_ma, ap, long, incl, ecc, a):
#         """
#         :param epoch_ma: epoch of mean anomaly, in Terrestrial Time JD. [float]
#         :param ap: argument of perihelion, in degrees. [float]
#         :param long: longitude of ascending node, in degrees. [float]
#         :param incl: inclination, in degrees. [float]
#         :param ecc: eccentricity, dimensionless. [float]
#         :param a: semi-major axis length, in AU. [float]
#         """
#         self.epoch_ma = epoch_ma
#         self.ap = ap
#         self.long = long
#         self.incl = incl
#         self.ecc = ecc
#         self.a = a
#         self.h = None  # H-G model reduced magnitude (placeholder value).
#         self.g = 0.15  # H-G model phase factor (default value).
#         self.name = ''
#
#
# def laguerre_delta(func, funcp, funcpp):
#     """ Returns Laguerre's method estimate the nearest root of a function that is nearly quadratic.
#     Reputed to be more robust to starting estimates than simply solving quadratic formula.
#     :param func: value of function at some x.
#     :param funcp: first derivative of function at the same x.
#     :param funcpp: second derivative of function at the same x.
#     :return: Best estimate of x shift needed to get near y=0. Will probably require iteration.
#     """
#     g = funcp / func
#     h = g * g - funcpp / func
#     numerator1 = g + sqrt(2 * h - g * g)
#     numerator2 = g - sqrt(2 * h - g * g)
#     numerator = numerator1 if abs(numerator1) > abs(numerator2) else numerator2
#     return - 2.0 / numerator


# def get_eph(mp, an, location='V28'):
#     """ Get one night's ephemeris for one minor planet.
#     :param mp: minor planet id [string or int]
#     :param an: Astronight ID, e.g. 20200110 [string or int]
#     :param location: (longitude, latitude, elevation) [tuple of strings, as in astropy]
#     :return: [pandas DataFrame], with columns:
#        Date [string], RA (degrees), Dec (degrees), Delta (earth dist, AU), r (sun dist, AU),
#        Elongation (deg), Phase (deg), V (magnitudes), Proper motion (arcsec/hour),
#        Direction (as compass, degrees).
#     """
#     mp_string = str(mp)
#     an_string = str(an)
#     date_string = '-'.join([an_string[0:4], an_string[4:6], an_string[6:8]])
#     time_string = '00:00:00'
#     df = MPC.get_ephemeris(mp_string, start=date_string + ' ' + time_string,
#                            number=14, step='1h', location=location).to_pandas()
#     df['Date'] = [dt.to_pydatetime().replace(tzinfo=timezone.utc) for dt in df['Date']]
#     print(df.columns)
#     df = df.drop(['Uncertainty 3sig', 'Unc. P.A.'], axis=1)
#     retur
