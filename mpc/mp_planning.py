__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
from datetime import datetime, timezone, timedelta
from math import ceil

import pandas as pd
from astroquery.mpc import MPC
import requests
from bs4 import BeautifulSoup

from mpc.mp_astrometry import calc_exp_time, PAYLOAD_DICT_TEMPLATE, get_one_html_from_list
from photrix.user import Astronight
from photrix.util import degrees_as_hex, ra_as_hours

CURRENT_PHOT_MPS_FULLPATH = 'J:/Astro/Images/MP Photometry/$Planning/current_phot_mps.txt'
MIN_MP_ALTITUDE = 30  # degrees
DSW = ('254.34647d', '35.11861269964489d', '2220m')
DSNM = ('251.10288d', '31.748657576406853d', '1372m')
EXP_TIME_TABLE_PHOTOMETRY = [(13, 40), (14, 60), (15, 120), (16, 240)]  # (v_mag, exp_time sec), phot only.

MPFILE_DIRECTORY = 'C:/Dev/Photometry/MPfile'
CURRENT_MPFILE_VERSION = '1.0'
MPEC_REQUIRED_HEADER_START = 'Date (UTC)   RA              Dec         delta   r     elong  ' +\
                             'ph_ang   ph_ang_bisector   mag  \'/hr    PA'
EPH_REQUIRED_HEADER_START = 'Date         RA            Dec       Mag       E.D.     S.D.    Ph'\
                            '      E    Alt   Az    PABL    PABB     M Ph    ME    GL    GB'


def make_an_table(an, location=DSW):
    """  Make dataframe of one night's MP photometry planning data, one row per MP.
    :param an: Astronight, e.g. 1604 [string or int]
    :param location: Astropy-style location tuple (long, lat, elev). [3-tuple of strings]
    :return: table of planning data, one row per current MP. [pandas DataFrame]
    """
    an_string = str(an)
    an_object = Astronight(an_string, 'DSW')
    dark_start, dark_end = an_object.ts_dark.start, an_object.ts_dark.end
    df_mps = read_current_phot_mps()
    dict_list = []
    for mp in df_mps['MP']:
        mp_dict = {'MP': mp}
        df_eph = get_eph(mp, an, location)

        # Add rows for min & max altitude, drop low-altitude rows from top and bottom.
        # TODO: this is not right, needs to take into account dark_start and dark_end, as well.
        # Probably: find fractional row for each of altitude and darkness, take more restrictive,
        #     and only then do the interpolation and truncation.
        df_eph['Status'] = 'OK'
        alt = df_eph['Altitude']
        idx = df_eph.index
        for irow in range(len(df_eph) - 1):
            if alt[irow] <= MIN_MP_ALTITUDE < alt[irow + 1]:
                df_eph.loc[idx[:irow], 'Status'] = 'Drop'
                fract = (MIN_MP_ALTITUDE - alt[irow]) / (alt[irow + 1] - alt[irow])
                for col in df_eph.columns:
                    this_val = df_eph.loc[idx[irow], col]
                    next_val = df_eph.loc[idx[irow + 1], col]
                    df_eph.loc[idx[irow], col] = this_val + fract * (next_val - this_val)
            elif alt[irow] > MIN_MP_ALTITUDE >= alt[irow + 1]:
                df_eph.loc[idx[irow + 1:]] = 'Drop'
                fract = (MIN_MP_ALTITUDE - alt[irow]) / (alt[irow + 1] - alt[irow])
                for col in df_eph.columns:
                    this_val = df_eph.loc[idx[irow], col]
                    next_val = df_eph.loc[idx[irow + 1], col]
                    df_eph.loc[idx[irow + 1], col] = this_val + fract * (next_val - this_val)
        rows_to_keep = (df_eph['Status'] != 'Drop')
        df_eph = df_eph[rows_to_keep]

        # Make dict for this MP:
        mp_dict['V'] = df_eph['Vmag'].mean()
        mp_dict['exp_time'] = calc_exp_time(mp_dict['V'], EXP_TIME_TABLE_PHOTOMETRY)
        mp_dict['UTC_start'] = df_eph['Date'][0]
        mp_dict['UTC_end'] = df_eph['Date'][-1]
        mp_dict['ACP_string'] = '#IMAGE MP_' + mp + '  Clear=' + str(int(mp_dict['exp_time']))\
                                + 'sec(100)  ' + ra_as_hours(df_eph['RA'].mean()) + ' ' \
                                + degrees_as_hex(df_eph['Dec'].mean() + '  ;')
        mp_dict['Elongation'] = df_eph['Elongation'].mean()
        mp_dict['Phase'] = df_eph['Phase'].mean()
        mp_dict['Moon_dist'] = df_eph['Moon distance'].min()
        mp_dict['Moon_alt'] = df_eph['Moon altitude'].max()
        dict_list.append(mp_dict)
    df_an_table = pd.DataFrame(data=dict_list)
    df_an_table.index = df_an_table['MP']
    return df_an_table


def get_eph(mp, an, location='V28'):
    """ Get one night's ephemeris for one minor planet.
    :param mp: minor planet id [string or int]
    :param an: Astronight ID, e.g. 20200110 [string or int]
    :param location: (longitude, latitude, elevation) [tuple of strings, as in astropy]
    :return: [pandas DataFrame], with columns:
       Date [string], RA (degrees), Dec (degrees), Delta (earth dist, AU), r (sun dist, AU),
       Elongation (deg), Phase (deg), V (magnitudes), Proper motion (arcsec/hour),
       Direction (as compass, degrees).
    """
    mp_string = str(mp)
    an_string = str(an)
    date_string = '-'.join([an_string[0:4], an_string[4:6], an_string[6:8]])
    time_string = '00:00:00'
    df = MPC.get_ephemeris(mp_string, start=date_string + ' ' + time_string,
                           number=14, step='1h', location=location).to_pandas()
    df['Date'] = [dt.to_pydatetime().replace(tzinfo=timezone.utc) for dt in df['Date']]
    print(df.columns)
    df = df.drop(['Uncertainty 3sig', 'Unc. P.A.'], axis=1)
    return df


def read_current_phot_mps():
    """ Read current_phot_mps.txt, deliver dataframe of photometry target MPs and their motives.
    In current_phot_mps.txt, each MP row looks like: MP 2005 XS1  4.55  ;  motive
       where 4.55 is the period in hours (or ? if unknown),
       and motive is the reason we want data *for this night* (e.g., 'late phase').
    :return: [pandas DataFrame] with columns MP, Motive [both string].
    """
    mp_dict_list = []
    with open(CURRENT_PHOT_MPS_FULLPATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('MP '):
                halves = line[len('MP '):].split(';', maxsplit=1) + ['']  # (to ensure at least 2 elements)
                content = halves[0]
                mp = ''
                # Extract period from line content:
                p_index = content.find('P=')
                if p_index == -1:
                    period = '?'
                else:
                    period_word = content[p_index:].split(maxsplit=1)[0]
                    try:
                        period = period_word[2:]
                        x = float(period)
                    except ValueError:
                        period = '?'
                    mp = content[0:p_index].strip()  # use only content to the left of period.
                    content = content[0:p_index] + content[p_index + len(period_word):]  # remove period.
                # Complete remaining data for the MP entry:
                if mp == '':
                    mp = content.strip()
                motive = halves[1].strip()
                mp_dict_list.append({'MP': mp, 'Period': period, 'Motive': motive, 'Coverage': []})
            if line.startswith('AN'):
                halves = line[len('AN'):].split(';', maxsplit=1) + ['']  # (to ensure at least 2 elements)
                content = halves[0].split()
                an_string = content[0].strip()
                # Ensure that hhmm values are OK to use:
                hhmm_start = content[1].strip()
                hhmm_end = content[2].strip()
                if len(hhmm_start) != 4 or len(hhmm_end) != 4:
                    print('ERROR: hhmm_start and hhmm_end must have length 4 [',
                          hhmm_start, hhmm_end, ']')
                    return
                try:
                    hh_start = int(hhmm_start[0:2])
                    mm_start = int(hhmm_start[2:4])
                    hh_end = int(hhmm_end[0:2])
                    mm_end = int(hhmm_end[2:4])
                except ValueError:
                    print('ERROR: hhmm_start and hhmm_end must be legal hhmm time values [',
                          hhmm_start, hhmm_end, ']')
                    return
                if not((0 <= hh_start <= 23) and (0 <= mm_start <= 59) and
                       (0 <= hh_end <= 23) and (0 <= mm_end <= 59)):
                    print('ERROR: hhmm_start and hhmm_end must be legal hhmm time values [',
                          hhmm_start, hhmm_end, ']')
                    return
                comment = halves[1].strip()
                if not mp_dict_list:  # if it's empty.
                    print('ERROR in current_phot_mps_txt: AN line must follow a MP line.')
                    exit(-1)
                (mp_dict_list[-1])['Coverage'].append({'AN': an_string,
                                                       'hhmm_start': hhmm_start, 'hmm_end': hhmm_end,
                                                       'comment': comment})
    df = pd.DataFrame(data=mp_dict_list)
    df.index = df['MP']
    return df


def photometry_exp_time_from_v_mag(v_mag):
    """  Given V mag, return *Clear* filter exposure time suited to lightcurve photometry.
    :param v_mag: target V magnitude [float]
    :return: suitable exposure time in Clear filter suited to lightcurve photometry. [float]
    """
    return calc_exp_time(v_mag, EXP_TIME_TABLE_PHOTOMETRY)


MPFILE_STUFF____________________________________________________ = 0


def make_mpfile(mp_number, utc_date_start=None, days=90, mpfile_directory=MPFILE_DIRECTORY):
    """ Make new MPfile text file for upcoming apparition.
    :param mp_number: MP's number, e.g., 7084. [int or string]
    :param utc_date_start: date to start, e.g. '2020-02-01' or '20200201', default is today [string].
    :param days: number of days to include in ephemeris. [int]
    :param mpfile_directory: where to write file (almost always use default). [string]
    :return:
    """
    mp_number = str(mp_number)
    s = str(utc_date_start).replace('-', '')
    utc_start = '-'.join([s[0:4], s[5:6], s[7:8]])
    datetime_start = datetime(year=int(s[0:4]), month=int(s[5:6]), day=int(s[7:8]))
    days = max(days, 7)

    # Get strings from MPC (minorplanetcenter.com):
    n_days_per_call = 90
    n_calls = ceil(days / n_days_per_call)
    parameter_dict = PAYLOAD_DICT_TEMPLATE.copy()
    parameter_dict['TextArea'] = str(mp_number)
    parameter_dict['i'] = '1'  # interval between lines
    parameter_dict['u'] = 'd'  # units of interval; 'h' for hours, 'd' for days, 'm' for minutes
    parameter_dict['long'] = '-109'.replace("+", "%2B")   # longitude in deg
    parameter_dict['lat'] = '+31.96'.replace("+", "%2B")  # latitude in deg
    parameter_dict['alt'] = '1400'    # elevation (MPC "altitude") in m
    parameter_dict['igd'] = 'n'   # 'n' = don't suppress is sun up
    parameter_dict['ibh'] = 'n'   # 'n' = don't suppress line if MP down
    eph_lines = []
    for i_call in range(n_calls):
        dt_this_call = datetime_start + i_call * timedelta(days=90)
        parameter_dict['d'] = '{:%Y %m %d}'.format(dt_this_call)
        parameter_dict['l'] = str(n_days_per_call)
        text = '\n'.join(get_one_html_from_list(mp_list=[mp_number], utc_date_string=parameter_dict['d'],
                                                payload_dict=parameter_dict))
        soup = BeautifulSoup(text, features='html5lib')
        lines = [str(s).strip() for s in soup.find_all('pre')[0].contents]
        this_eph_lines = []
        for line in lines:
            this_eph_lines.extend(line.split('\n'))
        this_eph_lines = [s for s in this_eph_lines[3:-1] if not s.startswith(('/', '<'))]
        eph_lines.extend(this_eph_lines)

    # Parse MPC strings, make new dataframe:
    utc_strings = [s[:11].strip().replace(' ', '-') for s in eph_lines][:days]
    mpc_data = [s[17:94].strip() for s in eph_lines][:days]
    df_mpc = pd.DataFrame({'DateUTC': utc_strings, 'MPC_string': mpc_data})
    df_mpc.index = df_mpc['DateUTC'].values

    # Get strings from minorplanet.info One Asteroid Lookup:
    url = 'http://www.minorplanet.info/PHP/generateOneAsteroidInfo.php/'
    parameter_dict = {'AstNumber': str(mp_number), 'AstName': '',
                      'Longitude': '-109', 'Latitude': '32',  # for V16 DSNM Animas
                      'StartDate': '',  # assign this within loop, below.
                      'UT': '0', 'subOneShot': 'Submit'}
    n_days_per_call = 30
    n_calls = ceil(days / n_days_per_call)
    eph_lines = []
    for i_call in range(n_calls):
        dt_this_call = datetime_start + i_call * timedelta(days=30)
        parameter_dict['StartDate'] = '{:%Y-%m-%d}'.format(dt_this_call)
        r = requests.post(url, data=parameter_dict)
        soup = BeautifulSoup(r.text, features='html5lib')
        this_eph_lines = [str(s) for s in soup.find_all('pre')[1].contents[0].strings]
        eph_lines.extend(this_eph_lines)

    # Parse minorplanet.info strings, make dataframe:
    utc_strings = [s[:11].strip() for s in eph_lines][:days]
    mpinfo_strings = [s[91:] for s in eph_lines][:days]
    df_mpinfo = pd.DataFrame({'MP_info': mpinfo_strings})  # (don't duplicate DateUTC)
    df_mpinfo.index = utc_strings

    # Merge mpinfo dataframe into MPC dataframe:
    df_eph = pd.merge(df_mpc, df_mpinfo, how='left', left_index=True, right_index=True)
    df_eph['Output'] = [date + '  ' + mpc + mpinfo for (date, mpc, mpinfo) in zip(df_eph['DateUTC'],
                                                                                  df_eph['MPC_string'],
                                                                                  df_eph['MP_info'])]
    # Write MPfile text file:
    apparition_year = (datetime_start + timedelta(days=days/2)).year
    utc_start_string = '{:%Y-%m-%d}'.format(datetime_start)
    utc_end_string = '{:%Y-%m-%d}'.format(datetime_start + timedelta(days=days))
    mpfile_name = 'MP_' + str(mp_number) + '_' + str(apparition_year) + '.txt'
    mpfile_fullpath = os.path.join(mpfile_directory, mpfile_name)
    with open(mpfile_fullpath, 'w') as this_file:
        this_file.write('\n'.join(['; MPfile text file for MP photometry during one apparition.',
                                   '; Generated by mpc.mp_planning.make_mpfile()',
                                   '#MP'.ljust(13) + str(mp_number).ljust(24) + '; minor planet number',
                                   '#NAME'.ljust(13) + 'XXX'.ljust(24) + '; minor planet name',
                                   '#APPARITION'.ljust(13) + str(apparition_year).ljust(24) + '; year',
                                   '#MOTIVE'.ljust(13) + 'XXX',
                                   '#PERIOD'.ljust(13) + 'nn.nnn  [n]'.ljust(24) +
                                   '; hours or ? [certainty per LCDB (1-3[+-])',
                                   '#AMPLITUDE'.ljust(13) + '0.nn'.ljust(24) + '; magnitudes expected',
                                   '#PRIORITY'.ljust(13) + 'n'.ljust(24) + '; 0-10 (6=normal)',
                                   '#UTC_RANGE'.ljust(13) +
                                   (utc_start_string + ' ' + utc_end_string).ljust(24),
                                   '#VERSION'.ljust(13) + '1.0'.ljust(24) +
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
        this_file.write('\n' + '\n'.join(df_eph['Output']))
        print(mpfile_fullpath, 'written.')


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
                    if (fname.endswith(".txt")) and (not fname.startswith("$"))]
    return mpfile_names


class MPfile:
    def __init__(self, mpfile_name, mpfile_directory=MPFILE_DIRECTORY):
        data_dict = dict()
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
        self.apparition = self._directive_value(lines, '#APPARITION')
        self.motive = self._directive_value(lines, '#MOTIVE')
        words = self._directive_words(lines, '#PERIOD')
        self.period = float(words[0])
        if len(words) >= 2:
            self.period_certainty = words[1]
        else:
            self.period_certainty = '?'
        self.priority = int(self._directive_words(lines, '#PRIORITY')[0])
        self.date_range = self._directive_words(lines, '#DATE_RANGE')[:2]
        words = self._directive_words(lines, '#MIN_PHASE')
        self.min_phase = float(words[0])
        self.an_min_phase = words[1]

        # ---------- Observations (already made) section:
        obs_values = [line[len('#OBS'):].strip() for line in lines if line.upper().startswith('#OBS')]
        obs = [value.split() for value in obs_values]  # nested list

        # ---------- Ephemeris (minor) section:
        eph_dict_list = []
        i_eph_directive = None
        for i, line in enumerate(lines):
            if line.upper().startswith('#EPHEMERIS'):
                i_eph_directive = i
                break
        if (not lines[i_eph_directive + 1].strip().startswith(EPH_REQUIRED_HEADER_START)) or\
            (not (lines[i_eph_directive + 2].startswith('----'))):
            print(' >>>>> ERROR: ' + mpfile_name +
                  ':  MPEC header doesn\'t match expected from minorplanet.info page.')
            self.is_valid = False
            return
        eph_lines = lines[i_eph_directive + 3:]
        for line in eph_lines:
            eph_dict = dict()
            # words = line.split()
            # eph_dict['DateString'] = words[0]
            # date_parts = words[0].split('-')
            # eph_dict['Datetime'] = datetime(year=int(date_parts[0]),
            #                                 month=int(date_parts[1]),
            #                                 day=int(date_parts[2]))
            # eph_dict['RA'] = 15.0 * (float(words[1]) + float(words[2]) / 60.0 + float(words[3]) / 3600.0)
            # dec_sign = -1 if words[4].startswith('-') else 1.0
            # dec_abs_value = abs(float(words[4])) + float(words[5]) / 60.0 + float(words[6]) / 3600.0
            # eph_dict['Dec'] = dec_sign * dec_abs_value
            # eph_dict['V_mag'] = float(words[7])
            # eph_dict['Delta'] = float(words[8])          # earth-MP, in AU
            # eph_dict['R'] = float(words[9])              # sun-MP, in AU
            # eph_dict['Phase'] = float(words[10])         # phase angle, in degrees
            # eph_dict['Elong'] = float(words[11])         # from sun, in degrees
            # eph_dict['PAB_longitude'] = float(words[14])  # phase angle bisector longitude, in degrees
            # eph_dict['PAB_latitude'] = float(words[15])   # phase angle bisector latitude, in degrees
            # # Motion and direction not given by minorplanet.info, must be merged in from MPC. Ugh.
            # eph_dict['Gal_longitude'] = float(words[18])
            # eph_dict['Gal_latitude'] = float(words[19])
            eph_dict_list.append(eph_dict)
        self.eph_dict_list = eph_dict_list
        self.df_eph = pd.DataFrame(data=eph_dict_list)
        self.df_eph.index = self.df_eph
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

    def _write_new_file(self):
        # eph_lines = ['#EPHEMERIS',
        #              '; ================= For MP ' + str(mp_number) +
        #              ', retrieved from minorplanet.info One Asteroid Info '
        #              '{:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))]
        pass

    # This is from a previous version from class MPfile, which read ephem data from projectpluto.com.
    # Replaced 20200204 by code to read (30 days at a time, merged) from minorplanet.info.
    # # ---------- MPEC (projectpluto.com) section:
    # mpec_dict_list = []
    # i_mpec_directive = None
    # for i, line in enumerate(lines):
    #     if line.upper().startswith('#MPEC'):
    #         i_mpec_directive = i
    #         break
    # if not lines[i_mpec_directive + 1].startswith('Ephemerides'):
    #     print(' >>>>> ERROR: ' + mpfile_name + ':  MPEC section appears to be missing')
    #     self.is_valid = False
    #     return
    # if (not lines[i_mpec_directive + 2].startswith(MPEC_REQUIRED_HEADER_START)) or\
    #     (not (lines[i_mpec_directive + 3].startswith('----'))):
    #     print(' >>>>> ERROR: ' + mpfile_name +
    #           ':  MPEC header wrong (wrong options selected when downloading MPEC?)')
    #     self.is_valid = False
    #     return
    # mpec_lines = lines[i_mpec_directive + 4:]
    # for line in mpec_lines:
    #     mpec_dict = dict()
    #     words = line.split()
    #     mpec_dict['Date_utc'] = datetime(year=int(words[0]), month=int(words[1]), day=int(words[2]))
    #     mpec_dict['RA'] = 15.0 * (float(words[3]) + float(words[4]) / 60.0 + float(words[5]) / 3600.0)
    #     mpec_dict['Dec'] = float(words[6]) + float(words[7]) / 60.0 + float(words[8]) / 3600.0
    #     mpec_dict['Delta'] = float(words[9])           # earth-MP, in AU
    #     mpec_dict['R'] = float(words[10])              # sun-MP, in AU
    #     mpec_dict['Elong'] = float(words[11])          # from sun, in degrees
    #     mpec_dict['Phase'] = float(words[12])          # degrees
    #     mpec_dict['PAB_longitude'] = float(words[13])  # "
    #     mpec_dict['PAB_latitude'] = float(words[14])   # "
    #     mpec_dict['V_mag'] = float(words[15])
    #     mpec_dict['Motion'] = float(words[16])      # arcseconds per minute (0.50 normal)
    #     mpec_dict['Motion_dir'] = float(words[17])  # motion direction, deg eastward from north
    #     mpec_dict_list.append(mpec_dict)
    # self.mpec_dict_list = mpec_dict_list
    # self.df_mpec = pd.DataFrame(data=mpec_dict_list)
    # self.is_valid = True

    # Previous code, superseded by get_minorplanet_info_eph().
    # def get_mpec_eph_lines(mp_number, an_start, days, mpc_code='V16'):
    #     """ Return ephemeris-only text of projectpluto.com's pseudo-MPEC web service for one MP."""
    #     s = str(an_start)
    #     data_dict = {'obj_name': str(mp_number),
    #                  'year': ' '.join([s[0:4], s[5:6], s[7:8]]),
    #                  'n_steps': str(days),
    #                  'stepsize': '1',
    #                  'mpc_code': mpc_code,
    #                  'faint_limit': '20', 'ephem_type': '0', 'alt_az': 'on', 'phase': 'on', 'pab': '0',
    #                  'motion': '1', 'element_center': '-2', 'epoch': 'default', 'resids': '0', 'language': 'e'}
    #     r = requests.post('https://www.projectpluto.com/cgi-bin/fo/fo_serve.cgi', data=data_dict)
    #     in_lines = r.text.split('\n')
    #     out_lines = ['; ================= Retrieved from projectpluto.com '
    #                  '{:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))]
    #     header_line_found = False
    #     for line in in_lines:
    #         if not header_line_found:
    #             if line.startswith('<a name=\"eph\">'):
    #                 idx_start = line.find('<b>Ephem', 60) + 3
    #                 idx_end = line.find('</b></a>', idx_start)
    #                 header_line = line[idx_start:idx_end]
    #                 out_lines.append(header_line)
    #                 header_line_found = True
    #         else:
    #             if not line.startswith('</pre>'):
    #                 out_lines.append(line)
    #     return '\n'.join(out_lines)

