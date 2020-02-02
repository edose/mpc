__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
from datetime import datetime, timezone, timedelta

import pandas as pd
from astroquery.mpc import MPC

from mpc.mp_astrometry import calc_exp_time
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


def exp_time_from_V(v_mag):
    """  Given V mag, return *Clear* filter exposure time suited to lightcurve photometry.
    :param v_mag: target V magnitude [float]
    :return: suitable exposure time in Clear filter suited to lightcurve photometry. [float]
    """
    return calc_exp_time(v_mag, EXP_TIME_TABLE_PHOTOMETRY)


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

        # ---------- MPEC (projectpluto.com) section:
        mpec_dict_list = []
        i_mpec_directive = None
        for i, line in enumerate(lines):
            if line.upper().startswith('#MPEC'):
                i_mpec_directive = i
                break
        if not lines[i_mpec_directive + 1].startswith('Ephemerides'):
            print(' >>>>> ERROR: ' + mpfile_name + ':  MPEC section appears to be missing')
            self.is_valid = False
            return
        if (not lines[i_mpec_directive + 2].startswith(MPEC_REQUIRED_HEADER_START)) or\
            (not (lines[i_mpec_directive + 3].startswith('----'))):
            print(' >>>>> ERROR: ' + mpfile_name +
                  ':  MPEC header wrong (wrong options selected when downloading MPEC?)')
            self.is_valid = False
            return
        mpec_lines = lines[i_mpec_directive + 4:]
        for line in mpec_lines:
            mpec_dict = dict()
            words = line.split()
            mpec_dict['Date_utc'] = datetime(year=int(words[0]), month=int(words[1]), day=int(words[2]))
            mpec_dict['RA'] = 15.0 * (float(words[3]) + float(words[4]) / 60.0 + float(words[5]) / 3600.0)
            mpec_dict['Dec'] = float(words[6]) + float(words[7]) / 60.0 + float(words[8]) / 3600.0
            mpec_dict['Delta'] = float(words[9])           # earth-MP, in AU
            mpec_dict['R'] = float(words[10])              # sun-MP, in AU
            mpec_dict['Elong'] = float(words[11])          # from sun, in degrees
            mpec_dict['Phase'] = float(words[12])          # degrees
            mpec_dict['PAB_longitude'] = float(words[13])  # "
            mpec_dict['PAB_latitude'] = float(words[14])   # "
            mpec_dict['V_mag'] = float(words[15])
            mpec_dict['Motion'] = float(words[16])      # arcseconds per minute (0.50 normal)
            mpec_dict['Motion_dir'] = float(words[17])  # motion direction, deg eastward from north
            mpec_dict_list.append(mpec_dict)
        self.mpec_dict_list = mpec_dict_list
        self.df_mpec = pd.DataFrame(data=mpec_dict_list)
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
