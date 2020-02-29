__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
from datetime import datetime, timezone, timedelta
from math import ceil, floor

import pandas as pd
from astroquery.mpc import MPC
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches

from mpc.mp_astrometry import calc_exp_time, PAYLOAD_DICT_TEMPLATE, get_one_html_from_list
from photrix.user import Astronight
from photrix.util import degrees_as_hex, ra_as_hours, RaDec, datetime_utc_from_jd

CURRENT_PHOT_MPS_FULLPATH = 'C:/Astro/MP Photometry/$Planning/current_phot_mps.txt'
MIN_MP_ALTITUDE = 30  # degrees
MIN_MOON_DISTANCE = 45  # degrees
DSW = ('254.34647d', '35.11861269964489d', '2220m')
DSNM = ('251.10288d', '31.748657576406853d', '1372m')
EXP_TIME_TABLE_PHOTOMETRY = [(13, 40), (14, 60), (15, 120), (16, 240)]  # (v_mag, exp_time sec), phot only.
EXP_OVERHEAD = 20  # Nominal exposure overhead, in seconds.
MIN_OBSERVABLE_MINUTES = 40  # in minutes

MPFILE_DIRECTORY = 'C:/Dev/Photometry/MPfile'
CURRENT_MPFILE_VERSION = '1.0'
MPEC_REQUIRED_HEADER_START = 'Date (UTC)   RA              Dec         delta   r     elong  ' +\
                             'ph_ang   ph_ang_bisector   mag  \'/hr    PA'
EPH_REQUIRED_HEADER_START = 'Date         RA            Dec       Mag       E.D.     S.D.    Ph'\
                            '      E    Alt   Az    PABL    PABB     M Ph    ME    GL    GB'


def make_df_an_table(an_string, location=DSNM, do_coverage_plots=True):
    """  Make dataframe of one night's MP photometry planning data, one row per MP.
         USAGE: df = make_df_an_table('20200201')
    :param an_string: Astronight, e.g. 20200201 [string or int]
    :param location: Astropy-style location tuple (long, lat, elev). [3-tuple of strings]
    :param do_coverage_plots: True iff user wants Coverage Plots. [boolean]
    :return: table of planning data, one row per current MP, many columns including one for
                           coverage list of dataframes. [list of DataFrames]
    """
    an_string = str(an_string)  # (precaution in case int passed in)
    an_object = Astronight(an_string, 'DSNM')
    dark_start, dark_end = an_object.ts_dark.start, an_object.ts_dark.end
    mid_dark = an_object.local_middark_utc
    dark_no_moon_start, dark_no_moon_end = an_object.ts_dark_no_moon.start, an_object.ts_dark_no_moon.end
    mpfile_dict = make_mpfile_dict()

    # Nested function:
    def get_eph_for_utc(mpfile, datetime_utc):
        """ Interpolate data from mpfile object's ephemeris; return dict and status string.
        :param mpfile: MPfile filename of MP in question. [string]
        :param datetime_utc: target utc date and time. [python datetime object]
        :return: dict of results specific to this MP and datetime, status string 'OK' or other
                 (2-tuple of dict and string)
        """
        #
        mpfile_first_date_utc = mpfile.eph_dict_list[0]['DatetimeUTC']
        index = (datetime_utc - mpfile_first_date_utc).days
        if index < 0:
            return None, ' >>>>> Error: Requested datetime before mpfile ephemeris.'
        if index > len(mpfile.eph_dict_list):
            return None, ' >>>>> Error: Requested datetime after mpfile ephemeris.'
        return_dict = dict()
        i_low = int(floor(index))
        i_high = int(ceil(index))
        fract = int(index - i_low)
        for k in mpfile.eph_dict_list[0].keys():
            return_dict[k] = (1.0 - fract) * mpfile.eph_dict_list[i_low] +\
                             fract * mpfile.eph_dict_list[i_low]
        return return_dict, 'OK'

    an_dict_list = []  # results to be deposited here, to make a dataframe later.
    for mp in mpfile_dict.keys():
        mpfile = mpfile_dict[mp]
        # an_dict doesn't need to include defaults for case before or after mpfile ephemeris,
        #    because making the dataframe should put in NANs for missing keys anyway (check this later):
        an_dict = {'MPnumber': mpfile.number, 'MPname': mpfile.name, 'Motive': mpfile.motive,
                   'Priority': mpfile.priority, 'Period': mpfile.period}
        # Two iterations only:
        data, status, ts_observable, mp_radec = None, None, None, None  # keep stupid IDE happy.
        best_utc = mid_dark
        for i in range(2):
            data, status = get_eph_for_utc(mpfile, best_utc)
            an_dict['Status'] = status
            if status.upper() != 'OK':
                an_dict_list.append(an_dict)
                break
            mp_radec = RaDec(data['RA'], data['Dec'])
            ts_observable = an_object.ts_observable(mp_radec,
                                                    min_alt=MIN_MP_ALTITUDE,
                                                    min_moon_dist=MIN_MOON_DISTANCE)  # Timespan object
            mid_observable = ts_observable.midpoint  # for loop exit
            best_utc = mid_observable  # for loop continuation
        if ts_observable.seconds / 60.0 < MIN_OBSERVABLE_MINUTES:
            status = '(observable for only ' + str(int(ts_observable.seconds / 60.0)) + ' minutes)'
        if status.upper() == 'OK':
            an_dict['RA'] = data['RA']
            an_dict['Dec'] = data['Dec']
            an_dict['StartUTC'] = ts_observable.start
            an_dict['EndUTC'] = ts_observable.end
            an_dict['TransitUTC'] = an_object.transit(mp_radec)
            an_dict['V_mag'] = data['V_mag']
            an_dict['ExpTime'] = int(calc_exp_time(an_dict['V_mag'], EXP_TIME_TABLE_PHOTOMETRY))
            if an_dict['Period'] is not None:
                # Duty cycle is % of time spent observing this MP if one exposure per 1/60 of period.
                an_dict['DutyCyclePct'] = 100.0 * ((an_dict['ExpTime'] + EXP_OVERHEAD) / 60.0) / \
                                          an_dict['Period']
            else:
                an_dict['DutyCyclePct'] = None
            an_dict['PhotrixPlanning'] = 'IMAGE MP_' + mpfile.number + \
                                         '  Clear=' + str(an_dict['ExpTime']) + 'sec(1)  ' + \
                                         ra_as_hours(an_dict['RA']) + ' ' + \
                                         degrees_as_hex(an_dict['Dec'])
            if an_dict['Period'] is not None:
                an_dict['Coverage'] = make_df_coverage(an_dict['Period'], mpfile.obs_jds,
                                                       (an_dict['StartUTC'], an_dict['EndUTC']))
            else:
                an_dict['Coverage'] = None
            an_dict_list.append(an_dict)
    df_an_table = pd.DataFrame(data=an_dict_list)
    df_an_table.index = df_an_table['MPName'].values

    if do_coverage_plots:
        make_coverage_plots(an_object, df_an_table)
    return df_an_table


def make_coverage_plots(an_string, an_object, df):
    """ Make N-vs-UTC plots, one per MP, of phase coverage by previous nights' observations.
    :param an_string: astronight identifier (e.g., '20200201') for night being planned. [string]
    :param an_object: Astronight object for the night being planned. [Astronight object]
    :param df: df_an_table, the master planning table for one Astronight [pandas DataFrame].
    :return: None [makes plots 3 x 3 per Figure (page)].
    """
    # Nested functions:
    def make_labels_9_subplots(ax, title, xlabel, ylabel, text='', zero_line=True):
        ax.set_title(title, loc='center', pad=-3)  # pad in points
        ax.set_xlabel(xlabel, labelpad=-29)  # labelpad in points
        ax.set_ylabel(ylabel, labelpad=-5)  # "
        ax.text(x=0.5, y=0.95, s=text,
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if zero_line is True:
            ax.axhline(y=0, color='lightgray', linewidth=1, zorder=-100)

    def draw_y_line(ax, y_value, color='lightgray'):
        ax.axhline(y=y_value, color=color, linewidth=1, zorder=-100)

    def draw_x_line(ax, x_value, color='lightgray'):
        ax.axvline(x=x_value, color=color, linewidth=1, zorder=-100)

    max_nobs_to_plot = 5
    mps_to_plot = [name for (name, cov) in zip(df['MPName'], df['Coverage']) if cov is not None]
    n_plots = len(mps_to_plot)
    n_cols, n_rows = 3, 3
    n_plots_per_figure = n_cols * n_rows
    n_figures = ceil(n_plots / n_plots_per_figure)
    dark_start, dark_end = an_object.ts_dark.start, an_object.ts_dark.end

    for i_figure in range(n_figures):
        n_plots_remaining = n_plots - (i_figure * n_plots_per_figure)
        n_plots_this_figure = min(n_plots_remaining, n_plots_per_figure)
        if n_plots_this_figure >= 1:
            # Start new Figure:
            fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(15, 9))
            fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
            fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
            fig.suptitle('MP planning for AN ' + an_string + '     ::      Page ' +
                         str(i_figure + 1) + ' of ' + str(n_figures),
                         color='darkblue', fontsize=20)
            fig.canvas.set_window_title('MP planning for AN ' + an_string)
            subplot_text = 'rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
            fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=12,
                     color='dimgray')
            for i_plot in range(n_plots_this_figure):
                this_mp = mps_to_plot[i_plot]
                i_col = i_plot % n_cols
                i_row = int(floor(i_plot / n_cols))
                ax = axes[i_row, i_col]
                make_labels_9_subplots(ax, 'MP_' + this_mp + '  AN ' + an_string,
                                       'UTC', 'mMag', '(text here)', zero_line=False)
                # Make left box if any unavailable timespan before available timespan:
                left_box_utc_start = dark_start
                left_box_utc_end = df.loc[this_mp, 'StartUTC']
                if left_box_utc_end > left_box_utc_start:
                    ax.add_patch(patches.Rectangle((left_box_utc_start, max_nobs_to_plot),
                                                   left_box_utc_end - left_box_utc_start,
                                                   max_nobs_to_plot,
                                                   linewidth=1, alpha=0.4, zorder=+100,
                                                   edgecolor='gray', facecolor='lightgray'))
                # Make right box if any unavailable timespan after available timespan:
                right_box_utc_start = df.loc[this_mp, 'EndUTC']
                right_box_utc_end = dark_end
                if right_box_utc_end > right_box_utc_start:
                    ax.add_patch(patches.Rectangle((right_box_utc_start, max_nobs_to_plot),
                                                   right_box_utc_end - right_box_utc_start,
                                                   max_nobs_to_plot,
                                                   linewidth=1, alpha=0.4, zorder=+100,
                                                   edgecolor='gray', facecolor='lightgray'))
                datetime_values = (df.loc[this_mp, 'Coverage'])['DateTimeUTC']
                dt0 = datetime_values.iloc[0]
                utc_zero = datetime(year=dt0.year, month=dt0.month, day=dt0.day)
                x = [(dt - utc_zero).total_seconds() / 3600.0 for dt in datetime_values]
                y = (df.loc[this_mp, 'Coverage'])['Nobs']
                ax.plot(x, y, linewidth=5, alpha=1, color='blue', zorder=+50)
                ax.fill_between(x, 0, y, facecolor='lightblue')
                x_transit = ((df.loc[this_mp, 'Coverage'])['DateTimeUTC'] - utc_zero) / 3600.0
                draw_x_line(ax, x_transit)
                ax.set_xlim(dark_start, dark_end)
                ax.set_ylim(0, max_nobs_to_plot)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0 / 6.0))
                # Remove any empty subplots from this (last) Figure:
                for i_plot in range(n_plots_this_figure, n_plots_per_figure):
                    i_col = i_plot % n_cols
                    i_row = int(floor(i_plot / n_cols))
                    ax = axes[i_row, i_col]
                    ax.remove()
        plt.show()


def photometry_exp_time_from_v_mag(v_mag):
    """  Given V mag, return *Clear* filter exposure time suited to lightcurve photometry.
    :param v_mag: target V magnitude [float]
    :return: suitable exposure time in Clear filter suited to lightcurve photometry. [float]
    """
    return calc_exp_time(v_mag, EXP_TIME_TABLE_PHOTOMETRY)


def make_df_coverage(period, obs_jds, target_jds, resolution_minutes=10):
    """ Construct high-resolution array describing how well tonight's phases have previously been observed.
    :param period: MP lightcurve period, in hours. [float]
    :param obs_jds: start,end pairs of Julian Dates for previous obs, this MP. [list of 2-tuples of floats]
    :param target_jds: start,end pair of JDs of proposed new observations. [2-tuple or list of floats]
    :param resolution_minutes: approximate time resolution of output dataframe, in minutes. [float]
    :return: 1 row / timepoint in new obs window, columns = JD, DateTimeUTC, Phase, Nobs. [pandas DataFrame]
    """
    # First, accumulate coverage by MP phase only:
    raw_n_phase_array = int(ceil(period * 60.0 / resolution_minutes))
    n_phase_array = min(max(raw_n_phase_array, 100), 1000)
    phase_array = n_phase_array * [0]
    phase_zero_jd = min([float(obs_jd[0]) for obs_jd in obs_jds])  # earliest obs JD
    for obs_jd in obs_jds:
        raw_phase_start = (float(obs_jd[0]) - phase_zero_jd) * 24 / period  # jd in days, period in hours.
        raw_phase_end = (float(obs_jd[1]) - phase_zero_jd) * 24 / period    # "
        phase_floor = floor(raw_phase_start)
        i_phase_start = int(round((raw_phase_start - phase_floor) * n_phase_array))
        i_phase_end = int(round((raw_phase_end - phase_floor) * n_phase_array))
        for i in range(i_phase_start, i_phase_end + 1):
            i_to_increment = i % n_phase_array
            phase_array[i_to_increment] += 1

    # Now, propagate the phase-coverage array to a time-coverage array:
    n_time_array = ceil((target_jds[1] - target_jds[0]) * 24 * 60 / resolution_minutes) + 1
    actual_resolution_days = (target_jds[1] - target_jds[0]) / (n_time_array - 1)
    jd_array = [target_jds[0] + i * actual_resolution_days for i in range(n_time_array)]
    target_phase_array = [((jd - phase_zero_jd) * 24 / period) % 1 for jd in jd_array]
    phase_index_array = [int((phase * n_phase_array) % n_phase_array) for phase in target_phase_array]
    time_array = [phase_array[i] for i in phase_index_array]

    # Make dataframe:
    dt_array = [datetime_utc_from_jd(jd) for jd in jd_array]
    df_coverage = pd.DataFrame({'JD': jd_array, 'DateTimeUTC': dt_array, 'Phase': target_phase_array,
                                'Nobs': time_array})
    return df_coverage


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
    # utc_start = '-'.join([s[0:4], s[5:6], s[7:8]])
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
        self.apparition = self._directive_value(lines, '#APPARITION')
        self.motive = self._directive_value(lines, '#MOTIVE')
        words = self._directive_words(lines, '#PERIOD')
        if words is not None:
            try:
                self.period = float(words[0])
            except ValueError:
                print(' >>>>> Error: Period present but incorrect. (MP=' + self.number + ')')
                self.period = None
            if len(words) >= 2:
                self.period_certainty = words[1]
            else:
                self.period_certainty = '?'
        amplitude_string = self._directive_value(lines, '#AMPLITUDE')
        if amplitude_string is None:
            print(' >>>>> Warning: Amplitude is missing. (MP=' + self.number + ')')
            self.amplitude = None
        else:
            try:
                self.amplitude = float(amplitude_string)
            except ValueError:
                print(' >>>>> Error: Amplitude present but incorrect. (MP=' + self.number + ')')
                self.amplitude = None
        priority_string = self._directive_value(lines, '#PRIORITY')
        try:
            self.priority = int(priority_string)
        except ValueError:
            print(' >>>>> Error: Priority present but incorrect. (MP=' + self.number + ')')
            self.priority = None
        self.utc_range = self._directive_words(lines, '#UTC_RANGE')[:2]

        # ---------- Observations (already made) section:
        obs_strings = [line[len('#OBS'):].strip() for line in lines if line.upper().startswith('#OBS')]
        self.obs_jds = [value.split() for value in obs_strings]  # nested list of strings (not floats)

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
            eph_dict['DateUTC'] = words[0]
            date_parts = words[0].split('-')
            eph_dict['Datetime'] = datetime(year=int(date_parts[0]),
                                            month=int(date_parts[1]),
                                            day=int(date_parts[2]))
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
        self.df_eph.index = self.df_eph['DateUTC'].values
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


# def get_eph(mp, an, locan dftion='V28'):
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