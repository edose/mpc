__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os
from datetime import datetime, timezone, timedelta
from math import ceil, floor, sqrt

import pandas as pd
# from astroquery.mpc import MPC
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches

from mpc.mp_astrometry import calc_exp_time, PAYLOAD_DICT_TEMPLATE, get_one_html_from_list
from photrix.user import Astronight
from photrix.util import degrees_as_hex, ra_as_hours, RaDec, datetime_utc_from_jd, jd_from_datetime_utc,\
    hhmm_from_datetime_utc

# MP_ASTROMETRY_PLANNING:
MIN_MP_ALTITUDE = 30  # degrees
MIN_MOON_DISTANCE = 45  # degrees
DSW = ('254.34647d', '35.11861269964489d', '2220m')
DSNM = ('251.10288d', '31.748657576406853d', '1372m')
EXP_TIME_TABLE_PHOTOMETRY = [(13, 60), (14, 80), (15, 160), (16, 300)]  # (v_mag, exp_time sec), phot only.
EXP_OVERHEAD = 20  # Nominal exposure overhead, in seconds.
MIN_OBSERVABLE_MINUTES = 40  # in minutes

MPFILE_DIRECTORY = 'C:/Dev/Photometry/MPfile'
ACP_PLANNING_TOP_DIRECTORY = 'C:/Astro/ACP'
MP_PHOTOMETRY_PLANNING_DIRECTORY = 'C:/Astro/MP Photometry/$Planning'
CURRENT_MPFILE_VERSION = '1.0'

MAIN_WORKFLOW_____________________________________________________________ = 0


def plan(an_string, site_name='DSW', min_moon_dist=MIN_MOON_DISTANCE):
    """ Main planning workflow for MP photometry. Requires a
    :param an_string: Astronight, e.g. 20200201 [string or int]
    :param site_name: name of site for Site object. [string]
    :param min_moon_dist: min dist from min (degrees) to consider MP observable [float].
    :return: [None]
    """
    # Make and print table of values, 1 line/MP, sorted by earliest observable UTC:
    df_an_table = make_df_an_table(an_string, site_name='DSW', min_moon_dist=min_moon_dist)
    if df_an_table is None:
        print('No MPs observable for AN', an_string + '.')
        return
    df = df_an_table.copy()
    lines = ['MP Photometry planning for AN ' + an_string + ':',
             ''.rjust(19) + 'Start Tran  End    V   Exp/s  Duty    P/hr']
    for i in df.index:
        duty_cycle_string = '  [na]' if df.loc[i, 'DutyCyclePct'] is None \
            else str(round(df.loc[i, 'DutyCyclePct'])).rjust(5) + '%'
        period_string = '   [na]' if df.loc[i, 'Period'] is None \
            else '{0:7.2f}'.format(df.loc[i, 'Period'])
        line_elements = [df.loc[i, 'MPnumber'].rjust(6),
                         df.loc[i, 'MPname'].ljust(12),
                         hhmm_from_datetime_utc(df.loc[i, 'StartUTC']),
                         hhmm_from_datetime_utc(df.loc[i, 'TransitUTC']),
                         hhmm_from_datetime_utc(df.loc[i, 'EndUTC']),
                         '{0:5.1f}'.format(df.loc[i, 'V_mag']),
                         str(int(round(df.loc[i, 'ExpTime']))).rjust(5),
                         duty_cycle_string,
                         period_string,
                         '  ' + df.loc[i, 'PhotrixPlanning']]
        lines.append(' '.join(line_elements))
    print('\n'.join(lines))

    # Make ACP AN directory if doesn't exist, write text file to it:
    text_file_directory = os.path.join(ACP_PLANNING_TOP_DIRECTORY, 'AN' + an_string)
    os.makedirs(text_file_directory, exist_ok=True)
    text_filename = 'MP_table_' + an_string + '.txt'
    text_file_fullpath = os.path.join(text_file_directory, text_filename)
    with open(text_file_fullpath, 'w') as this_file:
        this_file.write('\n'.join(lines))

    # Display plots; also write to PNG files:
    make_coverage_plots(an_string, site_name, df_an_table)


def make_df_an_table(an_string, site_name='DSW', min_moon_dist=MIN_MOON_DISTANCE):
    """  Make dataframe of one night's MP photometry planning data, one row per MP.
         USAGE: df = make_df_an_table('20200201')
    :param an_string: Astronight, e.g. 20200201 [string or int]
    :param site_name: name of site for Site object. [string]
    :param min_moon_dist: min dist from min (degrees) to consider MP observable [float].
    :return: table of planning data, one row per current MP, many columns including one for
                           coverage list of dataframes. [list of DataFrames]
    """
    an_string = str(an_string)  # (precaution in case int passed in)
    an_object = Astronight(an_string, site_name)
    # dark_start, dark_end = an_object.ts_dark.start, an_object.ts_dark.end
    mid_dark = an_object.local_middark_utc
    # dark_no_moon_start, dark_no_moon_end = an_object.ts_dark_no_moon.start, an_object.ts_dark_no_moon.end
    mpfile_dict = make_mpfile_dict()

    # Nested function:
    def get_eph_for_utc(mpfile, datetime_utc, min_moon_dist=MIN_MOON_DISTANCE):
        """ Interpolate data from mpfile object's ephemeris; return dict and status string.
            Current code requires that ephemeris line spacing spacing = 1 day.
        :param mpfile: MPfile filename of MP in question. [string]
        :param datetime_utc: target utc date and time. [python datetime object]
        :param min_moon_dist: min dist from min (degrees) to consider MP observable [float].
        :return: dict of results specific to this MP and datetime, status string 'OK' or other
                 (2-tuple of dict and string)
        """
        #
        mpfile_first_date_utc = mpfile.eph_dict_list[0]['DatetimeUTC']
        index = (datetime_utc - mpfile_first_date_utc).total_seconds() / 24 / 3600
        if index < 0:
            return None, ' >>>>> Error: Requested datetime before mpfile ephemeris.'
        if index >= len(mpfile.eph_dict_list):
            return None, ' >>>>> Error: Requested datetime after mpfile ephemeris.'
        return_dict = dict()
        i_low = int(floor(index))  # line in ephemeris just previous to target datetime.
        # i_high = int(ceil(index))
        fract = index - i_low  # fraction of timespan after previous line.
        for k in mpfile.eph_dict_list[0].keys():
            value_before, value_after = mpfile.eph_dict_list[i_low][k], mpfile.eph_dict_list[i_low + 1][k]
            # Add interpolated value if not a string;
            #    (use this calc form, because you can subtract but not add datetime objects):
            if isinstance(value_before, datetime) or isinstance(value_before, float):
                return_dict[k] = value_before + fract * (value_after - value_before)  # interpolated value.
        return return_dict, 'OK'

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
        for i in range(2):
            data, status = get_eph_for_utc(mpfile, best_utc, min_moon_dist=min_moon_dist)
            an_dict['Status'] = status
            if status.upper() != 'OK':
                an_dict_list.append(an_dict)
                break
            mp_radec = RaDec(data['RA'], data['Dec'])
            ts_observable = an_object.ts_observable(mp_radec,
                                                    min_alt=MIN_MP_ALTITUDE,
                                                    min_moon_dist=min_moon_dist)  # Timespan object
            mid_observable = ts_observable.midpoint  # for loop exit
            best_utc = mid_observable  # update for loop continuation.
        data, status = get_eph_for_utc(mpfile, best_utc, min_moon_dist=min_moon_dist)  # data we will use.
        if ts_observable.seconds / 60.0 < MIN_OBSERVABLE_MINUTES:
            status = '(observable for only ' + str(int(ts_observable.seconds / 60.0)) + ' minutes)'
        if status.upper() == 'OK':
            an_dict['RA'] = data['RA']
            an_dict['Dec'] = data['Dec']
            an_dict['StartUTC'] = ts_observable.start
            an_dict['EndUTC'] = ts_observable.end
            an_dict['TransitUTC'] = an_object.transit(mp_radec)
            an_dict['V_mag'] = data['V_mag']
            an_dict['ExpTime'] = float(round(float(calc_exp_time(an_dict['V_mag'],
                                                                 EXP_TIME_TABLE_PHOTOMETRY))))
            if an_dict['Period'] is not None:
                # Duty cycle is % of time spent observing this MP if one exposure per 1/60 of period.
                an_dict['DutyCyclePct'] = 100.0 * ((an_dict['ExpTime'] + EXP_OVERHEAD) / 60.0) / \
                                          an_dict['Period']
            else:
                an_dict['DutyCyclePct'] = None
            an_dict['PhotrixPlanning'] = 'IMAGE MP_' + mpfile.number + \
                                         '  Clear=' + str(an_dict['ExpTime']) + 'sec(***)  ' + \
                                         ra_as_hours(an_dict['RA']) + ' ' + \
                                         degrees_as_hex(an_dict['Dec'], seconds_decimal_places=0)
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
    df_an_table = df_an_table.sort_values(by='StartUTC')
    return df_an_table


def make_coverage_plots(an_string, site_name, df_an_table):
    """ Make Nobs-vs-UTC plots, one per MP, i.e., plots of phase coverage by previous nights' observations.
    :param an_string: Astronight, e.g. 20200201 [string or int]
    :param site_name: name of site for Site object. [string]
    :param df_an_table: the master planning table for one Astronight [pandas DataFrame].
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
    max_nobs_to_plot = 5  # max number of previous coverages (y-axis) to plot.
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
            for i_plot, this_mp in enumerate(mps_to_plot):
                i_col = i_plot % n_cols
                i_row = int(floor(i_plot / n_cols))
                ax = axes[i_row, i_col]
                ax_p = axes_p[i_row, i_col]
                subplot_title = 'MP ' + this_mp +\
                                '    {0:.1f} h'.format(df.loc[this_mp, 'Period']) +\
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

                # Plot PHASE coverage curve:
                x = (df.loc[this_mp, 'PhaseCoverage'])['Phase']
                y = (df.loc[this_mp, 'PhaseCoverage'])['PhaseCoverage']
                ax_p.plot(x, y, linewidth=3, alpha=1, color='darkgreen', zorder=+50)
                ax_p.fill_between(x, 0, y, facecolor=(0.83, 0.87, 0.83), zorder=+49)

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

                # Complete HOURLY coverage plot:
                ax.grid(b=True, which='major', axis='x', color='lightgray',
                        linestyle='dotted', zorder=-1000)
                ax.set_xlim(hours_dark_start, hours_dark_end)
                ax.set_ylim(0, max_nobs_to_plot)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0 / 6.0))
                x_transit = ((df.loc[this_mp, 'TransitUTC']) - utc_zero).total_seconds() / 3600.0
                ax.axvline(x=x_transit, color='lightblue', zorder=+40)

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
            plt.show()

            # Save HOURLY coverage plots:
            filename = 'MP_hourly_coverage_' + an_string + '_{0:02d}'.format(i_figure + 1) + '.png'
            mp_photometry_planning_fullpath = os.path.join(MP_PHOTOMETRY_PLANNING_DIRECTORY, filename)
            print('Saving hourly coverage to', mp_photometry_planning_fullpath)
            fig.savefig(mp_photometry_planning_fullpath)
            acp_planning_fullpath = os.path.join(ACP_PLANNING_TOP_DIRECTORY, 'AN' + an_string, filename)
            print('Saving hourly coverage to', acp_planning_fullpath)
            fig.savefig(acp_planning_fullpath)

            # Save PHASE coverage plots:
            filename = 'MP_phase_coverage_' + an_string + '_{0:02d}'.format(i_figure + 1) + '.png'
            mp_photometry_planning_fullpath = os.path.join(MP_PHOTOMETRY_PLANNING_DIRECTORY, filename)
            print('Saving phase coverage to', mp_photometry_planning_fullpath)
            fig_p.savefig(mp_photometry_planning_fullpath)
            acp_planning_fullpath = os.path.join(ACP_PLANNING_TOP_DIRECTORY, 'AN' + an_string, filename)
            print('Saving phase coverage to', acp_planning_fullpath)
            fig_p.savefig(acp_planning_fullpath)

SUPPORT_____________________________________________________________ = 0


def photometry_exp_time_from_v_mag(v_mag):
    """  Given V mag, return *Clear* filter exposure time suited to lightcurve photometry.
    :param v_mag: target V magnitude [float]
    :return: suitable exposure time in Clear filter suited to lightcurve photometry. [float]
    """
    return calc_exp_time(v_mag, EXP_TIME_TABLE_PHOTOMETRY)


def make_df_coverage(period, obs_jd_ranges, target_jd_ranges, resolution_minutes=10):
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


def make_df_phase_coverage(period, obs_jd_ranges, phase_entries=100):
    """ Construct high-res array for 1 MP describing how well all phases have previously been observed.
    :param period: MP lightcurve period, in hours. Required, else this function can't work. [float]
    :param obs_jd_ranges: start,end pairs of Julian Dates for previous obs, this MP.
        Typically obtained from an updated MPfile for that MP. [list of 2-tuples of floats]
    :param target_jd_ranges: start,end pair of JDs of proposed new observations.
        Presumably tonight's available observation timespan. [2-tuple or list of floats]
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


MPFILE_STUFF____________________________________________________ = 0


def make_mpfile(mp_number, utc_date_start=None, days=90, mpfile_directory=MPFILE_DIRECTORY):
    """ Make new MPfile text file for upcoming apparition.
    :param mp_number: MP's number, e.g., 7084. [int or string]
    :param utc_date_start: UTC (not AN) start date, e.g. '2020-02-01' or '20200201', default=today [string].
    :param days: number of days to include in ephemeris. [int]
    :param mpfile_directory: where to write file (almost always use default). [string]
    :return: [None]
    """
    mp_number = str(mp_number)
    s = str(utc_date_start).replace('-', '')
    datetime_start = datetime(year=int(s[0:4]), month=int(s[4:6]),
                              day=int(s[6:8])).replace(tzinfo=timezone.utc)
    days = max(days, 7)

    # Get strings from MPC (minorplanetcenter.com):
    n_days_per_call = 90
    n_calls = ceil(days / n_days_per_call)
    parameter_dict = PAYLOAD_DICT_TEMPLATE.copy()
    parameter_dict['TextArea'] = str(mp_number)
    parameter_dict['i'] = '1'  # interval between lines
    parameter_dict['u'] = 'd'  # units of interval; 'h' for hours, 'd' for days, 'm' for minutes
    parameter_dict['long'] = '-105.6'.replace("+", "%2B")   # DSW longitude in deg
    parameter_dict['lat'] = '+35.12'.replace("+", "%2B")  # DSW latitude in deg
    parameter_dict['alt'] = '2220'    # DSW elevation (MPC "altitude") in m
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
        print(mpfile_fullpath, 'written. \n   >>>>> Now please edit with name, period, other data.')


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
    """ One object contains all current-apparition data for one MP.
    Fields:
        .format_version [str, currently '1.0']
        .number: MP number [str representing an integer]
        .name: text name of MP, e.g., 'Dido' or '1952 TX'. [str]
        .apparition: identifier (usually year) of this apparition, e.g., '2020'. [str]
        .motive: special reason to do photometry, or 'Pet' if simply a favorite. [str]
        .period: expected rotational period, in hours. [float]
        .period_certainty: LCDB certainty code, e.g., '1' or '2-'. [str]
        .amplitude: expected amplitude, in magnitudes. [float]
        .priority: priority code, 0=no priority, 10=top priority, 6=normal. [int]
        .utc_range: first & last date within the ephemeris (not observations). [2-tuple of str]
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
        utc_range_strs = self._directive_words(lines, '#UTC_RANGE')[:2]
        # self.utc_range = [float(range) for range in utc_range_strs]
        self.utc_range = self._directive_words(lines, '#UTC_RANGE')[:2]
        self.utc_range = []
        for utc_str in utc_range_strs:
            year_str, month_str, day_str = tuple(utc_str.split('-'))
            utc_dt = datetime(int(year_str), int(month_str), int(day_str)).replace(tzinfo=timezone.utc)
            self.utc_range.append(utc_dt)

        # ---------- Observations (already made) section:
        obs_strings = [line[len('#OBS'):].strip() for line in lines if line.upper().startswith('#OBS')]
        obs_jd_range_strs = [value.split() for value in obs_strings]  # nested list of strings (not floats)
        self.obs_jd_ranges = []
        for range in obs_jd_range_strs:
            self.obs_jd_ranges.append([float(range[0]), float(range[1])])

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


ANCILLARY_only________________________________________________________ = 0


class KeplerObject:
    def __init__(self, epoch_ma, ap, long, incl, ecc, a):
        """
        :param epoch_ma: epoch of mean anomaly, in Terrestrial Time JD. [float]
        :param ap: argument of perihelion, in degrees. [float]
        :param long: longitude of ascending node, in degrees. [float]
        :param incl: inclination, in degrees. [float]
        :param ecc: eccentricity, dimensionless. [float]
        :param a: semi-major axis length, in AU. [float]
        """
        self.epoch_ma = epoch_ma
        self.ap = ap
        self.long = long
        self.incl = incl
        self.ecc = ecc
        self.a = a
        self.h = None  # H-G model reduced magnitude (placeholder value).
        self.g = 0.15  # H-G model phase factor (default value).
        self.name = ''


def laguerre_delta(func, funcp, funcpp):
    """ Returns Laguerre's method estimate the nearest root of a function that is nearly quadratic.
    Reputed to be more robust to starting estimates than simply solving quadratic formula.
    :param func: value of function at some x.
    :param funcp: first derivative of function at the same x.
    :param funcpp: second derivative of function at the same x.
    :return: Best estimate of x shift needed to get near y=0. Will probably require iteration.
    """
    g = funcp / func
    h = g * g - funcpp / func
    numerator1 = g + sqrt(2 * h - g * g)
    numerator2 = g - sqrt(2 * h - g * g)
    numerator = numerator1 if abs(numerator1) > abs(numerator2) else numerator2
    return - 2.0 / numerator


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
