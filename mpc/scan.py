__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

from collections import OrderedDict
import os
from random import randint
from time import sleep
from webbrowser import open_new_tab
from math import sqrt, log, exp
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

MPC_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAYLOAD_DICT_TEMPLATE = OrderedDict([
    ('ty', 'e'),  # e = 'Return Ephemerides'
    ('TextArea', ''),  # the MP IDs
    ('d', '20181122'),  # first date
    ('l', '28'),  # number of dates/times (str of integer)
    ('i', '30'),  # interval between ephemerides (str of integer)
    ('u', 'm'),  # units of interval; 'h' for hours, 'd' for days, 'm' for minutes
    ('uto', '0'),  # UTC offset in hours if u=d
    ('long', '-107.55'),  # longitude in deg; make plus sign safe in code below
    ('lat', '+35.45'),  # latitude in deg; make plus sign safe in code below
    ('alt', '2200'),  # altitude in m
    ('raty', 'a'),  # 'a' = full sexigesimal, 'x' for decimal degrees
    ('s', 't'),  # N/A (total motion and direction)
    ('m', 'm'),  # N/A (motion in arcsec/minute)
    ('igd', 'y'),  # 'y' = suppress line if sun up
    ('ibh', 'y'),  # 'y' = suppress line if MP down
    ('adir', 'S'),  # N/A
    ('oed', ''),  # N/A (display elements)
    ('e', '-2'),  # N/A (no elements output)
    ('resoc', ''),  # N/A (residual blocks)
    ('tit', ''),  # N/A (HTML title)
    ('bu', ''),  # N/A
    ('ch', 'c'),  # N/A
    ('ce', 'f'),  # N/A
    ('js', 'f')  # N/A
])
MAX_MP_PER_HTML = 100
MPC_URL_STUB = 'https://cgi.minorplanetcenter.net/cgi-bin/mpeph2.cgi/?'
GET_HEADER = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:64.0) Gecko/20100101 Firefox/64.0'}
MIN_TABLE_WORDS = 25  # any line with this many white-space-delimited words presumed an ephem table line.
MIN_MP_ALTITUDE = 40
MAX_SUN_ALTITUDE = -12
MAX_V_MAG = 19.0  # requires 3 x 5-min stack in R, but really this will be through Clear filter (not R).
MIN_UNCERTAINTY = 2.1  # minimum orbit uncertainty to consider following up (in arcseconds).
MIN_MOON_DIST = 45
DF_COLUMN_ORDER = ['number', 'name', 'score', 'code', 'last_obs', 'status', 'uncert', 'v_mag',
                   'motion', 'motion_pa', 'mp_alt', 'moon_phase', 'moon_alt',
                   'ra', 'dec', 'utc', 'comments', 'roster']

# PET_MPS = [(588, 'Achilles'),
#            (911, 'Agamemnon'),
#            (1404, 'Ajax'),
#            (209, 'Dido'),
#            (435, 'Ella'),
#            (4954, 'Eric'),
#            (23989, 'Farpoint'),
#            (2415, 'Ganesa'),
#            (1309, 'Hyperborea'),
#            (3124, 'Kansas'),
#            (2278, 'Pannekoek'),
#            (6480, 'Scarlatti'),
#            (54439, 'Topeka'),
#            (2554, 'Skiff')]

PET_KEYWORDS = ['farpoint', 'eskridge', 'hug', 'dose', 'sandlot']

UNCERTAINTY_TARGET = 0.5  # arcseconds below which uncertainty is presumed not to be reduced
TARGET_OVERHEAD = 60  # seconds to start new target
IMAGE_OVERHEAD = 19  # seconds to start, download, solve new image
PROCESSING_OVERHEAD = 300  # penalty (seconds) for processing data from one target
EXP_TIME_TABLE = [(16, 120), (17, 240), (18, 420), (19, 600)]  # entry = (v_mag, exp_time in sec).
MP_FILTER_NAME = 'Clear'


def calc_exp_time(v_mag):
    # Check for outside limits:
    if v_mag < EXP_TIME_TABLE[0][0]:
        return EXP_TIME_TABLE[0][1]
    n = len(EXP_TIME_TABLE)
    # Check for equals an entry:
    if v_mag > EXP_TIME_TABLE[n-1][0]:
        return EXP_TIME_TABLE[n-1][1]
    for (v_mag_i, t_i) in EXP_TIME_TABLE:
        if v_mag == v_mag_i:
            return t_i
    # Usual case: linear interpolation in mag (& thus in log(i)):
    for i, entry in enumerate(EXP_TIME_TABLE[:-1]):
        v_mag_i, t_i = entry
        v_mag_next, t_next = EXP_TIME_TABLE[i + 1]
        if v_mag < v_mag_next:
            slope = log(t_next / t_i) / (v_mag_next - v_mag_i)
            log_t = log(t_i) + (v_mag - v_mag_i) * slope
            return exp(log_t)


def calc_score(v_mag, uncert):
    # benefit is roughly: improvement in arcsec uncertainty.
    # cost is roughly hours of scope+user time.
    benefit = uncert - UNCERTAINTY_TARGET
    cost_in_hours = (3 * (TARGET_OVERHEAD + 3 * (calc_exp_time(v_mag) + IMAGE_OVERHEAD)) +
                       PROCESSING_OVERHEAD) / 3600.0
    return benefit / cost_in_hours


MIN_SCORE = calc_score(v_mag=18, uncert=4)  # scores less than this do not get included.
# MIN_SCORE = 0  # for test only


def go(mp_list=None, mp_start=100000, date_utc=None, max_mps=10000, max_candidates=100):
    print('min score =', '{:.1f}'.format(MIN_SCORE))
    if date_utc is None:
        target_date = datetime.now(timezone.utc) + timedelta(days=1)
        date_string = '{0:04d}{1:02d}{2:02d}'.format(target_date.year, target_date.month, target_date.day)
    else:
        date_string = date_utc
    all_dict_list = []
    first_index_next_html = 0
    last_mp = mp_start + max_mps - 1
    while len(all_dict_list) < max_candidates:
        if mp_list is not None:
            mp_list_next_html = mp_list[first_index_next_html:][:100]
            n_mps_this_html = len(mp_list_next_html)
            if n_mps_this_html >= 1:
                print(str(n_mps_this_html) + ' MPs', end='', flush=True)
        else:
            n_mps_this_html = min(last_mp - (mp_start + first_index_next_html) + 1, MAX_MP_PER_HTML)
            mp_list_next_html = list(range(mp_start + first_index_next_html,
                                           mp_start + first_index_next_html + n_mps_this_html))
            if n_mps_this_html >= 1:
                print(str(mp_list_next_html[0]), str(mp_list_next_html[-1]), str(n_mps_this_html),
                    end='', flush=True)
        if n_mps_this_html <= 0:
            break
        lines = get_one_html_from_list(mp_list_next_html, date=date_string)
        html_dict_list = parse_html_lines(lines)
        # html_dict_list = mp_list_next_html[0:2]  # TEST ONLY
        all_dict_list.extend(html_dict_list)
        first_index_next_html += n_mps_this_html
        print(' --> ' + str(len(all_dict_list)))
    print('Done.')
    # This sort order in case LST zero comes in middle of night:
    df = pd.DataFrame(all_dict_list).reindex(columns=DF_COLUMN_ORDER).sort_values(by=['utc', 'ra'])
    return df


def parse_html_lines(lines):
    mp_dict_list = []
    mp_block_limits = chop_html(lines)
    for limits in mp_block_limits:
        mp_dict = extract_mp_data(lines, limits)
        if any([v is None for v in mp_dict.values()]):
            print(mp_dict)
        if mp_dict.get('v_mag', None) is not None and mp_dict.get('uncert', None) is not None:
            score = calc_score(float(mp_dict['v_mag']), float(mp_dict['uncert']))
            if score > MIN_SCORE:
                mp_dict_list.append(mp_dict)
                # print(mp_dict['number'])
                # print(mp_dict)
    return mp_dict_list


def get_one_html_contiguous(start=200000, n=MAX_MP_PER_HTML, date='20181122'):
    """ Gets MPC HTML text, returns list of strings """
    payload_dict = PAYLOAD_DICT_TEMPLATE.copy()

    # Construct TextArea field:
    mp_list = [str(i) for i in range(start, start + n)]
    return get_one_html_from_list(mp_list, date)


def get_one_html_from_list(mp_list=None, date='20181122'):
    """ Gets MPC HTML text, returns list of strings """
    payload_dict = PAYLOAD_DICT_TEMPLATE.copy()

    # Construct TextArea field:
    text_area = '%0D%0A'.join([str(mp) for mp in mp_list])
    payload_dict['TextArea'] = text_area

    payload_dict['d'] = date

    # Make longitude and latitude safe (from '+' characters)
    payload_dict['long'] = payload_dict['long'].replace("+", "%2B")
    payload_dict['lat'] = payload_dict['lat'].replace("+", "%2B")

    # Construct URL and header for GET call:
    payload_string = '&'.join([k + '=' + v for (k, v) in payload_dict.items()])
    url = MPC_URL_STUB + payload_string

    # Make GET call, parse return text.
    # print(url)
    r = requests.get(url, headers=GET_HEADER)
    return r.text.splitlines()


def get_html_from_file(fullpath='C:\Dev\mpc\mpc.html'):
    if os.path.exists(fullpath) and os.path.isfile(fullpath):
        with open(fullpath) as fov_file:
            lines = fov_file.readlines()
            return [line.rstrip() for line in lines]
    else:
        return []


def chop_html(html_lines):
    """Return list of start and end line numbers defining minor planet blocks of text within html_lines."""
    # Collect lines numbers for all vertical block delimiters (including end of file):
    hr_line_numbers = [0]
    for i_line, line in enumerate(html_lines):
        if '<hr>' in line:
            hr_line_numbers.append(i_line)
    hr_line_numbers.append(len(html_lines) - 1)

    # Make a block if MP data actually between two successive horizontal lines:
    mp_block_limits = []
    for i_hr_line in range(len(hr_line_numbers) - 2):
        for i_line in range(hr_line_numbers[i_hr_line], hr_line_numbers[i_hr_line + 1]):
            if html_lines[i_line].strip().lower().startswith('<p>discovery date'):
                mp_block_limits.append((hr_line_numbers[i_hr_line], hr_line_numbers[i_hr_line + 1]))
                break
    return mp_block_limits


def extract_mp_data(html_lines, mp_block_limits):
    """ Get data out of a HTML text block devoted to one MP.
    :param html_lines:
    :param mp_block_limits:
    :return: dict of MP data, keys=number, name, code, last_obs, status, v_mag, motion, mp_alt.
    """
    mp_dict = dict()
    mp_dict['comments'] = ''
    mp_is_pet = False

    # Handle one-per-MP items:
    for i_line in range(mp_block_limits[0], mp_block_limits[1]):
        line = html_lines[i_line]
        if line.strip().startswith('<b>'):
            mp_dict['number'] = line.split(')')[0].split('(')[-1].strip()
            mp_dict['name'] = line.split(')')[-1].split('<')[0].strip()
        if line.strip().startswith('Last observed on'):
            mp_dict['last_obs'] = line.strip()[len('Last observed on'):].strip().replace('.', '')
        if line.strip().startswith('<p><pre>'):
            if i_line + 1 < mp_block_limits[1]:
                mp_dict['code'] = html_lines[i_line + 1].strip().split()[0]
        if 'further observations?' in line.lower():
            mp_dict['status'] = ''
            if line.strip().split('>')[3].strip().lower().startswith('not necessary'):
                mp_dict['status'] = 'no'
            if line.strip().split('>')[3].strip().lower().startswith('useful'):
                mp_dict['status'] = 'USEFUL'
        if 'discovery site' in line.lower():
            this_site = line.strip().split(':', maxsplit=1)[1].strip()
            if this_site.lower() in PET_KEYWORDS:
                mp_dict['comments'] = mp_dict['comments'] + this_site + '; '
                mp_is_pet = True
        if 'discoverer(s)' in line.lower():
            discoverer = line.strip().split(':', maxsplit=1)[1].strip()
            if any([p in discoverer.lower() for p in PET_KEYWORDS]):
                mp_dict['comments'] = mp_dict['comments'] + discoverer + '; '
                mp_is_pet = True

    mp_dict_short = mp_dict.copy()

    # Handle items in ephemeris table. Use line with highest altitude:
    # Find line limits of ephemeris table (slight superset from <pre> and </pre> HTML tags:
    table_start_found = False
    max_mp_alt = None
    uncertainty_raw_url = ''  # to keep IDE happy
    for i_line in range(mp_block_limits[0], mp_block_limits[1]):
        line = html_lines[i_line]
        if '<pre>' in line:
            table_start_found = True
        if table_start_found:
            if '</pre>' in line:
                break
        line_split = line.split()
        if len(line_split) >= MIN_TABLE_WORDS:
            mp_alt = float(line.split()[18])
            v_mag = float(line_split[14])
            sun_alt = float(line_split[19])
            moon_dist = float(line_split[21])
            moon_alt = float(line_split[22])
            if (mp_alt >= MIN_MP_ALTITUDE and v_mag <= MAX_V_MAG and sun_alt <= MAX_SUN_ALTITUDE and
               (moon_dist >= MIN_MOON_DIST or moon_alt < 0)):
                if max_mp_alt is None:
                    this_mp_alt_is_max_so_far = True
                else:
                    this_mp_alt_is_max_so_far = (mp_alt > max_mp_alt)
                if this_mp_alt_is_max_so_far:
                    max_mp_alt = mp_alt
                    mp_dict['utc'] = ' '.join(line_split[0:4])
                    mp_dict['ra'] = ':'.join(line_split[4:7])
                    mp_dict['dec'] = ':'.join(line_split[7:10])
                    mp_dict['mp_alt'] = '{:d}'.format(round(mp_alt))
                    mp_dict['v_mag'] = '{:.1f}'.format(v_mag)
                    mp_dict['motion'] = line_split[15]
                    mp_dict['motion_pa'] = line_split[16]
                    mp_dict['moon_phase'] = line_split[20]
                    mp_dict['moon_alt'] = '{:d}'.format(round(moon_alt))
                    exp_time = calc_exp_time(v_mag)
                    mp_dict['roster'] = 'IMAGE MP_' + mp_dict['number'].zfill(6) + \
                                        '  ' + MP_FILTER_NAME + '={:.0f}'.format(exp_time) + 'sec(3)' + \
                                        '  ' + mp_dict['ra'] +\
                                        '  ' + mp_dict['dec']
                    uncertainty_raw_url = line_split[-1]
    if mp_dict['comments'] == '':
        mp_dict['comments'] = '-'
    if mp_dict.get('v_mag', None) is not None:
        uncertainty = get_uncertainty(uncertainty_raw_url)
        if uncertainty >= MIN_UNCERTAINTY:
            mp_dict['uncert'] = '{:.1f}'.format(uncertainty)
            mp_dict['score'] = '{:.1f}'.format(calc_score(float(mp_dict['v_mag']), uncertainty))
        else:
            mp_dict = mp_dict_short
    return mp_dict


def get_uncertainty(raw_url):
    url = raw_url.split('href=\"')[-1].split('&OC')[0]
    # print(url)
    lines = requests.get(url, headers=GET_HEADER).text.splitlines()
    pre_line_found = False
    uncertainties_2 = []
    for line in lines:
        if line.strip().lower().startswith('</pre>'):
            break
        if line.strip().lower().startswith('<pre>'):
            pre_line_found = True
            continue
        if pre_line_found:
            words = line.split()
            if len(words) >= 3 and 'variant orbit' in line.lower():
                this_uncertainty_2 = float(words[0])**2 + float(words[1])**2
                uncertainties_2.append(this_uncertainty_2)
    uncertainties_2.sort(reverse=True)  # in-place.
    return sqrt((uncertainties_2[0] + uncertainties_2[1])/2.0)


def t():
    lines = get_html_from_file()
    mp_block_limits = chop_html(lines)
    for limits in mp_block_limits:
        mp_dict = extract_mp_data(lines, limits)
        #if mp_dict.get('v_mag') is not None:
        print(mp_dict)


def web(start=200000, n=50, date='20181201'):
    lines = get_one_html_contiguous(start, n, date)
    mp_block_limits = chop_html(lines)
    data = []
    for limits in mp_block_limits:
        mp_dict = extract_mp_data(lines, limits)
        data.append(mp_dict)
        print(mp_dict['number'])
    for mp_dict in data:
        print(mp_dict)


if __name__ == "__main__":
    get_one_html_contiguous()

