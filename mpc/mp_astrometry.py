__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

from collections import OrderedDict
import os
from random import randint
from time import sleep
from webbrowser import open_new_tab
from math import sqrt, log, exp, ceil
from datetime import datetime, timezone, timedelta
import webbrowser

import requests
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan import Observer


MPC_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANNING_ROOT_DIRECTORY = 'C:/Astro/ACP'

MPC_HTML_MONTH_CODES = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

PAYLOAD_DICT_TEMPLATE = OrderedDict([
    ('ty', 'e'),  # e = 'Return Ephemerides'
    ('TextArea', ''),  # the MP IDs
    ('d', '20181122'),  # first utc date
    ('l', '28'),  # number of dates/times (str of integer)
    ('i', '30'),  # interval between ephemerides (str of integer)
    ('u', 'm'),  # units of interval; 'h' for hours, 'd' for days, 'm' for minutes
    ('uto', '0'),  # UTC offset in hours if u=d
    ('c', ''),   # observatory code
    ('long', '-107.55'),  # longitude in deg; make plus sign safe in code below
    ('lat', '+35.45'),  # latitude in deg; make plus sign safe in code below
    ('alt', '2200'),  # elevation (MPC "altitude") in m
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
MPC_URL_STUB = 'https://cgi.minorplanetcenter.net/cgi-bin/mpeph2.cgi'
GET_HEADER = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:64.0) Gecko/20100101 Firefox/64.0'}
MIN_TABLE_WORDS = 25  # any line with this many white-space-delimited words presumed an ephem table line.
MIN_MP_ALTITUDE = 40
MAX_SUN_ALTITUDE = -12
MAX_V_MAG = 19.0  # in Clear filter
MINIMUM_SCORING = {'v_mag': 18.2, 'uncert': 4}
MIN_UNCERTAINTY = 2.1  # minimum orbit uncertainty to consider following up (in arcseconds).
MIN_MOON_DIST = 45  # degrees
FORCE_INCLUDE_IN_YEARS = 2.0
DF_COLUMN_ORDER = ['number', 'score', 'transit', 'ACP', 'min9', 'uncert', 'v_mag',
                   'comments', 'last_obs', 'motion',
                   'name', 'code', 'status', 'motion_pa', 'mp_alt',
                   'moon_phase', 'moon_alt', 'ra', 'dec', 'utc']

PET_MPS = [(8900, 'AAVSO'),
           (588, 'Achilles'),
           (396, 'Aeolia'),
           (911, 'Agamemnon'),
           (1404, 'Ajax'),
           (1172, 'Aneas'),
           (29401, 'Asterix'),
           (24265, 'Banthonytwarog'),
           (12709, 'Bergen op Zoom'),
           (13053, 'Bertrandrussell'),
           (2689, 'Bruxelles'),
           (19521, 'Chaos'),
           (1991, 'Darwin'),
           (23257, 'Denny'),
           (209, 'Dido'),
           (55555, 'DNA'),
           (12410, 'Donald Duck'),
           (435, 'Ella'),
           (4954, 'Eric'),
           (23989, 'Farpoint'),
           (17473, 'Freddiemercury'),
           (121022, 'Galliano'),
           (2415, 'Ganesa'),
           (16452, 'Goldfinger'),
           (100027, 'Hannaharendt'),
           (33529, 'Henden'),
           (361450, 'Houellebecq'),
           (7470, 'Jabberwock'),
           (9007, 'James Bond'),
           (221150, ' Jerryfoote'),
           (6758, 'Jesseowens'),
           (4738, 'Jimihendrix'),
           (8794, 'Joepatterson'),
           (33165, 'Joschhambsch'),
           (3124, 'Kansas'),
           (2807, 'Karl Marx'),
           (3811, 'Karma'),
           (25594, 'Kessler'),
           (9563, 'Kitty'),
           (216, 'Kleopatra'),
           (10221, 'Kubrick'),
           (15072, 'Landolt'),
           (218692, 'Leesnyder'),
           (6735, 'Madhatter'),
           (228029, 'MANIAC'),
           (7447, 'Marcusaurelius'),
           (2362, 'Mark Twain'),
           (1647, 'Menelaus'),
           (367732, 'Mikesimonsen'),
           (3633, 'Mira'),
           (21651, 'Mission Valley'),
           (8889, 'Mockturtle'),
           (4106, 'Nada'),
           (11365, 'NASA'),
           (23238, 'Ocasio-Cortez'),
           (1143, 'Odysseus'),
           (679, 'Pax'),
           (17222, 'Perlmutter'),
           (6817, 'Pest'),
           (100417, 'Phillipglass'),
           (12927, 'Pinocchio'),
           (13227, 'Poor'),
           (88705, 'Potato'),
           (7919, 'Prime'),
           (4040, 'Purcell'),
           (12426, 'Racquetball'),
           (17518, 'Redqueen'),
           (120218, 'Richardberry'),
           (16421, 'Roadrunner'),
           (18932, 'Robinhood'),
           (15907, 'Robot'),
           (17058, 'Rocknroll'),
           (1288, 'Santa'),
           (13092, 'Schrodinger'),
           (4856, 'Seaborg'),
           (13070, 'Seanconnery'),
           (6480, 'Scarlatti'),
           (2608, 'Seneca'),
           (2985, 'Shakespeare'),
           (30444, 'Shemp'),
           (129234, 'Silly'),
           (5325, 'Silver'),
           (24068, 'Simonsen'),
           (7934, 'Sinatra'),
           (214476, 'Stephencolbert'),
           (9632, 'Sudo'),
           (19019, 'Sunflower'),
           (14917, 'Taco'),
           (5408, 'The'),
           (1604, 'Tombaugh'),
           (54439, 'Topeka'),
           (249521, 'Truth'),
           (530, 'Turandot'),
           (22791, 'Twarog'),
           (9951, 'Tyrranosaurus'),
           (13069, 'Umbertoeco'),
           (22260, 'Ur'),
           (1282, 'Utopia'),
           (340071, 'Vanmunster'),
           (25399, 'Vonnegut'),
           (17942, 'Whiterabbit'),
           (274301, 'Wikipedia'),
           (100046, 'Worms'),
           (54509, 'Yorp'),
           (7707, 'Yes'),
           (16745, 'Zappa'),
           (3834, 'Zappafrank'),
           (4321, 'Zero')]

PET_KEYWORDS = ['farpoint', 'eskridge', 'hug', 'dose', 'sandlot']

UNCERTAINTY_AFTER_OBS = 0.5  # arcseconds below which uncertainty is presumed not to be reduced
TARGET_OVERHEAD = 60  # seconds to start new target
IMAGE_OVERHEAD = 19  # seconds to start, download, solve new image
EXPOSURES_PER_BLOCK = 9  # assuming 3 stacks of 3 exposures (thus 9 images taken consecutively).
PROCESSING_OVERHEAD = 600  # penalty (seconds) for processing data from one target
EXP_TIME_TABLE_ASTROMETRY = [(16, 50), (17, 75), (18, 150), (19, 300)]  # entry = (v_mag, exp_time sec).
MP_FILTER_NAME = 'Clear'

THIS_LONGITUDE = float(PAYLOAD_DICT_TEMPLATE['long'])
THIS_LATITUDE = float(PAYLOAD_DICT_TEMPLATE['lat'])
THIS_ELEVATION = float(PAYLOAD_DICT_TEMPLATE['alt'])  # MPC's "alt" is really elevation ASL.
THIS_LOCATION = Observer(latitude=THIS_LATITUDE,
                         longitude=THIS_LONGITUDE,
                         elevation=THIS_ELEVATION * u.m)


USER_FUNCTIONS___________________________________ = 0


def go(mp_list=None, mp_start=150000, date_utc=None, max_mps=10000, max_candidates=100,
       keep_useful_only=True, include_old_in_years=FORCE_INCLUDE_IN_YEARS):
    """ Print list and return dataframe of MPs suitable for astrometric followup on given night.
    :param mp_list: list of MP numbers to screen (in place of mp_start and max_mps). [int]
    :param mp_start: starting MP number of consecutive range to screen [int]
    :param date_utc: UTC date string, e.g. '20200110'; if None, uses next night [string]
    :param max_mps: maximum number of MPs to screen. [int]
    :param max_candidates: maximum number of candidates to return. [int]
    :param keep_useful_only: True to keep all MPC "useful for orbit refinement" MPs. [boolean]
    :param include_old_in_years: True to keep all MPs whose last obs was at least this old. [float]
    :return:
    """
    min_score = calc_score(MINIMUM_SCORING['v_mag'], MINIMUM_SCORING['uncert'])
    print('Minimum score =', '{:.1f}'.format(min_score))
    if date_utc is None:
        date_string = next_date_utc()
    else:
        date_string = date_utc
    print('UTC date =', date_string)
    print('Keep Useful Only = ', str(keep_useful_only))
    if include_old_in_years is None:
        print('No inclusion for old obs.')
    else:
        print('Include any MP w/last obs >=', str(include_old_in_years), 'years ago.')
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
        lines = get_one_html_from_list(mp_list_next_html, utc_date_string=date_string)
        html_dict_list = parse_html_lines(lines, keep_useful_only, include_old_in_years)
        all_dict_list.extend(html_dict_list)
        first_index_next_html += n_mps_this_html
        print(' --> ' + str(len(all_dict_list)))
    # This sort order in case LST zero comes in middle of night:
    df = pd.DataFrame(all_dict_list).reindex(columns=DF_COLUMN_ORDER).sort_values(by=['transit', 'ra'])
    pd.set_option('display.width', 144)  # default is 80
    pd.set_option('display.max_colwidth', 100)
    print('Done. ' + str(len(df)) + ' found.')
    if len(df) >= 1:
        print(df.reindex(columns=['score', 'uncert', 'ACP', 'min9', 'last_obs']))
        for idx in df.index:
            print(df.loc[idx, 'number'])
        return df
    else:
        print('No MPs found.')
        return None


def pets(date_utc=None):
    """ Make browser page for MPC ephemeres for all pet MPS.
    :param date_utc: UTC date string, e.g. '20200110'; if None, uses next night [string]
    """
    n = len(PET_MPS)
    n_pages = ceil(n/100)
    for i_page in range(n_pages):
        i_low = 100 * i_page
        i_high = min(i_low + 100, n)
        html([mp for (mp, name) in PET_MPS[i_low: i_high]], date_utc=date_utc)


def go_pets(years=0.5):
    """ Print list and return dataframe of *pet* MPs suitable for astrometric followup on given night.
    [may never happen, as most are very bright.]
    USAGE: df = go_pets()
    :param years: accept pet MPs with most recent obs older than this. [float]
    """
    df = go(mp_list=[mp for (mp, x) in PET_MPS],
            keep_useful_only=False, include_old_in_years=years)


SUPPORT___________________________________________ = 0


def calc_exp_time(v_mag, exp_time_table=None):
    # Check for outside limits:
    if v_mag < exp_time_table[0][0]:
        return exp_time_table[0][1]
    n = len(exp_time_table)
    # Check for equals an entry:
    if v_mag > exp_time_table[n-1][0]:
        return exp_time_table[n-1][1]
    for (v_mag_i, t_i) in exp_time_table:
        if v_mag == v_mag_i:
            return t_i
    # Usual case: linear interpolation in mag (& thus in log(i)):
    for i, entry in enumerate(exp_time_table[:-1]):
        v_mag_i, t_i = entry
        v_mag_next, t_next = exp_time_table[i + 1]
        if v_mag < v_mag_next:
            slope = log(t_next / t_i) / (v_mag_next - v_mag_i)
            log_t = log(t_i) + (v_mag - v_mag_i) * slope
            return exp(log_t)


def calc_seconds_per_block(v_mag):
    return TARGET_OVERHEAD + EXPOSURES_PER_BLOCK \
           * (calc_exp_time(v_mag, EXP_TIME_TABLE_ASTROMETRY) + IMAGE_OVERHEAD)


def calc_score(v_mag, uncert):
    # benefit is roughly: improvement in arcsec uncertainty.
    # cost is roughly hours of scope+user time.
    benefit = uncert - UNCERTAINTY_AFTER_OBS
    cost_in_hours = (calc_seconds_per_block(v_mag) + PROCESSING_OVERHEAD) / 3600.0
    return benefit / cost_in_hours


def combine(df_list):
    # Usage: df_20181214 = combine([df1, df2, df3])
    return pd.concat(df_list, ignore_index=True).sort_values(by=['transit', 'ra'])


def append(df1, df2):
    # Usage: df_20181214 = append(df_20181214, df1)
    return df1.append(df2, ignore_index=True).sort_values(by=['transit', 'ra'])


def get_transit_time(ra, dec, date_string):
    """  Return UTC transit time for obj at RA,Dec for ~midnight on date_string date.
    :param ra: (degrees) [float]
    :param dec: (degrees) [float]
    :param date_string: e.g., '20181211' [string]
    :return:
    """
    year = int(date_string[0:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])
    dt = datetime(year, month, day, 6)
    coord = SkyCoord(ra, dec, unit='deg')
    return THIS_LOCATION.target_meridian_transit_time(dt, coord).to_datetime()


def parse_html_lines(lines, keep_useful_only=False, include_old_in_years=None):
    min_score = calc_score(MINIMUM_SCORING['v_mag'], MINIMUM_SCORING['uncert'])
    mp_dict_list = []
    mp_block_limits = chop_html(lines)
    for limits in mp_block_limits:
        mp_dict = extract_mp_data(lines, limits,
                                  keep_always=False,
                                  keep_useful_only=keep_useful_only,
                                  include_old_in_years=include_old_in_years)
        if any([v is None for v in mp_dict.values()]):
            print(mp_dict)
        if mp_dict.get('v_mag', None) is not None and mp_dict.get('uncert', None) is not None:
            score = calc_score(float(mp_dict['v_mag']), float(mp_dict['uncert']))
            useful = 'useful' in mp_dict.get('status', '').lower()
            if include_old_in_years is None:
                old_enough = False
            else:
                years_old = get_years_old(mp_dict['last_obs'])
                old_enough = (years_old >= include_old_in_years > 0)
            if old_enough:
                worth_including = True  # keep if old enough, regardless of usefulness or score.
            elif keep_useful_only and not useful:
                worth_including = False  # skip if required to be MPC-useful but is not.
            else:
                worth_including = score >= min_score
            if worth_including:
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


def get_one_html_from_list(mp_list=None, utc_date_string='20181122', payload_dict=None):
    """ Gets MPC HTML text for a list of MPs, on a given UTC date.
    :param mp_list: list of MP numbers [list of ints].
    :param utc_date_string: UTC date to query (not necessarily AN date!) [string].
    :return: MPC HTML text from MPEC [list of strings].
    """
    if payload_dict is None:
        payload_dict = PAYLOAD_DICT_TEMPLATE.copy()

    # Construct TextArea field:
    text_area = '%0D%0A'.join([str(mp) for mp in mp_list])
    payload_dict['TextArea'] = text_area

    payload_dict['d'] = utc_date_string

    # Make longitude and latitude safe (from '+' characters)
    payload_dict['long'] = payload_dict['long'].replace("+", "%2B")
    payload_dict['lat'] = payload_dict['lat'].replace("+", "%2B")

    # ##################  GET VERSION.  ######################
    # # Construct URL and header for GET call:
    payload_string = '&'.join([k + '=' + v for (k, v) in payload_dict.items()])
    url = MPC_URL_STUB + '/?' + payload_string
    # Make GET call, parse return text.
    r = requests.get(url, headers=GET_HEADER)
    # ################# End GET VERSION. #####################
    # ##################  POST VERSION.  ######################
    # url = MPC_URL_STUB
    # # Make POST call, parse return text.
    # r = requests.get(url, data=payload_dict)
    # ################# End GET VERSION. #####################
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


def extract_mp_data(html_lines, mp_block_limits, keep_always=False, keep_useful_only=False,
                    include_old_in_years=False):
    """ Get data out of a HTML text block devoted to one MP.
    :param html_lines:
    :param mp_block_limits:
    :param keep_always: True iff always return data. True value overrides keep_useful_only and
               include_old_in_years; False allows them to operate [boolean].
    :param keep_useful_only: (no-op if keep_always=True) True iff user wants to keep MPs that MPC deems
               useful to observe [boolean].
    :param include_old_in_years: (no-op if keep_always=True) True iff user wants to include any and all MPs
               whose last observations are older than FORCE_INCLUDE_IN_YEARS number of years [boolean].
    :return: dict of MP data, keys=number, name, code, last_obs, status, v_mag, motion, mp_alt.
    """
    mp_dict = dict()
    mp_dict['comments'] = ''
    mp_dict['status'] = ''  # default in case 'Futher observations?' line is missing (rare).

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
            if line.strip().split('>')[3].strip().lower().startswith('not necessary'):
                mp_dict['status'] = 'no'
            if line.strip().split('>')[3].strip().lower().startswith('useful'):
                mp_dict['status'] = 'USEFUL'
        if 'discovery site' in line.lower():
            this_site = line.strip().split(':', maxsplit=1)[1].strip()
            if this_site.lower() in PET_KEYWORDS:
                mp_dict['comments'] = mp_dict['comments'] + this_site + '; '
        if 'discoverer(s)' in line.lower():
            discoverer = line.strip().split(':', maxsplit=1)[1].strip()
            if any([p in discoverer.lower() for p in PET_KEYWORDS]):
                mp_dict['comments'] = mp_dict['comments'] + discoverer + '; '
        if mp_dict['comments'] == '':
            mp_dict['comments'] = '-'

    # Return now if no need to parse table & look up uncertainty data (major speed advantage).
    mp_dict_short = mp_dict.copy()
    is_useful = (mp_dict['status'].lower() == 'useful')
    if include_old_in_years is None:
        old_enough_to_force = True
    else:
        old_enough_to_force = (get_years_old(mp_dict['last_obs']) >= include_old_in_years)
    if (keep_useful_only and not is_useful) and (not old_enough_to_force):  # i.e., if no need to keep
        return mp_dict_short

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
            high_enough = mp_alt >= MIN_MP_ALTITUDE
            bright_enough = v_mag <= MAX_V_MAG
            sun_low_enough = sun_alt <= MAX_SUN_ALTITUDE
            moon_distant_enough = (moon_dist >= MIN_MOON_DIST or moon_alt < 0)
            if high_enough and bright_enough and sun_low_enough and moon_distant_enough:
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
                    mp_dict['moon_dist'] = line_split[21]
                    mp_dict['moon_alt'] = '{:d}'.format(round(moon_alt))
                    mp_dict['min9'] = '{:d}'.format(round(calc_seconds_per_block(v_mag) / 60.0))
                    exp_time = calc_exp_time(v_mag, EXP_TIME_TABLE_ASTROMETRY)
                    uncertainty_raw_url = [word for word in line_split
                                           if 'cgi.minorplanetcenter.net/cgi' in word][1]
                    ra_degrees = ra_as_degrees(mp_dict['ra'])
                    dec_degrees = dec_as_degrees(mp_dict['dec'])
                    date_string = ''.join(line_split[0:3])
                    transit_time = get_transit_time(ra_degrees, dec_degrees, date_string)
                    mp_dict['transit'] = '{0:02d}{1:02d}'.format(transit_time.hour, transit_time.minute)
                    mp_dict['ACP'] = 'IMAGE MP_' + mp_dict['number'].zfill(6) + \
                                     '_t' + mp_dict['transit'] + \
                                     '  ' + MP_FILTER_NAME + '={:.0f}'.format(exp_time) + 'sec(3)' + \
                                     '  ' + mp_dict['ra'] + \
                                     '  ' + mp_dict['dec']

    if mp_dict.get('v_mag', None) is not None:
        uncertainty = get_uncertainty(uncertainty_raw_url)
        # print('\n', mp_dict['number'], 'get_uncertainty()', mp_dict['utc'])
        mp_dict['uncert'] = '{:.1f}'.format(uncertainty)
        mp_dict['score'] = '{:.1f}'.format(calc_score(float(mp_dict['v_mag']), uncertainty))
    else:
        mp_dict = mp_dict_short
    return mp_dict


# def parse_one_mp_from_html(mp_dict=None, html_lines=[], mp_block_limits=None):
#     if mp_dict is None:
#         mp_dict = dict()
#     if len(html_lines) <= 0:
#         return None
#     if mp_block_limits is None:
#         mp_block_limits = chop_html(html_lines)[0]
#     table_start_found = False
#     max_mp_alt = None
#     uncertainty_raw_url = ''  # to keep IDE happy
#     for i_line in range(mp_block_limits[0], mp_block_limits[1]):
#         line = html_lines[i_line]
#         if '<pre>' in line:
#             table_start_found = True
#         if table_start_found:
#             if '</pre>' in line:
#                 break
#         line_split = line.split()
#         if len(line_split) >= MIN_TABLE_WORDS:
#             mp_alt = float(line.split()[18])
#             v_mag = float(line_split[14])
#             sun_alt = float(line_split[19])
#             moon_dist = float(line_split[21])
#             moon_alt = float(line_split[22])
#             high_enough = mp_alt >= MIN_MP_ALTITUDE
#             bright_enough = v_mag <= MAX_V_MAG
#             sun_low_enough = sun_alt <= MAX_SUN_ALTITUDE
#             moon_distant_enough = (moon_dist >= MIN_MOON_DIST or moon_alt < 0)
#             if high_enough and bright_enough and sun_low_enough and moon_distant_enough:
#                 if max_mp_alt is None:
#                     this_mp_alt_is_max_so_far = True
#                 else:
#                     this_mp_alt_is_max_so_far = (mp_alt > max_mp_alt)
#                 if this_mp_alt_is_max_so_far:
#                     max_mp_alt = mp_alt
#                     mp_dict['utc'] = ' '.join(line_split[0:4])
#                     mp_dict['ra'] = ':'.join(line_split[4:7])
#                     mp_dict['dec'] = ':'.join(line_split[7:10])
#                     mp_dict['mp_alt'] = '{:d}'.format(round(mp_alt))
#                     mp_dict['v_mag'] = '{:.1f}'.format(v_mag)
#                     mp_dict['motion'] = line_split[15]
#                     mp_dict['motion_pa'] = line_split[16]
#                     mp_dict['moon_phase'] = line_split[20]
#                     mp_dict['moon_alt'] = '{:d}'.format(round(moon_alt))
#                     mp_dict['min9'] = '{:d}'.format(round(calc_seconds_per_block(v_mag) / 60.0))
#                     exp_time = calc_exp_time(v_mag)
#                     uncertainty_raw_url = [word for word in line_split
#                                            if 'cgi.minorplanetcenter.net/cgi' in word][1]
#                     ra_degrees = ra_as_degrees(mp_dict['ra'])
#                     dec_degrees = dec_as_degrees(mp_dict['dec'])
#                     date_string = ''.join(line_split[0:3])
#                     transit_time = get_transit_time(ra_degrees, dec_degrees, date_string)
#                     mp_dict['transit'] = '{0:02d}{1:02d}'.format(transit_time.hour, transit_time.minute)
#                     mp_dict['ACP'] = 'IMAGE MP_' + mp_dict['number'].zfill(6) + \
#                                      '_t' + mp_dict['transit'] + \
#                                      '  ' + MP_FILTER_NAME + '={:.0f}'.format(exp_time) + 'sec(9)' + \
#                                      '  ' + mp_dict['ra'] + \
#                                      '  ' + mp_dict['dec']
#     return mp_dict, uncertainty_raw_url


def get_uncertainty(raw_url):
    url = raw_url.split('href=\"')[-1].split('&OC')[0]
    # print(url)  # for test
    # return 1    # for test
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


def html(mp_list=None, mp_start=None, df=None, date_utc=None):
    if mp_list is None and df is None and mp_start is None:  # sentinel.
        print('Please provide df= or mp_list= or mp_start.')
        return
    if mp_list is not None:
        pass  # use this.
    elif mp_start is not None:
        mp_list = list(range(mp_start, mp_start+100))
    else:
        mp_list = df['number']
    if len(mp_list) > 100:
        print('There are', str(len(mp_list)),
              'MPs requested. Displaying only the first 100 (MPC\'s max allowed).')
        mp_list = mp_list[0:100]
    if date_utc is None:
        date_utc = next_date_utc()

    payload_dict = PAYLOAD_DICT_TEMPLATE.copy()
    # Construct TextArea field:
    text_area = '%0D%0A'.join([str(mp) for mp in mp_list])
    payload_dict['TextArea'] = text_area
    payload_dict['d'] = date_utc
    # date = ''.join(df['utc'][0].split()[0:3])
    # payload_dict['d'] = date
    # Make longitude and latitude safe (from '+' characters)
    payload_dict['long'] = payload_dict['long'].replace("+", "%2B")
    payload_dict['lat'] = payload_dict['lat'].replace("+", "%2B")

    # ##################  GET VERSION.  ######################
    # # Construct URL and header for GET call:
    payload_string = '&'.join([str(k) + '=' + str(v) for (k, v) in payload_dict.items()])
    url = MPC_URL_STUB + '/?' + payload_string
    webbrowser.open_new(url)


def write_csv(df=None, an_date=None):
    # Usage: write_csv(df=df, an_date='20190119')  NB: an_date is prob. 1 day before UTC_date.
    fullpath = os.path.join(PLANNING_ROOT_DIRECTORY, 'AN' + an_date, 'mps_' + an_date + '.csv')
    df.to_csv(fullpath)


UTILITY_FUNCTIONS___________________________________ = 0


def get_years_old(last_obs_string):
    words = last_obs_string.split()
    year = int(words[0])
    month = MPC_HTML_MONTH_CODES[words[1][0:3].lower()]
    day = int(words[2])
    dt_obs = datetime(year=year, month=month, day=day, hour=12).replace(tzinfo=timezone.utc)
    dt_now = datetime.now(timezone.utc)
    return (dt_now - dt_obs).total_seconds() / 365.25 / 24 / 3600


def next_date_utc():
    target_date = datetime.now(timezone.utc) + timedelta(days=1)
    date_utc = '{0:04d}{1:02d}{2:02d}'.format(target_date.year, target_date.month, target_date.day)
    return date_utc


def ra_as_degrees(ra_string):
    """
    :param ra_string: string in either full hex ("12:34:56.7777" or "12 34 56.7777"),
               or degrees ("234.55")
    :return float of Right Ascension in degrees between 0 and 360.
    """
    ra_list = parse_hex(ra_string)
    if len(ra_list) == 1:
        ra_degrees = float(ra_list[0])  # input assumed to be in degrees.
    elif len(ra_list) == 2:
        ra_degrees = 15 * (float(ra_list[0]) + float(ra_list[1])/60.0)  # input assumed in hex.
    else:
        ra_degrees = 15 * (float(ra_list[0]) + float(ra_list[1]) / 60.0 +
                           float(ra_list[2])/3600.0)  # input assumed in hex.
    if (ra_degrees < 0) | (ra_degrees > 360):
        ra_degrees = None
    return ra_degrees


def dec_as_degrees(dec_string):
    """ Input: string in either full hex ("-12:34:56.7777") or degrees ("-24.55")
        Returns: float of Declination in degrees, required to be -90 to +90, inclusive.
    """
    dec_degrees = hex_degrees_as_degrees(dec_string)
    if (dec_degrees < -90) | (dec_degrees > +90):
        dec_degrees = None
    return dec_degrees


def hex_degrees_as_degrees(hex_degrees_string):
    """
    :param hex_degrees_string: string in either full hex ("-12:34:56.7777", or "-12 34 56.7777"),
        or degrees ("-24.55")
    :return float of degrees (not limited)
    """
    # dec_list = hex_degrees_string.split(":")
    dec_list = parse_hex(hex_degrees_string)
    # dec_list = [dec.strip() for dec in dec_list]
    if dec_list[0].startswith("-"):
        sign = -1
    else:
        sign = 1
    if len(dec_list) == 1:
        dec_degrees = float(dec_list[0])  # input assumed to be in degrees.
    elif len(dec_list) == 2:
        dec_degrees = sign * (abs(float(dec_list[0])) + float(dec_list[1])/60.0)  # input is hex.
    else:
        dec_degrees = sign * (abs(float(dec_list[0])) + float(dec_list[1]) / 60.0 +
                              float(dec_list[2])/3600.0)  # input is hex.
    return dec_degrees


def parse_hex(hex_string):
    """
    Helper function for RA and Dec parsing, takes hex string, returns list of floats.
    :param hex_string: string in either full hex ("12:34:56.7777" or "12 34 56.7777"),
               or degrees ("234.55")
    :return: list of strings representing floats (hours:min:sec or deg:arcmin:arcsec).
    """
    colon_list = hex_string.split(':')
    space_list = hex_string.split()  # multiple spaces act as one delimiter
    if len(colon_list) >= len(space_list):
        return [x.strip() for x in colon_list]
    return space_list




