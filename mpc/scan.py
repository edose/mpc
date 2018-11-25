__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

from collections import OrderedDict
import os
from random import randint
from time import sleep
from webbrowser import open_new_tab

import requests
import pandas as pd

MPC_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAYLOAD_DICT_TEMPLATE = OrderedDict([
    ('ty', 'e'),  # e = 'Return Ephemerides'
    ('TextArea', ''),  # the MP IDs
    ('d', '20181122'),  # first date
    ('l', '14'),  # number of dates/times
    # TODO: should be one line for every 30 minutes, not for every 1 hour.
    ('i', '1'),  # interval between ephemerides
    ('u', 'h'),  # units of interval; 'h' for hours, 'd' for days
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
MIN_TABLE_WORDS = 25
MIN_ALTITUDE = 30
MAX_SUN_ALTITUDE = -12
MAX_V_MAG = 20


def get_100_candidates(mp_start=100000, date='20181125'):
    all_dict_list = []
    html_start = mp_start
    for i_html in range(100):  # limit number of calls to MPC to get this done:
        if i_html > 0:
            sleep(randint(3, 7))  # delay for 3 to 7 seconds (playing nicely with MPC server).
        this_dict_list = scan_one_html(html_start, date)
        if len(all_dict_list) >= MAX_MP_PER_HTML:
            break
        all_dict_list.extend(this_dict_list)
        print(str(i_html) + ' ' +
              str(html_start) + '  + ' +
              str(len(this_dict_list)) + '   total: ' +
              str(len(all_dict_list)))
        html_start += MAX_MP_PER_HTML
    all_dict_list = all_dict_list[:MAX_MP_PER_HTML]  # truncate if we got a few more than needed.

    # Construct URL and display condensed MPC results in browser:
    payload_dict = PAYLOAD_DICT_TEMPLATE.copy()
    mp_list = [item['number'] for item in all_dict_list]
    payload_dict['TextArea'] = '%0D%0A'.join(mp_list)
    payload_dict['d'] = date
    payload_dict['long'] = payload_dict['long'].replace("+", "%2B")  # make safe
    payload_dict['lat'] = payload_dict['lat'].replace("+", "%2B")    # make safe
    payload_string = '&'.join([k + '=' + v for (k, v) in payload_dict.items()])
    url = MPC_URL_STUB + payload_string
    open_new_tab(url)


def scan_one_html(mp_start=100000, date='20181125'):
    mp_dict_list = []
    lines = get_one_html(start=mp_start, date=date)
    mp_block_limits = chop_html(lines)
    for limits in mp_block_limits:
        mp_dict = extract_mp_data(lines, limits)
        if mp_dict.get('v_mag', None) is not None:
            mp_dict_list.append(mp_dict)
            print(mp_dict['number'])
    return mp_dict_list


def get_one_html(start=200000, n=MAX_MP_PER_HTML, date='20181122'):
    """ Gets MPC HTML text, returns list of strings """
    payload_dict = PAYLOAD_DICT_TEMPLATE.copy()

    # Construct TextArea field:
    mp_list = [str(i) for i in range(start, start + n)]
    text_area = '%0D%0A'.join(mp_list)
    payload_dict['TextArea'] = text_area

    payload_dict['d'] = date

    # Make longitude and latitude safe (from '+' characters)
    payload_dict['long'] = payload_dict['long'].replace("+", "%2B")
    payload_dict['lat'] = payload_dict['lat'].replace("+", "%2B")

    # Construct URL and header for GET call:
    payload_string = '&'.join([k + '=' + v for (k, v) in payload_dict.items()])
    url = MPC_URL_STUB + payload_string

    # Make GET call, parse return text.
    print(url)
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

    # Handle one-per-MP items:
    for i_line in range(mp_block_limits[0], mp_block_limits[1]):
        line = html_lines[i_line]
        if line.strip().startswith('<b>('):
            mp_dict['number'] = line.split(')')[0].split('(')[-1].strip()
            mp_dict['name'] = line.split(')')[-1].split('<')[0].strip()
        if line.strip().startswith('Last observed on'):
            mp_dict['last_obs'] = line.strip()[len('Last observed on'):].strip().replace('.', '')
        if line.strip().startswith('<p><pre>'):
            if i_line + 1 < mp_block_limits[1]:
                mp_dict['code'] = html_lines[i_line + 1].strip().split()[0]
        if line.find('Further observations?') != -1:
            mp_dict['status'] = ''
            if line.strip().split('>')[3].strip().lower().startswith('not necessary'):
                mp_dict['status'] = 'no'
            if line.strip().split('>')[3].strip().lower().startswith('useful'):
                mp_dict['status'] = 'USEFUL'

    # Handle items in ephemeris table. Use line with highest altitude:
    # Find line limits of ephemeris table (slight superset from <pre> and </pre> HTML tags:
    table_start_found = False
    max_mp_alt = None
    for i_line in range(mp_block_limits[0], mp_block_limits[1]):
        line = html_lines[i_line]
        if '<pre>' in line:
            table_start_found = True
        if table_start_found:
            if '</pre>' in line:
                break
        if len(line.split()) >= MIN_TABLE_WORDS:
            mp_alt = float(line.split()[18])
            if max_mp_alt is None:
                this_mp_alt_is_max_so_far = True
            else:
                this_mp_alt_is_max_so_far = (mp_alt > max_mp_alt)
            if this_mp_alt_is_max_so_far:
                max_mp_alt = mp_alt
                v_mag = float(line.split()[14])
                sun_alt = float(line.split()[19])
                # TODO: Also get uncertainty in arcsec (from corrected URL in line) & 'if' on that as well.
                if mp_alt >= MIN_ALTITUDE and v_mag <= MAX_V_MAG and sun_alt <= MAX_SUN_ALTITUDE:
                    motion = float(line.split()[15])
                    mp_dict['v_mag'] = v_mag
                    mp_dict['motion'] = motion
                    mp_dict['mp_alt'] = mp_alt
    return mp_dict


def t():
    lines = get_html_from_file()
    mp_block_limits = chop_html(lines)
    for limits in mp_block_limits:
        mp_dict = extract_mp_data(lines, limits)
        #if mp_dict.get('v_mag') is not None:
        print(mp_dict)


def web(start=200000, n=50, date='20181201'):
    lines = get_one_html(start, n, date)
    mp_block_limits = chop_html(lines)
    data = []
    for limits in mp_block_limits:
        mp_dict = extract_mp_data(lines, limits)
        data.append(mp_dict)
        print(mp_dict['number'])
    for mp_dict in data:
        print(mp_dict)




if __name__ == "__main__":
    get_one_html()
