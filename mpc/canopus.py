__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import os

MP_TOP_DIRECTORY = 'C:/Astro/MP Photometry/'


def canopus(mp_top_directory=MP_TOP_DIRECTORY, rel_directory=None):
    """ Read all FITS in mp_directory, rotate right, bin 2x2, invalidating plate solution.
    Intended for making images suitable (North Up, smaller) for photometric reduction in Canopus 10.
    Tests OK ~20191101.
    :param mp_top_directory: top path for FITS files [string]
    :param rel_directory: rest of path to FITS files, e.g., 'MP_768/AN20191020' [string]
    : return: None
    """
    this_directory = os.path.join(mp_top_directory, rel_directory)
    # clean_subdirectory(this_directory, 'Canopus')
    # output_directory = os.path.join(this_directory, 'Canopus')
    output_directory = this_directory
    import win32com.client
    app = win32com.client.Dispatch('MaxIm.Application')
    count = 0
    for entry in os.scandir(this_directory):
        if entry.is_file():
            fullpath = os.path.join(this_directory, entry.name)
            doc = win32com.client.Dispatch('MaxIm.Document')
            doc.Openfile(fullpath)
            doc.RotateRight()  # Canopus requires North-up.
            doc.Bin(2)  # to fit into Canopus image viewer.
            doc.StretchMode = 2  # = High, the better to see MP.
            output_filename, output_ext = os.path.splitext(entry.name)
            output_fullpath = os.path.join(output_directory, output_filename + '_Canopus' + output_ext)
            doc.SaveFile(output_fullpath, 3, False, 3, False)  # FITS, no stretch, floats, no compression.
            doc.Close  # no parentheses is actually correct. (weird MaxIm API)
            count += 1
            print('*', end='', flush=True)
    print('\n' + str(count), 'converted FITS now in', output_directory)
