import os
import json
import zipfile
import shutil

from pathlib import Path
from time import gmtime, strftime

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write(buffer, path, write_mode, override=False):
    p = Path(path)
    if override and p.exists():
        shutil.rmtree(p)

    with p.open(write_mode) as fout:
        fout.write(buffer)


def read(path, read_mode):
    p  = Path(path)
    with p.open(read_mode) as fin:
        buffer = fin.read()
    return buffer


def readlines(images_txt_file, empty_list_if_no_file=False):
    # A list of files, stored in in_images_file where each line contains a name of
    # a file in directory in_dir
    p = Path(images_txt_file)
    if not p.exists():
        if empty_list_if_no_file:
            return []
        else:
            err_str = f'File does not exist: {images_txt_file}'
            raise IOError(err_str)
    
    with open(p) as f:
        lines = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    lines = [x.strip() for x in lines]

    return lines


def list_directory(in_dir, empty_list_if_no_dir=False, return_full_path=False):
    """
    Get list of files in directory
    :param images_dir:
    :return: list of entries within this directory
    :raises IOError if directory not found
    """
    p = Path(in_dir)
    if p.is_dir():
        ldirs = sorted([f for f in p.iterdir() if f.is_file()])
        if return_full_path:
            return [os.path.join(in_dir, ld) for ld in ldirs]
        else:
            return [f.name for f in ldirs]
    elif empty_list_if_no_dir:
        return []
    else:
        err_str = f'Not a directory: {p}'
        raise IOError(err_str)

def list_files_recursive(root_dir, file_types):
    """
    Get names of all image files recusrivly in all subdirectories from a directory.
    :param dir_to_list: root directory
    :param file_types: A string of list of strins with extentions of files to list
    :return: A dictionary indexed by a path relative to the root directory. The value (sorted) list of full path of all files in dir_to_list with the given extentions
    """
    p = Path(root_dir)

    ret_dict = dict()
    for dir_name, _, _ in os.walk(p):
        key = os.path.relpath(dir_name, p)
        rq_files = list_files(dir_name, file_types)
        ret_dict[key] = rq_files
    return ret_dict

def list_files(dir_to_list, file_types):
    """
    Get names of all files of the required file types in the specified directory
    :param dir_to_list: directory
    :param file_types: A string or list of strins with extentions of files to list, e.g. ['jpg','zip']
    :return: A (sorted) list of full path of all files in dir_to_list with the given extentions
    """
    p = Path(dir_to_list)
    if isinstance(file_types, str):
        file_types = [file_types]

    if not isinstance(file_types, tuple):
        file_types = list(file_types)
    
    if not isinstance(file_types, list):
        file_types = [file_types]
    
    dir_to_list = os.path.join(dir_to_list, '')
    ftypes = ['*.{ftype}'.format(ftype=ftype) for ftype in file_types]

    fnames = []
    for ftype in ftypes:
        fnames.extend(p.glob(ftype))
    return sorted(fnames)

def list_image_files(image_dir):
    """
    Get names of all image files in a directory.
    :param image_dir: Directory with images
    :return: A (sorted) list of names of all image files in imega_dir
    """
    return list_files(dir_to_list=image_dir, file_types=['jpg', 'png', 'JPG', 'bmp'])

def list_npy_files(root_dir):
    """
    Get names of all npy files (numpy) in directory.
    :param root_dir: Directory with npy files
    :return: A (sorted) list of names of all image files in imega_dir
    """
    return list_files(dir_to_list=root_dir, file_types='npy')

def list_pbtxt_files(root_dir):
    """
    Get names of all text protobuf files (pbtxt) in directory.
    :param root_dir: Directory with pbtxt files
    :return: A (sorted) list of names of all image files in imega_dir
    """
    return list_files(dir_to_list=root_dir, file_types='pbtxt')

def list_video_files(video_dir):
    """
    Get names of all video files in a directory.
    :param video_dir: Directory with video files
    :return: A (sorted) list of names of all video files in video_dir
    """
    return list_files(dir_to_list=video_dir, file_types=['mp4', 'mov'])

def list_zip_files(in_dir):
    """
    Get names of all compressed files in a directory.
    :param in_dir: Directory with video files
    :return: A (sorted) list of names of all zip files in in_dir
    """
    return list_files(dir_to_list=in_dir, file_types=['zip'])

def list_json_files(root_dir):
    """
    Get names of all json files in a directory.
    :param root_dir: Directory with json file
    :return: A (sorted) list of names of all json files in directory
    """
    return list_files(dir_to_list=root_dir, file_types='json')

def list_csv_files(root_dir):
    """
    Get names of all csv files in a directory.
    :param root_dir: Directory with csv files
    :return: A (sorted) list of names of all csv files in directory
    """
    return list_files(dir_to_list=root_dir, file_types='csv')

def read_json_file(path):
    p = Path(path)
    json_str = read(str(p), 'r')
    json_dict = json.loads(json_str)
    return json_dict


def json_string(json_dict):
    json_str = json.dumps(json_dict, indent=2, sort_keys=True, cls=NumpyEncoder)
    return json_str

def write_json_file(json_dict, path, overwrite=False):
    # json_str = json.dumps(json_dict, indent=2, sort_keys=True, cls=NumpyEncoder)
    write(json_string(json_dict), path, 'w', override=overwrite)
