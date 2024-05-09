import csv
import fnmatch
import gzip
import json
import lzma
import os
import pickle
import shutil
from typing import Union, List, Dict


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def delete_file_or_dir(dir):
    if os.path.isfile(dir):
        os.remove(dir)
    elif os.path.exists(dir):
        shutil.rmtree(dir)
    else:
        pass


def find_files(dir, name_pattern):
    """
    Search for files matching a specified pattern in a given directory and its subdirectories.

    Args:
    - dir: String of root directory path to search.
    - name_pattern: String of pattern to match filename against (e.g. '*.txt' to match all txt files).

    Returns:
    - A list of full paths to the found files.
    """
    matches = []
    for root, dirs, files in os.walk(dir):
        for filename in fnmatch.filter(files, name_pattern):
            matches.append(os.path.join(root, filename))
    return matches


def save_compressed_file_7z(data, file_path):  # 7z
    create_dir(os.path.dirname(file_path))
    with lzma.open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_compressed_file_7z(file_path):  # 7z
    with lzma.open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def save_compressed_file_gz(data, file_path, compresslevel=6):  # gz
    create_dir(os.path.dirname(file_path))
    with gzip.open(file_path, "wb", compresslevel=compresslevel) as file:
        pickle.dump(data, file)


def load_compressed_file_gz(file_path):  # gz
    with gzip.open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def read_csv(file_path, has_header=True) -> Union[List[List], List[Dict]]:
    """
    Read a CSV file and return its content.

    Args:
    - file_path (str): Path to the CSV file.
    - has_header (bool): Whether the CSV file has a header. Default is True.

    Returns:
    - list of list or dict: Content of the CSV file.
      If has_header is True, return a list of dictionaries;
      if has_header is False, return a list of lists.
    """
    data = []
    with open(file_path, newline='', encoding='utf-8') as f:
        if has_header:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                data.append(dict(row))
        else:
            csvreader = csv.reader(f)
            for row in csvreader:
                data.append(row)
    return data


def load_json(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def save_json(data, file_path, indent=4, **kwargs):
    create_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf8") as f:
        f.write(f"{json.dumps(data, ensure_ascii=False, indent=indent, **kwargs)}\n")


def load_jsonl(file_path) -> list:
    data = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {line}")
                continue
    return data


def save_jsonl(data, file_path, **kwargs):
    create_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf8") as f:
        for ins in data:
            f.write(f"{json.dumps(ins, ensure_ascii=False, **kwargs)}\n")


def compress_png_image(image_path, print_info=False):
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imwrite(image_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if print_info:
        print(f'Done for "{image_path}".')
