from pathlib import Path
import urllib.request
import zipfile
import shutil
import os

aligned_test_url = "https://www.ugrad.cs.jhu.edu/~dlu17/aligned_test.zip"
aligned_train_url = "https://www.ugrad.cs.jhu.edu/~dlu17/aligned_train.zip"
list_partition_label_url = "https://www.ugrad.cs.jhu.edu/~dlu17/list_patition_label.txt"
manual_url = "https://www.ugrad.cs.jhu.edu/~wwang136/manual.zip"

rafdb = Path('./data/RAF')
rafdb.mkdir(exist_ok=True, parents=True)

# download and unzip the data using python

def download_and_unzip(url, dest):
    # if already exists, return
    if dest.exists():
        return
    print(f"Downloading {url} to {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Unzipping {dest}")
    with zipfile.ZipFile(dest, 'r') as zip_ref:
        zip_ref.extractall(rafdb)

def remove_parent_dir(target):
    # rename extracted folder
    extracted_folder = target
    extracted_folder.rename("tmp")
    # move out of parent folder
    for f in Path("tmp").iterdir():
        shutil.move(f, target.parent)
    # remove tmp folder
    shutil.rmtree("tmp")


download_and_unzip(aligned_test_url, rafdb / 'aligned_test.zip')
download_and_unzip(aligned_train_url, rafdb / 'aligned_train.zip')
download_and_unzip(manual_url, rafdb / 'manual.zip')

remove_parent_dir(rafdb / 'aligned_test')
remove_parent_dir(rafdb / 'aligned_train')

# download the list_patition_label.txt file
urllib.request.urlretrieve(list_partition_label_url, rafdb / 'list_partition_label.txt')
