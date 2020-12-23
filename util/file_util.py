import os
import shutil

def create_file(filename):
    file = open(filename, 'w+')

def append_to_file(filename, content):
    f = open(filename, "a")
    f.write(content)
    f.close()

def delete_and_create_folders_and_subfolders(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)
    os.makedirs(folder_path)