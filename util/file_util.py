import os

def create_file(filename):
    file = open(filename, 'w+')

def append_to_file(filename, content):
    f = open(filename, "a")
    f.write('\n\n' + content)
    f.close()

def create_folders_and_subfolders(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)