import os


def append_to_file(filename, content):
    f = open(filename, "a")
    f.write('\n\n' + content)
    f.close()


def create_and_write_file(filename, content):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w+')
    f.write(content)
    f.close()


def create_folders_and_subfolders(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
