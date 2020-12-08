def create_file(filename):
    file = open(filename, 'w+')

def append_to_file(filename, content):
    f = open(filename, "a")
    f.write(content)
    f.close()