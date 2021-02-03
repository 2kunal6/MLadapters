from owlready2 import *
import os

from util import file_util
from service.file_content_creator import create_file_contents

child_parent_map = {}
onto = get_ontology("ml_algorithms.owl").load()
queue = []
root_class = None
print([i for i in onto.annotation_properties()])
for onto_class in list(onto.classes()):
    if(onto_class.label[0] == 'MLalgorithms'):
        root_class = onto_class

queue.append(root_class)
dir_structure = [root_class.label[0]]
file_path = []
child_parent_map[root_class] = None
while queue:
    node = queue.pop(0)
    file = dir_structure.pop(0)
    if onto.get_children_of(node):
        file_util.create_folders_and_subfolders(file)
    file_util.create_file(file + ".py")
    create_file_contents(file, node, child_parent_map)
    file_path.append(file)
    for child in onto.get_children_of(node):
        queue.append(child)
        child_parent_map[child] = node
        dir_structure.append(os.path.join(file, child.label[0]))

print(file_path)