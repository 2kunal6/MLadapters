from owlready2 import *
from util import file_util
from util.graph_util import Graph
from service import python_content_creator

import shutil

#onto = get_ontology("https://github.com/2kunal6/SemanticWebLab/blob/master/ml-hierarchy.owl")
#onto = get_ontology("https://raw.githubusercontent.com/2kunal6/SemanticWebLab/master/ml-hierarchy.owl")
onto = get_ontology("ml-hierarchy.owl")

onto.load()

print(list(onto.classes()))

g = Graph()

for onto_class in list(onto.classes()):
    ontology_class = str(onto_class)
    if (ontology_class == "ml-hierarchy.MachineLearningAlgorithms"):
        for algorithms_ontology_class in list(onto_class.descendants()):
            algorithms_onto_class = str(algorithms_ontology_class)
            algorithms_onto_class = algorithms_onto_class.replace('ml-hierarchy.', '')

            parent_classes = onto.get_parents_of(algorithms_ontology_class)
            if(not parent_classes):
                continue

            g.addEdge(str(parent_classes[0]).replace('ml-hierarchy.', ''), algorithms_onto_class)
        break
print('starting BFS')
file_structures = g.BFS('MachineLearningAlgorithms')
print(file_structures)

shutil.rmtree('MachineLearningAlgorithms', ignore_errors=True)

for file_structure in file_structures:
    file_structure_splits = file_structure.split('/')
    for i in range(len(file_structure_splits)):
        file_structure_splits[i] = file_structure_splits[i].split('----')[-1]

    file_util.create_folders_and_subfolders("/".join(file_structure_splits))

    filename = "/".join(file_structure_splits) + '/' + (file_structure_splits[-1]) + '.py'
    file_util.create_file(filename)

    parentclass = None
    if(len(file_structure_splits) > 1):
        parentclass = file_structure_splits[-2]

    file_util.append_to_file(filename, python_content_creator.create_class(file_structure_splits[-1], parentclass))
    for onto_class in list(onto.classes()):
        ontology_class = str(onto_class)
        if(ontology_class.lower().endswith(file_structure.split('/')[-1].lower())):
            file_util.append_to_file(filename, "'''" + str(onto_class.comment) + "'''")
            isSupervised = str(onto_class.isSupervised)
            break

    init_function_parameter = ''
    for onto_class in list(onto.object_properties()):
        ontology_class = str(onto_class)
        ontology_class = ontology_class.replace('ml-hierarchy.', '')
        if (ontology_class.lower() == ('has' + file_structure_splits[-1] + 'parameter').lower()):
            for param in onto_class.subclasses():
                actual_param = str(param).split('----')[-1]
                actual_param = actual_param.replace('ml-hierarchy.', '')
                init_function_parameter = init_function_parameter + actual_param + ':' + str(param.default_value.first()) + ','
            init_function_parameter = init_function_parameter[:-1]
            break
    if(init_function_parameter != ''):
        file_util.append_to_file(filename, python_content_creator.create_init_function(init_function_parameter))

    for onto_class in list(onto.object_properties()):
        ontology_class = str(onto_class)
        if (ontology_class.lower().replace('ml-hierarchy.', '') == ('has' + file_structure_splits[-1] + 'function').lower()):
            for function_ontology_class in list(onto_class.subclasses()):
                file_util.append_to_file(filename, python_content_creator.create_function(function_ontology_class, isSupervised, init_function_parameter))