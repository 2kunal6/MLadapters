from owlready2 import *
from util import file_creator

#onto = get_ontology("https://github.com/2kunal6/SemanticWebLab/blob/master/ml-hierarchy.owl")
onto = get_ontology("https://raw.githubusercontent.com/2kunal6/SemanticWebLab/master/ml-hierarchy.owl")

onto.load()

print(list(onto.classes()))

for ontology_class in list(onto.classes()):
    ontology_class = str(ontology_class)
    ontology_class = ontology_class.replace('ml-hierarchy.', '')
    file_creator.create_file('ml_algorithms/' + ontology_class + '.py')
