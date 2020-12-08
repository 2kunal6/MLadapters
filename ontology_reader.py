from owlready2 import *
from util import file_creator

#onto = get_ontology("https://github.com/2kunal6/SemanticWebLab/blob/master/ml-hierarchy.owl")
onto = get_ontology("https://raw.githubusercontent.com/2kunal6/SemanticWebLab/master/ml-hierarchy.owl")

onto.load()

print(list(onto.classes()))

for onto_class in list(onto.classes()):
    ontology_class = str(onto_class)
    if (ontology_class == "ml-hierarchy.MachineLearningAlgorithms"):
        for algorithms_onto_class in list(onto_class.descendants()):
            algorithms_onto_class = str(algorithms_onto_class)
            algorithms_onto_class = algorithms_onto_class.replace('ml-hierarchy.', '')
            file_creator.create_file('ml_algorithms/' + algorithms_onto_class + '.py')
        break
