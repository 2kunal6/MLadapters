import os


def extract_entities(stmt, i, ind, start, end):
    return stmt[i + ind][stmt[i + ind].rfind(start) + len(start): stmt[i + ind].rfind(end)]


def read_pos_from_ontology(ontology_file):
    file = open(ontology_file, "r")
    content = file.readlines()
    pos_dict = {}
    for i, line in enumerate(content):
        if "<owl:Axiom>" in line:
            class_name = extract_entities(content, i, 1, "#", "\"")
            func_name = extract_entities(content, i, 5, "#", "\"")
            param_name = extract_entities(content, i, 6, "#", "\"").split("__")[-1]
            pos = float(extract_entities(content, i, 9, "\">", "</pos>"))
            if pos_dict.get(class_name):
                if pos_dict[class_name].get(func_name):
                    pos_dict[class_name][func_name].update({param_name: pos})
                else:
                    pos_dict[class_name][func_name] = {param_name: pos}
            else:
                entry = {class_name: {func_name: {param_name: pos}}}
                pos_dict.update(entry)
    print(pos_dict)
    return pos_dict


def test():
    read_pos_from_ontology(os.path.join(os.pardir, "ml_algorithms.owl"))


if __name__ == "__main__":
    test()
