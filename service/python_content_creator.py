def create_class(class_name, parent_class_list):
    if not parent_class_list:
        return 'class ' + class_name + ':\n\tpass'
    else:
        parent_class_list_str = list(map(str, parent_class_list))
        for i in range(len(parent_class_list_str)):
            parent_class_list_str[i] = parent_class_list_str[i].replace('ml-hierarchy.', '')
        return 'class ' + class_name + '(' + "".join(parent_class_list_str) + '):\n\t\tpass'

def create_function(function_name, parameter_list):
    if not parameter_list:
        return '\tdef ' + function_name + ':\n\tpass'
    else:
        parameter_list_str = list(map(str, parameter_list))
        for i in range(len(parameter_list_str)):
            parameter_list_str[i] = parameter_list_str[i].replace('ml-hierarchy.', '')
        return '\tdef ' + function_name + '(' + "".join(parameter_list_str) + '):\n\t\tpass'