def create_class(class_name, parent_class_list):
    if not parent_class_list:
        return 'class ' + class_name + ':\n'
    else:
        parent_class_list_str = list(map(str, parent_class_list))
        for i in range(len(parent_class_list_str)):
            parent_class_list_str[i] = parent_class_list_str[i].replace('ml-hierarchy.', '')
        return 'class ' + class_name + '(' + "".join(parent_class_list_str) + '):\n'

def create_function(function_name, parameter_list):
    if not parameter_list:
        return '\tdef ' + function_name + ':\n\tpass'
    else:
        param_order_dict = {}
        for param in parameter_list:
            pos = str(list(param.subclasses())[0])
            pos = int(pos.replace('ml-hierarchy.', ''))
            param_order_dict[pos] = param

        param_list = ''
        for k in sorted(param_order_dict):
            param_list = param_list + str(param_order_dict[k]).replace('ml-hierarchy.', '') + ','
        param_list = param_list[:-1]
        return '\tdef ' + function_name + '(' + param_list + '):\n\t\tpass'