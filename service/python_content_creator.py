def create_class(class_name, parent_class_list):
    if not parent_class_list:
        return 'class ' + class_name + ':\n'
    else:
        parent_class_list_str = list(map(str, parent_class_list))
        for i in range(len(parent_class_list_str)):
            parent_class_list_str[i] = parent_class_list_str[i].replace('ml-hierarchy.', '')
        return 'class ' + class_name + '(' + "".join(parent_class_list_str) + '):\n'

def create_function(function_ontology_class, isSupervised):
    function_name = (str(function_ontology_class)).replace('ml-hierarchy.', '')
    function_name = function_name.split('----')[-1]
    parameter_list = function_ontology_class.subclasses()
    function_def = ''
    if not parameter_list:
        function_def = '\tdef ' + function_name + ':\n\t' + "'''" + str(function_ontology_class.comment) + "'''" + '\n\t\t'
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
        function_def = '\tdef ' + function_name + '(' + param_list + '):\n\t' + "'''" + str(function_ontology_class.comment) + "'''" + '\n\t\t'

    if(function_name == 'fit'):
        if(isSupervised == '[True]'):
            function_def = function_def + 'self._model.fit(X, y)'
        else:
            function_def = function_def + 'self._model.fit(X)'
    elif(function_name == 'predict'):
        function_def = function_def + 'self._model.predict(X)'
    else:
        function_def = function_def + 'pass'

    return function_def
