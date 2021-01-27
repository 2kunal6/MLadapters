def create_class(class_name, parent_class_list):
    if not parent_class_list:
        return 'class ' + class_name + ':\n\tpass\n'
    else:
        parent_class_list_str = list(map(str, parent_class_list))
        for i in range(len(parent_class_list_str)):
            parent_class_list_str[i] = parent_class_list_str[i].replace('ml-hierarchy.', '')

        class_body = 'class ' + class_name + '(' + "".join(parent_class_list_str) + '):\n'
        if("".join(parent_class_list_str) == 'MachineLearningAlgorithms'):
            class_body = class_body + '\tpass\n'
        return class_body

def create_function(function_ontology_class, isSupervised, imports_value):
    function_name = (str(function_ontology_class)).replace('ml-hierarchy.', '')
    function_name = function_name.split('----')[-1]
    parameter_list = function_ontology_class.subclasses()
    function_def = ''
    if not parameter_list:
        function_def = '\tdef ' + function_name + ':\n\t\t' + "'''" + str(function_ontology_class.comment.first()) + "'''" + '\n\t\t'
    else:
        param_order_dict = {}
        for param in parameter_list:
            default_value_decorated = str(param.default_value.first())
            if(default_value_decorated is not None and not default_value_decorated.replace('.', '').isdigit()):
                default_value_decorated = '\'' + default_value_decorated + '\''

            param_order_dict[int(str(param.parameter_position.first()))] = str(param).split('----')[-1] + '=' + default_value_decorated

        param_list = ''
        param_list_without_default_val = ''
        for k in sorted(param_order_dict):
            param_list = param_list + str(param_order_dict[k]).replace('ml-hierarchy.', '') + ','
            param_list_without_default_val = param_list_without_default_val + str(param_order_dict[k]).replace('ml-hierarchy.', '').split('=')[0] + ','
        param_list = param_list[:-1]
        param_list_without_default_val = param_list_without_default_val[:-1]
        function_def = '\tdef ' + function_name + '(self, ' + param_list + '):\n\t\t' + "'''" + str(function_ontology_class.comment.first()) + "'''" + '\n\t\t'

    if(function_name == 'fit'):
        if(isSupervised == '[True]'):
            function_def = function_def + 'self._model.fit(X, y)'
        else:
            function_def = function_def + 'self._model.fit(X)'
    elif(function_name == 'predict'):
        function_def = function_def + 'self._model.predict(X)'
    elif(function_name == '__init__'):
        function_def = function_def + 'self._model = ' + imports_value.split(' ')[-1] + '(' + param_list_without_default_val + ")\n"
    else:
        function_def = function_def + 'pass'

    return function_def
