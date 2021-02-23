import os

member_propagation = {}
cp_map = {}

template = """{import_statements}\n\n
class {class_name}({parent}):
\t{functions}
"""


def generate_relative_imports(file_path):
    file_parts = file_path.split(os.path.sep)
    if len(file_parts) < 2:
        return ''
    if len(file_parts) == 2:
        return "from {parent}._{parent} import {parent}".format(
            parent=file_parts[0])
    class_name = file_parts[-2]
    file_parts[-2] = "_" + file_parts[-2]
    return "from {parent} import {class_name}".format(
        parent='.'.join(file_parts[:-1]), class_name=class_name)


def generate_imports_from_template(node, file_path):
    import_template = "{lib}{core}{relative}"
    core = node.core_import
    lib = node.lib_import
    relative = generate_relative_imports(file_path)
    return import_template.format(
        lib='\n' + node.lib_import.first() if lib else "",
        core='\n' + node.core_import.first() if core else "",
        relative='\n' + relative if relative else "")


def generate_class_from_template(file_path, node, parent, functions):
    return template.format(
        import_statements=generate_imports_from_template(node, file_path),
        class_name=node.label.first(),
        parent=parent.label.first() if parent else "",
        functions=functions)


def function_name_handler(func_name):
    if func_name == "init":
        return "__init__"
    return func_name


def generate_model_init_from_template(node, params, variables):
    stmt = ""
    if node.core_import:
        lib_name = node.core_import.first().split()[-1]
        variables.extend(params.split(", "))
        variables = set(variables)
        stmts = ["{var} = self.{var}".format(var=var) for var in variables]
        func_params = ",\n\t\t\t".join(stmts)
        stmt_template = "\n\t\t{returns} {lib_name}({func_params})"
        assign_return = "{var} =".format(var=node.returns.first())
        stmt = stmt + stmt_template.format(
            returns=assign_return, lib_name=lib_name, func_params=func_params)
    return stmt


def generate_function_body_from_template(node, func, target):
    print("FUNC: ", func.label.first())
    func_name = func.label.first()
    if func_name == "init":
        variables = [obj.label.first() for obj in target]
        stmts = ["self.{var} = {var}".format(var=var) for var in variables]
        stmt = "\n\t\t".join(stmts)
        params = get_inherited_params(node)
        if params:
            stmt = stmt + "\n\t\t{parent}.__init__(self, {params})".format(
                parent=cp_map[node].label.first(), params=params)
        stmt = stmt + generate_model_init_from_template(node, params, variables)
        return stmt
    else:
        variables = [obj.label.first() for obj in target]
        stmts = ["{var}={var}".format(var=var) for var in variables]
        params = ",\n\t\t\t".join(stmts)
        return "return self.model.{func_name}({params})".format(
            func_name=func_name,
            params=params
        )


def generate_function_param_from_template(node, target):
    params = ""
    param_template = "{var} = {value}"
    inh_var = member_propagation.get(node)
    variables = target
    if inh_var:
        variables = inh_var + target
    for obj in variables:
    #for obj in target:
        if not obj.default:
            param = obj.label.first()
        else:
            param = param_template.format(
                var=obj.label.first(), value=obj.default.first())
        params = params + ", " + param
    #inherited_params = get_inherited_params(node)
    #if inherited_params:
    #    params = ", " + inherited_params + params
    return params


def generate_function_from_template(node, func):
    func_template = """
\tdef {func_name}(self{params}):
\t\t{statements}\n"""
    print("\n---------------------")
    print("LABEL: ", func.label[0])
    var = func.label.first()
    target = eval("node." + var)
    print("TARGET: ", target)
    if var == "init":
        for child in node.descendants():
            if child != node:
                if member_propagation.get(child):
                    member_propagation[child] = member_propagation[child] + target
                else:
                    member_propagation[child] = target
    if target:
        func = func_template.format(
            func_name=function_name_handler(func.label.first()),
            params=generate_function_param_from_template(node, target),
            statements=generate_function_body_from_template(node, func, target)
        )
        return func


def get_inherited_params(node):
    inherited_variables = member_propagation.get(node)
    print("INH_P: ", inherited_variables)
    if inherited_variables:
        var = [obj.label.first() for obj in inherited_variables]
        params = ", ".join(var)
        return params
    return ""


def generate_init_by_member_propagation(node):
    func_template = """
\tdef __init__(self, {params}):
\t\t{parent}.__init__(self, {params}){stmt}\n"""
    params = get_inherited_params(node)
    return func_template.format(
        params=params,
        parent=cp_map[node].label.first(),
        stmt=generate_model_init_from_template(node, params, [])
    )


def generate_functions_from_template(node):
    function_names = set()
    func_data = ""
    print( node.get_class_properties())
    for func in node.get_class_properties():
        if func.label:
            func_data = func_data + generate_function_from_template(node, func)
            function_names.add(func.label[0])
    if not func_data and member_propagation.get(node):
        func_data = generate_init_by_member_propagation(node)
    return func_data


def create_file_contents(file_path, node, child_parent_map):
    print(node.label)
    global cp_map
    cp_map = child_parent_map
    parent = cp_map[node]
    if parent:
        print("PARENT: ", parent)
    func_data = generate_functions_from_template(node)
    if not func_data:
        func_data = "pass"
    content = generate_class_from_template(file_path, node, parent, func_data)
    #print(content)
    return content
