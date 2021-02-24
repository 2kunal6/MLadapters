import os

member_propagation = {}
cp_map = {}
pos_map = {}

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


def generate_model_init_from_template(node, inh_vars, variables):
    stmt = ""
    if node.core_import:
        lib_name = node.core_import.first().split()[-1]
        variables.extend(inh_vars)
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
        inh_vars = get_inherited_vars(node)
        params = ["{var}={var}".format(var=var) for var in inh_vars]
        param = ", ".join(params)
        if inh_vars:
            stmt = stmt + "\n\t\t{parent}.__init__(self, {params})".format(
                parent=cp_map[node].label.first(), params=param)
        stmt = stmt + generate_model_init_from_template(node, inh_vars, variables)
        return stmt
    else:
        variables = [obj.label.first() for obj in target]
        stmts = ["{var}={var}".format(var=var) for var in variables]
        params = ",\n\t\t\t".join(stmts)
        return "return self.model.{func_name}({params})".format(
            func_name=func_name,
            params=params
        )


def get_pos(node, func_name, obj):
    pos = pos_map.get(node.label.first(), {}).get(func_name, {}).get(obj.label.first())
    while cp_map[node]:
        if not pos:
            parent = cp_map[node]
            pos = pos_map.get(parent.label.first(), {}).get(func_name, {}).get(obj.label.first())
            node = parent
        if pos:
            return pos
    return None


def get_sorted_subarray(vars, node, func_name):
    pos_list = []
    empty_pos = 1
    for obj in vars:
        pos = get_pos(node, func_name, obj)
        if not pos:
            pos = len(vars) + empty_pos
            empty_pos += 1
        pos_list.append(pos)
    return [x for _, x in sorted(zip(pos_list, vars))]


def get_ordered_params(node, func_name, variables):
    non_default_vars = []
    default_vars = []
    for obj in variables:
        if not obj.default:
            non_default_vars.append(obj)
        else:
            default_vars.append(obj)
    ordered_vars = get_sorted_subarray(non_default_vars, node, func_name)
    ordered_vars.extend(get_sorted_subarray(default_vars, node, func_name))
    return ordered_vars


def generate_function_param_from_template(node, func_name, target):
    params = ""
    param_template = "{var}={value}"
    inh_var = member_propagation.get(node)
    variables = target
    if inh_var:
        variables = inh_var + target
    variables = get_ordered_params(node, func_name, variables)
    for obj in variables:
        if not obj.default:
            param = obj.label.first()
        else:
            param = param_template.format(
                var=obj.label.first(), value=obj.default.first())
        params = params + ", " + param
    return params


def generate_function_from_template(node, func):
    func_template = """
\tdef {func_name}(self{params}):
\t\t{statements}\n"""
    var = func.label.first()
    target = eval("node." + var)
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
            params=generate_function_param_from_template(node, func.label.first(), target),
            statements=generate_function_body_from_template(node, func, target)
        )
        return func


def get_inherited_vars(node):
    inherited_variables = member_propagation.get(node)
    if inherited_variables:
        var = [obj.label.first() for obj in inherited_variables]
        return var
    return ""


def generate_init_by_member_propagation(node):
    func_template = """
\tdef __init__(self, {params}):
\t\t{parent}.__init__(self, {params}){stmt}\n"""
    inh_vars = get_inherited_vars(node)
    return func_template.format(
        params=", ".join(inh_vars),
        parent=cp_map[node].label.first(),
        stmt=generate_model_init_from_template(node, inh_vars, [])
    )


def generate_functions_from_template(node):
    function_names = set()
    func_data = ""
    for func in node.get_class_properties():
        if func.label:
            func_data = func_data + generate_function_from_template(node, func)
            function_names.add(func.label[0])
    if not func_data and member_propagation.get(node):
        func_data = generate_init_by_member_propagation(node)
    return func_data


def create_file_contents(file_path, node, child_parent_map, pos_dict):
    print(node.label)
    global pos_map
    pos_map = pos_dict
    global cp_map
    cp_map = child_parent_map
    parent = cp_map[node]
    func_data = generate_functions_from_template(node)
    if not func_data:
        func_data = "pass"
    content = generate_class_from_template(file_path, node, parent, func_data)
    return content
