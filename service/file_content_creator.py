member_propagation = {}
cp_map = {}

template = """
{import_statements}
class {class_name}({parent}):
    {functions}
"""


def generate_imports_from_template(node):
    import_template = """\n{lib}\n{core}"""
    core = node.core_import
    lib = node.lib_import
    return import_template.format(
        lib=node.lib_import.first() if lib else "",
        core=node.core_import.first() if core else "")


def generate_class_from_template(node, parent, functions):
    return template.format(
        import_statements=generate_imports_from_template(node),
        class_name=node.label.first(),
        parent=parent.label.first() if parent else "",
        functions=functions)


def function_name_handler(func_name):
    if func_name == "init":
        return "__init__"
    return func_name


def generate_function_body_from_template(node, func, target):
    print("FUNC: ", func.label.first())
    if func.label.first() == "init":
        variables = [obj.label.first() for obj in target]
        stmts = ["self.{var} = {var}".format(var=var) for var in variables]
        stmt = "\n\t".join(stmts)
        params = get_inherited_params(node)
        if params:
            stmt = stmt + "\n\t{parent}.__init__(self, {params})".format(
                parent=cp_map[node].label.first(), params=params)
        if node.core_import:
            lib_name = node.core_import.first().split()[-1]
            variables.extend(params.split(", "))
            variables = set(variables)
            stmts = ["{var} = self.{var}".format(var=var) for var in variables]
            func_params = ",\n\t\t\t".join(stmts)
            stmt = stmt + "\n\tself.model = {lib_name}({func_params})".format(
                lib_name=lib_name, func_params=func_params)
        return stmt


def generate_function_param_from_template(node, target):
    params = ""
    param_template = "{var} = {value}"
    for obj in target:
        if not obj.default:
            param = obj.label.first()
        else:
            param = param_template.format(
                var=obj.label.first(), value=obj.default.first())
        params = params + ", " + param
    inherited_params = get_inherited_params(node)
    if inherited_params:
        params = ", " + inherited_params + params
    return params


def generate_function_from_template(node, func):
    func_template = """
    def {func_name}(self{params}):
        {statements}

    """
    print("LABEL: ", func.label[0])
    var = func.label.first()
    target = eval("node." + var)
    print("TARGET: ", target)
    if var == "init":
        for child in node.descendants():
            if child != node:
                if member_propagation.get(child):
                    member_propagation[child].extend(target)
                else:
                    member_propagation[child] = target
    if target:
        func = func_template.format(
            func_name=function_name_handler(func.label.first()),
            params=generate_function_param_from_template(node, target),
            statements=generate_function_body_from_template(node, func, target)
        )
        # print(func)
        # print(target[0].label.first())
        return func


def get_inherited_params(node):
    inherited_variables = member_propagation.get(node)
    if inherited_variables:
        var = [obj.label.first() for obj in inherited_variables]
        params = ", ".join(var)
        return params
    return ""


def generate_init_by_member_propagation(node):
    func_template = """
    def __init__(self, {params}):
        {parent}.__init__(self, {params})
    """
    return func_template.format(
        params=get_inherited_params(node),
        parent=cp_map[node].label.first()
    )


def generate_functions_from_template(node):
    function_names = set()
    func_data = ""
    for func in node.get_class_properties():
        if func.label:
            func_data = func_data + generate_function_from_template(node, func)
            function_names.add(func.label[0])
    # print(func_data)
    if not func_data and member_propagation.get(node):
        func_data = generate_init_by_member_propagation(node)
    return func_data


def create_file_contents(file, node, child_parent_map):
    print(node.label)
    global cp_map
    cp_map = child_parent_map
    parent = cp_map[node]
    if parent:
        print("PARENT: ", parent)
    func_data = generate_functions_from_template(node)
    if not func_data:
        func_data = "pass"
    print(generate_class_from_template(node, parent, func_data))
    # print("MEM_PROP: ", member_propagation)