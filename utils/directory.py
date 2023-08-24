import os

def get_project_dir():
    fwd = os.path.abspath(__file__)
    return fwd.replace(os.path.join("utils","directory.py"), "")


def path_to_package(dir_path:str):
    cpd = get_project_dir()
    print(dir_path, cpd)
    dir_path = dir_path.replace(cpd, "").replace(f"{os.path.sep}__init__.py", "")
    return dir_path.replace(os.path.sep, '.')