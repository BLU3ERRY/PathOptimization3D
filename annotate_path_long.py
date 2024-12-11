import os

repo_dir = os.path.dirname(__file__)

path_from = os.path.join(repo_dir, "Data", "_reference_path_long.csv")

with open(path_from, "r") as f:
    lines = f.readlines()

path_to = os.path.join(repo_dir, "Data", "_reference_path_long_annotated.csv")

with open(path_to, "w") as f:
    [f.write(line.replace("\t", " ")[:-1] + " 0\n") for line in lines]