import os

repo_dir = os.path.dirname(__file__)

path_from = os.path.join(repo_dir, "Data", "_reference_path_short.csv")

with open(path_from, "r") as f:
    lines = f.readlines()

path_to = os.path.join(repo_dir, "Data", "_reference_path_short_annotated.csv")

isdanger = 0
epsilon = 0.01

with open(path_to, "w") as f:
    for line in lines:
        x, y, z = line.split(",\t")
        x = float(x)
        y = float(y)
        z = float(z[:-2])
        if (x - 24.000000) ** 2 + (y - 42.599998) ** 2 <= epsilon**2:
            isdanger = 1
        elif (x + 98.845665) ** 2 + (y - 51.403393) ** 2 <= epsilon**2:
            isdanger = 0
        f.write(line.replace("\t", " ")[:-1] + f" {isdanger}\n")