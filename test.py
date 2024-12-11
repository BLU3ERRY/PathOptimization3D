import os

path = input(" : ")
with open(path, "r") as f:
    lines = f.readlines()

filename = os.path.basename(path).replace(".csv", "_reversed.csv")
dirname = os.path.dirname(path)

with open(os.path.join(dirname, filename), "w") as f:
    for line in lines[::-1]:
        f.write(line)