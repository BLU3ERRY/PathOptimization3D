from Source.utilities import smooth_reference_path
import numpy as np
import trimesh

with open(input(" : "), "r") as f:
    lines = f.readlines()

data = []

for line in lines:
    data.append(list(map(float, line.replace("\t", " ").replace("\n", " ").replace(",", " ").split())))

path, isdanger = smooth_reference_path(np.array(data), smooth_factor=1.0)

scene = trimesh.Scene()
pc = trimesh.PointCloud(path, [255, 0, 0])
scene.add_geometry(pc)

scene.show()