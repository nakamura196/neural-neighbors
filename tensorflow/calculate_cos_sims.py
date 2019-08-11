import numpy as np
import glob
import timeit
from helper import *
import json, os

CANDIDATES = 100
OUTPUT_DIR = "../data/nearest_neighbors2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = glob.glob("../data/image_vectors/*.npy")

features = []

for i in range(len(files)):

  file = files[i]

  a = np.load(file)  # 保存したファイルを呼び出してみる。

  features.append(a)

# Vectorized Computation

for i in range(len(features)):

  print(i)

  query_feat = features[i]

  sims = [(k, round(1 - spatial.distance.cosine(query_feat, v), 3))
          for k, v in enumerate(features)]

  obj_array = []

  flg = False

  for obj in sorted(sims, key=operator.itemgetter(1), reverse=True)[:CANDIDATES + 2]:
    filename = files[obj[0]].split("/")[-1].split(".")[0]
    similarity = obj[1]
    j = {
        "filename": filename,
        "similarity": similarity
    }
    if flg:
        obj_array.append(j)

    flg = True

  filename2 = files[i].split("/")[-1].split(".")[0]

  fw = open(OUTPUT_DIR+"/"+filename2+".json", 'w')
  json.dump(obj_array, fw, ensure_ascii=False, indent=4,
            sort_keys=True, separators=(',', ': '))
