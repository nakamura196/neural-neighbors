import json, glob, os

o_dir = "assets/captioned_nearest_neighbors"
os.makedirs(o_dir, exist_ok=True)

captions = json.load(open('data/full-captions.json'))

for i in glob.glob('data/nearest_neighbors2/*.json'):
  try:
    with open(i) as f:
      
      j = json.load(f)
      for k in j:
        try:
          k['caption'] = captions[k['filename']]
        except:
          k['caption'] = ''

    with open(o_dir+'/' + os.path.basename(i), 'w') as out:
      json.dump(j, out)
  except:
    print(i)
