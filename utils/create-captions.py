import glob, random, os, json

files = glob.glob(
    "assets/images/thumbs/*.jpg")

captions = {}

for file in files:
  filename = file.split("/")[-1].split(".")[0]
  captions[filename] = filename

fw = open("data/full-captions.json", 'w')
json.dump(captions, fw, ensure_ascii=False, indent=4,
          sort_keys=True, separators=(',', ': '))
