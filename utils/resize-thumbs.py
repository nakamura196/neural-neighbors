import glob, os

o_dir = "assets/images/resized"
os.makedirs(o_dir, exist_ok=True)

for i in glob.glob('assets/images/thumbs/*.jpg'):
  os.system('convert ' + i + ' -sampling-factor 4:2:0 -quality 85 ' +
            o_dir+'/' + os.path.basename(i))
