mkdir data
python utils/create-captions.py
rm -rf assets/images/resized
python utils/resize-thumbs.py
rm -rf assets/captioned*
python utils/add-captions-to-nn.py
python utils/create-random-selections.py
