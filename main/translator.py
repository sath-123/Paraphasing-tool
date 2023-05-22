import json
from operator import itemgetter
from itertools import groupby
from googletrans import Translator
f = open('/content/drive/MyDrive/annotations/captions_train2014.json', 'r')
x = f.read()
x = json.loads(x)
sortkeyfn = itemgetter('image_id')
x['annotations'].sort(key=sortkeyfn)
captions = []
for key,valuesiter in groupby(x['annotations'], key=sortkeyfn):
    captions.append(dict(type=key, items=list(v['caption'] for v in valuesiter)))

g = open('/content/drive/MyDrive/caption2014.txt', 'a')
translator = Translator()
for k in range(int(len(captions)/10)):
  for i in range(10):
    for j in range(len(captions[k*10+i]['items'])):
      translations = translator.translate(captions[k*10+i]['items'][j], dest = 'te')
      captions[k*10+i]['items'][j] = translations.text
  print(captions[k+10+i]['items'])

g.write(captions)