import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json

with open('info.labels',encoding='utf8') as f:
    txt = f.read()
    content = json.loads(txt)
    # json_cnt = json.dumps(content, indent=2)


target = "training/00055.jpg.3ic37j9h.ingestion-74bd75dd6d-g9g4f.jpg"

# print(content['files'][0])

for file in content['files']:
    if file['path'] == target:
        bbox = file['boundingBoxes'][0]
        #print(file)
        break

print(bbox)

img = Image.open(f'./{target}')

shape = [(bbox['x'],bbox['y']),(bbox['x']+bbox['width'],bbox['y']+bbox['height'])]

img2 = ImageDraw.Draw(img)
img2.rectangle(shape, outline="green")

img.show()