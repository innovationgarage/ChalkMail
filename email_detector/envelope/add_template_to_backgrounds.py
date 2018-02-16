import scipy.ndimage as spim
import numpy as np
import pandas as pd
import PIL.ImageOps
from PIL import Image, ImageDraw
import os

def read_img(img_path, img_name):
    return Image.open(os.path.join(img_path, img_name)).convert("L")
    
def draw_strokes(strokes, img, x_offset=0, y_offset=0):
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        row = stroke[0]
        col = stroke[1]
        for i in range(len(row)-1):
            draw.line((row[i]+x_offset, col[i]+y_offset, row[i+1]+x_offset, col[i+1]+y_offset), fill=50, width=2)
    return img

def get_strokes(drawing):
    rows = np.array([drawing[i][0] for i in range(len(drawing))])
    cols = np.array([drawing[i][1] for i in range(len(drawing))])
    w_s = np.max(np.max(rows)) - np.min(np.min(rows))
    h_s = np.max(np.max(cols)) - np.min(np.min(cols))
    if np.random.random()>=0.5:
        strokes = [(r, c) for (r,c) in zip(rows, cols)]
    else:
        strokes = [(r, c) for (r,c) in zip(cols, rows)]
    return strokes, w_s, h_s
    
def draw_on_background(bg_path, bg_name, df, out_path):
    drawing_type = 'envelope'
    samples = df.sample(30)
    for i, drawing in enumerate(samples.drawing.values):
        bg = read_img(bg_path, bg_name)
        w_bg, h_bg = bg.size
        label = '%s.%d'%(drawing_type, i)
        try:
            strokes, w_s, h_s = get_strokes(drawing)
            strokes = np.array(strokes)
            scale_h = 20./(h_bg*1./h_s*1.)
            scale_w = 20./(w_bg*1./w_s*1.)
            scale_factor = np.array([scale_h, scale_w]).mean()
            strokes = np.array(strokes)/scale_factor
            # x_offset = (np.random.random() * w_bg) - w_s*scale_factor
            # y_offset = (np.random.random() * h_bg) - h_s*scale_factor
            x_offset = 0.8 * w_bg
            y_offset = 0.1 * h_bg
            img = draw_strokes(strokes, bg, x_offset, y_offset)
            print os.path.join(out_path, "%s.jpg"%label)
            img.save(os.path.join(out_path, "%s_%s_08_01.jpg"%(label, bg_name.split('.jpg')[0])), quality=100)
        except:
            print 'drawing %s failed!'%label
            
bg_path = 'envelope/tmp1/'
#bg_name = 'wb.2.jpg'
img_path = 'envelope/tmp/'
df = pd.read_json('envelope/full_raw_envelope.ndjson', lines=True)
for bg_name in os.listdir(bg_path):
    draw_on_background(bg_path, bg_name, df, img_path)
