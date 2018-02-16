import scipy.ndimage as spim
import numpy as np
import pandas as pd
import PIL.ImageOps
from PIL import Image, ImageDraw
import os

def read_img(img_path, img_name):
    return Image.open(os.path.join(img_path, img_name)).convert("L")
    
def draw_strokes(strokes, img):
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        row = stroke[0]
        col = stroke[1]
        for i in range(len(row)-1):
            draw.line((row[i], col[i], row[i+1], col[i+1]), fill=50, width=10)
    return img

def get_strokes(drawing):
    rows = [drawing[i][0] for i in range(len(drawing))]
    cols = [drawing[i][1] for i in range(len(drawing))]
    w_s = np.max(rows) - np.min(rows)
    h_s = np.max(cols) - np.min(cols)
    strokes = [(r, c) for (r,c) in zip(rows, cols)]
    return strokes, w_s, h_s
    
def draw_on_background(bg_path, bg_name):
    bg = read_img(bg_path, bg_name)
    w_bg, h_bg = bg.size
    img_path = "envelope/bg_images"
    drawing_type = 'envelope'
    df = pd.read_json('envelope/full_raw_envelope.ndjson', lines=True)
    for i, drawing in enumerate(df.drawing.values[:10]):
        label = '%s.%d'%(drawing_type, i)
        strokes, w_s, h_s = get_strokes(drawing)
        strokes = np.array(strokes)
        scale_h = 15./(h_bg*1./h_s*1.)
        scale_w = 15./(w_bg*1./w_s*1.)
        scale_factor = np.array([scale_h, scale_w]).mean()
        strokes = np.array(strokes)/scale_factor
        img = draw_strokes(strokes, bg)
        img.save(os.path.join(img_path, "%s.jpg"%label), quality=100)


bg_path = 'envelope/backgrounds/'
bg_name = 'wb.21.jpg'
img_path = 'envelope/res/'
draw_on_background(bg_path, bg_name)
