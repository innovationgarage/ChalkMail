import scipy.ndimage as spim
import numpy as np
import pandas as pd
from skimage.draw import line_aa
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
    strokes = [(r, c) for (r,c) in zip(rows, cols)]
    return strokes
    
def draw_on_background(bg_path, bg_name):
    bg = read_img(bg_path, bg_name)
    img_path = "envelope/bg_images"
    drawing_type = 'envelope'
    df = pd.read_json('envelope/full_raw_envelope.ndjson', lines=True)
    for i, drawing in enumerate(df.drawing.values):
        label = '%s.%d'%(drawing_type, i)
        strokes = get_strokes(drawing)
        img = draw_strokes(strokes, bg)
        img.save(os.path.join(img_path, "%s.jpg"%label), quality=100)
