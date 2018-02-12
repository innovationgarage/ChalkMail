import numpy as np
import pandas as pd
from skimage.draw import line_aa
import PIL.ImageOps
from PIL import Image, ImageDraw
import pdb

def main():
    df = pd.read_json('envelope/full_raw_envelope.ndjson', lines=True, )
    drawing_type = 'envelope'
    
    with open('labels/all_labels.csv', 'w') as f:
        f.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')
        for i, drawing in enumerate(df.drawing.values):
            label = '%s.%d'%(drawing_type, i)
            try:
                img = draw(drawing, label)
                (h, w) = img.shape
                xmin, xmax, ymin, ymax = get_bbox(img)
                f.write('%s,%d,%d,%s,%d,%d,%d,%d\n'%(label,w,h,drawing_type,xmin,ymin,xmax,ymax))
            except:
                print 'drawing %s failed!'%label
    
def draw(drawing, label):
    print 'drawing %s'%label
    rows = [drawing[i][0] for i in range(len(drawing))]
    cols = [drawing[i][1] for i in range(len(drawing))]

    strokes = [(r, c) for (r,c) in zip(rows, cols)]
    
    max_row = np.max(np.max(rows))
    max_col = np.max(np.max(cols))
    min_row = np.min(np.min(rows))
    min_col = np.min(np.min(cols))
    
    nrows = int(np.ceil(min_row + max_row))
    ncols = int(np.ceil(min_col + max_col))

    #pdb.set_trace()
    
    img = np.zeros((nrows, ncols), dtype=np.uint8)
    
    for j, stroke in enumerate(strokes):
        draw_stroke(stroke, img)

    im = Image.fromarray(img)
    inv_im = PIL.ImageOps.invert(im)
    inv_im.save('envelope/images/%s.jpg'%label, quality=100)
    return img

def draw_stroke(stroke, img):
    row = stroke[0]
    col = stroke[1]
    for i in range(len(row)-1):
        rr, cc, val = line_aa(row[i], col[i], row[i+1], col[i+1])
        img[rr, cc] = val * 255

def get_bbox(img):
    non_zeros = np.where(img!=0)
    xmin = non_zeros[1].min()
    xmax = non_zeros[1].max()
    ymin = non_zeros[0].min()
    ymax = non_zeros[0].max()
    return xmin, xmax, ymin, ymax
    
def draw_bbox(img):
    import matplotlib.patches as patches
    (h, w) = img.shape
    fig,ax = plt.subplots(1)
    ax.imshow(img, cmap='gray_r')
    non_zeros = np.where(img!=0)
    xmin = non_zeros[1].min()
    xmax = non_zeros[1].max()
    ymin = non_zeros[0].min()
    ymax = non_zeros[0].max()
    rect = patches.Rectangle((xmin,ymin), w, h, linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    main()
    
