import numpy as np
import pandas as pd
from skimage.draw import line_aa
import PIL.ImageOps
from PIL import Image, ImageDraw
import os
import scipy.ndimage as spim

"""
This script reads the raw data ffrom Google drawing, and prepares them as training data for an object detection network. 
- read strokes and draw them into a single array (the image)
- apply some augmentation to add scatter to the data (scale, rotate, add pixel noise)
- label each image with a bounding box around the originam object(drawing)
"""

def main():
    n_samples = 9000
    transpose_probability = 0.5
    drawing_type = 'envelope'
    path_to_images = 'envelope/images/'
    path_to_labels = 'envelope/labels/'
    if not os.path.exists(path_to_labels):
        os.mkdir(path_to_labels)
    if not os.path.exists(path_to_images):
        os.mkdir(path_to_images)
    
    df = pd.read_json('envelope/full_raw_envelope.ndjson', lines=True)

    with open('envelope/labels/all_labels.csv', 'w') as f:
        f.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')
        for i, drawing in enumerate(df.drawing.values):
            label = '%s.%d'%(drawing_type, i)
            try:
                img = draw(drawing, label, transpose_probability)
                img = perform_augmentation(img)
                save_image(img, label)
                (h, w) = img.shape
                xmin, xmax, ymin, ymax = get_bbox(img)
                f.write('%s,%d,%d,%s,%d,%d,%d,%d\n'%(label,w,h,drawing_type,xmin,ymin,xmax,ymax))
#                draw_bbox(img, bg_val=0)
            except:
                print 'drawing %s failed!'%label
                
    # perform_augmentation(n_samples, source_dir=path_to_images)

#     with open('envelope/labels/augmented_labels.csv', 'w') as f:
#         f.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')
#         for image_name in os.listdir(os.path.join(path_to_images, 'output')):
# #            try:
#             image = Image.open(os.path.join(path_to_images, 'output', image_name))
#             image_array = np.array(image).astype('uint8')
#             (h, w, _) = image_array.shape
#             xmin, xmax, ymin, ymax = get_bbox(image_array[:,:,0], bg_val=255)
#             draw_bbox(image_array[:,:,0], bg_val=255)
#             f.write('%s,%d,%d,%s,%d,%d,%d,%d\n'%(image_name.split('.JPEG')[0],w,h,drawing_type,xmin,ymin,xmax,ymax))
#             # except:
#             #     print (image_name, ' is not an image dude!')
    
# def perform_augmentation(n_samples, source_dir):
#     p = Augmentor.Pipeline(source_dir)
#     p.flip_left_right(probability=0.4)
#     p.flip_top_bottom(probability=0.4)
#     p.resize(probability=0.4, width=600, height=600, resample_filter='ANTIALIAS')
#     p.rotate(probability=0.4, max_left_rotation=15, max_right_rotation=15)
#     p.sample(n_samples)
    
def draw(drawing, label, transpose_probability, pad=10):
    print 'drawing %s'%label
    rows = [drawing[i][0] for i in range(len(drawing))]
    cols = [drawing[i][1] for i in range(len(drawing))]
    strokes = [(r, c) for (r,c) in zip(rows, cols)]

    #find size of the canvas
    max_row = np.max(np.max(rows))
    max_col = np.max(np.max(cols))
    min_row = np.min(np.min(rows))
    min_col = np.min(np.min(cols))
    nrows = int(np.ceil(min_row + max_row))
    ncols = int(np.ceil(min_col + max_col))

    #initialize the canvas
    img = np.zeros((nrows, ncols), dtype=np.uint8)

    #draw strokes
    for j, stroke in enumerate(strokes):
        draw_stroke(stroke, img)

    #cut to size
    bg_val = 0
    pad = 10
    non_zeros = np.where(img!=bg_val)
    xmin = non_zeros[1].min()
    xmax = non_zeros[1].max()
    ymin = non_zeros[0].min()
    ymax = non_zeros[0].max()
    img = img[ymin-pad:ymax+pad, xmin-pad:xmax+pad]

    return img
        
def perform_augmentation(img, transpose_probability=0.5, min_zom_scale=0.2, max_zoom_scale=1.2):

    #rotate by 90 degrees
    #all these drawings are vertical by default
    if np.random.random()<=transpose_probability:
        img = img.T

    #zoom (-in or -out)
    zoom_scale = (max_zoom_scale - min_zom_scale) * np.random.random() + min_zom_scale
    img = spim.zoom(img, zoom_scale)

    #rotate by a random angle (<90 degrees)
    rotation_angle = 90. * np.random.random()
    img = spim.rotate(img, rotation_angle)

    return img

def save_image(img, label):
    im = Image.fromarray(img)
    inv_im = PIL.ImageOps.invert(im)
    inv_im.save('envelope/images/%s.jpg'%label, quality=100)

def draw_stroke(stroke, img):
    row = stroke[0]
    col = stroke[1]
    for i in range(len(row)-1):
        rr, cc, val = line_aa(row[i], col[i], row[i+1], col[i+1])
        img[rr, cc] = val * 255

def get_bbox(img, bg_val=0, pad=0):
    non_zeros = np.where(img!=bg_val)
    xmin = non_zeros[1].min()
    xmax = non_zeros[1].max()
    ymin = non_zeros[0].min()
    ymax = non_zeros[0].max()
    print xmin, ymin, xmax, ymax
    return xmin-pad, xmax+pad, ymin-pad, ymax+pad
    
def draw_bbox(img, bg_val=0):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    (h, w) = img.shape
    fig,ax = plt.subplots(1)
    ax.imshow(img, cmap='gray_r')
    non_zeros = np.where(img!=bg_val)
    xmin = non_zeros[1].min()
    xmax = non_zeros[1].max()
    ymin = non_zeros[0].min()
    ymax = non_zeros[0].max()
    rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    main()
    
