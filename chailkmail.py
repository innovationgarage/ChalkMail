import email_detector.deep_learning.my_model_inference
from google.cloud import vision
from google.cloud.vision import types
import argparse
import io
import cv2
import sys
import os.path

def detect_text(img, bbox):
    client = vision.ImageAnnotatorClient()

    img = cut_image(img, bbox)

    rotate = bbox[2] < bbox[3]
    if rotate:
        # height > width, so rotate, clockwize just because
        img=cv2.transpose(img)
        img=cv2.flip(img,flipCode=1)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,5)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite("xyzzy.jpg", img)
    
    ret, data = cv2.imencode(".jpg", img)
    data = data.tobytes()
    image = types.Image(content=data)

    response = client.text_detection(image=image)

    def fixpos(vertex):
        if rotate:
            return (bbox[2] - vertex.y, bbox[3] - vertex.x)
        else:
            return (vertex.x, vertex.y)
    
    def textbbox(text):
        return merge_bboxes(*(
            email_detector.deep_learning.my_model_inference.pos2bbox(
                (bbox[0] + x, bbox[1] + y, bbox[0] + x, bbox[1] + y))
            for (x, y)
            in (fixpos(vertex)
             for vertex in text.bounding_poly.vertices)))
    
    return {textbbox(text): text.description
            for text in response.text_annotations[:1]}

def merge_bboxes(*bboxes):
    (x1, y1, x2, y2) = zip(*[email_detector.deep_learning.my_model_inference.bbox2pos(b)
                             for b in bboxes])
    return email_detector.deep_learning.my_model_inference.pos2bbox((min(x1+x2), min(y1+y2), max(x1+x2), max(y1+y2)))

def cut_image(img, bbox, hmargin=0, vmargin=0):
    (x, y, w, h) = bbox
    x -= hmargin
    y -= vmargin
    w += 2*hmargin
    h += 2*vmargin
    return img[y:y + h, x:x + w]

img = cv2.imread(sys.argv[1])
labels = email_detector.deep_learning.my_model_inference.find_labels(img)
print "Labels", labels
merged_bbox = merge_bboxes(*labels.keys())
print "Merged", merged_bbox

out = img.copy()
for bbox, info in labels.iteritems():
    p = email_detector.deep_learning.my_model_inference.bbox2pos(bbox)
    cv2.rectangle(out, (p[0],p[1]), (p[2],p[3]), (0,0,255),2)
    cv2.putText(out, ("%(class_name)s (%(score).2f" % info).encode("utf-8"), (p[0], p[1]), cv2.FONT_HERSHEY_COMPLEX, 3.0, (0, 0, 255), 2)

p = email_detector.deep_learning.my_model_inference.bbox2pos(merged_bbox)
cv2.rectangle(out, (p[0],p[1]), (p[2],p[3]), (0,255,00),3)

cv2.imwrite(os.path.splitext(sys.argv[1])[0] + ".out.1.jpg", out)

texts = detect_text(img, merged_bbox)
print "Texts", texts

for bbox, txt in texts.iteritems():
    p = email_detector.deep_learning.my_model_inference.bbox2pos(bbox)
    cv2.rectangle(out, (p[0],p[1]), (p[2],p[3]), (255,0,0),2)
    cv2.putText(out, txt.encode("utf-8"), (p[0], p[1]), cv2.FONT_HERSHEY_COMPLEX, 3.0, (255, 0, 0), 2)

cv2.imwrite(os.path.splitext(sys.argv[1])[0] + ".out.2.jpg", out)
