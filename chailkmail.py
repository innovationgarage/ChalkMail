import email_detector.deep_learning.my_model_inference
from google.cloud import vision
from google.cloud.vision import types
import argparse
import io
import cv2
import sys
import os.path
import datetime
import threading

class Interact(threading.Thread):
    def __init__(self, *arg, **kw):
        threading.Thread.__init__(self, *arg, **kw)
        self.input = None
        self.output = None
        
    def run(self):
        cv2.namedWindow("image")
        camera = cv2.VideoCapture(0)
        for i in xrange(30):
            camera.read()
        while True:
            retval, img = camera.read()
            self.input = img
            if self.output is not None:
                cv2.imshow('image', self.output)
            else:
                cv2.imshow('image', self.input)
            cv2.waitKey(1)


def detect_text(img, bbox):
    client = vision.ImageAnnotatorClient()

    img = cut_image(img, bbox)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,5)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    rotate = bbox[2] < bbox[3]
    if rotate:
        # height > width, so rotate, clockwize just because
        img=cv2.transpose(img)
        img=cv2.flip(img,flipCode=1)        

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

def diff_bboxes(b1, b2):
    p1 = email_detector.deep_learning.my_model_inference.bbox2pos(b1)
    p2 = email_detector.deep_learning.my_model_inference.bbox2pos(b2)

    # ix, iy are the coordinates of the box between b1 and b2
    if p1[0] < p2[0]:
        ix = (p1[2], p2[0])
    else:
        ix = (p2[2], p1[0])
    if p1[1] < p2[1]:
        iy = (p1[3], p2[1])
    else:
        iy = (p2[3], p1[1])

    if abs(p1[0] - p2[0]) > abs(p1[1] - p2[1]):
        # horizontal
        return email_detector.deep_learning.my_model_inference.pos2bbox((ix[0], min(p1[1], p1[3], p2[1], p2[3]), ix[1], max(p1[1], p1[3], p2[1], p2[3])))
    else:
        return email_detector.deep_learning.my_model_inference.pos2bbox((min(p1[0], p1[2], p2[0], p2[2]), iy[0], max(p1[0], p1[2], p2[0], p2[2]), iy[1]))
    
def cut_image(img, bbox, hmargin=0, vmargin=0):
    (x, y, w, h) = bbox
    x -= hmargin
    y -= vmargin
    w += 2*hmargin
    h += 2*vmargin
    return img[y:y + h, x:x + w]

def draw_labels(out, labels):
    for bbox, info in labels.iteritems():
        p = email_detector.deep_learning.my_model_inference.bbox2pos(bbox)
        cv2.rectangle(out, (p[0],p[1]), (p[2],p[3]), (0,0,255),2)
        cv2.putText(out, ("%(class_name)s (%(score).2f" % info).encode("utf-8"), (p[0], p[1]), cv2.FONT_HERSHEY_COMPLEX, 3.0, (0, 0, 255), 2)

def draw_merged_bbox(out, bbox):
    if bbox is None: return
    p = email_detector.deep_learning.my_model_inference.bbox2pos(bbox)
    cv2.rectangle(out, (p[0],p[1]), (p[2],p[3]), (0,255,00),3)

def draw_texts(out, texts):
    if texts is None: return
    for bbox, txt in texts.iteritems():
        p = email_detector.deep_learning.my_model_inference.bbox2pos(bbox)
        cv2.rectangle(out, (p[0],p[1]), (p[2],p[3]), (255,0,0),2)
        cv2.putText(out, txt.encode("utf-8"), (p[0], p[1]), cv2.FONT_HERSHEY_COMPLEX, 3.0, (255, 0, 0), 2)

def read_emails(img):
    labels = email_detector.deep_learning.my_model_inference.find_labels(img)
    merged_bbox = texts = None
    if len(labels) == 2:
        merged_bbox = diff_bboxes(*labels.keys())
        texts = detect_text(img, merged_bbox)
    return labels, merged_bbox, texts

def say(txt):
    os.system("echo '%s' | festival --tts" % txt)

if __name__ == '__main__':
    if sys.argv[1:]:
        img = cv2.imread(sys.argv[1])
        labels, merged_bbox, texts = read_emails(img)
        out = img.copy()
        draw_labels(out, labels)
        draw_merged_bbox(out, merged_bbox)
        draw_texts(out, texts)
        cv2.imwrite(os.path.splitext(sys.argv[1])[0] + ".out.jpg", out)
    else:
        try:
            interact = Interact()
            interact.start()

            say("We salute you")

            spamtime = None

            last_frametime = datetime.datetime.now()
            while True:
                img = interact.input
                if img is None: continue
                print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                labels, merged_bbox, texts = read_emails(img)

                out = img.copy()
                draw_labels(out, labels)
                draw_merged_bbox(out, merged_bbox)
                draw_texts(out, texts)
                interact.output = out

                if len(labels) == 1:
                    say("Ready at your command")
                if texts is not None:
                    if spamtime is None:
                        spamtime = datetime.datetime.now()
                        say("Ready to spam %s" % texts.values()[0])
                    elif spamtime - datetime.datetime.now() > datetime.timedelta(seconds=10):
                        say("To late. Spamming %s in three, two, one. Spam." % texts.values()[0])
                        spamtime = None
                else:
                    if spamtime is not None:
                        say("git revert. git revert. git revert.")
                    spamtime = None

                frametime = datetime.datetime.now()
                print "Frametime %s seconds per frame" % (frametime - last_frametime).total_seconds()
                last_frametime = frametime
        except Exception, e:
            import traceback
            import sys
            import pdb
            print e
            traceback.print_exc()
            sys.last_traceback = sys.exc_info()[2]
            pdb.pm()
