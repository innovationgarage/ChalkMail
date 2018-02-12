import sys
import cv2
import json
import utils_image

def main(image_filename, response_filename, result_filename):
    #read original file
    im = utils_image.read_image(image_filename)

    with open(response_filename, 'rb') as f:
        data = json.loads(json.loads(f.read()))
    
    #draw face, boxes and text for each response
    for r in data['responses']:
        
	if 'faceAnnotations' in r:
	    utils_image.draw_face(im, r['faceAnnotations'])
            
	if 'labelAnnotations' in r:
	    strs = map(lambda a: a['description'], r['labelAnnotations'])
	    im = utils_image.draw_text(im, ", ".join(strs))
            
	for field in ['textAnnotations', 'logoAnnotations']:
	    if field in r:
		for a in r[field]:
		    utils_image.draw_box(im, a['boundingPoly']['vertices'])

    #save to output file
    utils_image.save_image(result_filename, im)
    
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])


