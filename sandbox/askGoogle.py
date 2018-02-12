import generateinput
import generatejson
import readoutput
import showresults
import sys
import os

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

for image_filename in absoluteFilePaths('tst_images'):

    input_filename = image_filename.replace('images/', 'inputs/')
    input_filename = "%s.txt"%os.path.splitext(input_filename)[0]
    request_filename = image_filename.replace('images/', 'requests/')
    request_filename = "%s.json"%os.path.splitext(request_filename)[0]
    response_filename = image_filename.replace('images/', 'responses/')
    response_filename = "%s.json"%os.path.splitext(response_filename)[0]
    result_filename = image_filename.replace('images/', 'results/')
    result_filename = "%s.jpg"%os.path.splitext(result_filename)[0]

    print image_filename, ' to ', result_filename
    
#    generateinput.main(image_filename, input_filename, detection_type=5, max_results=10) #Logo detection
    generateinput.main(image_filename, input_filename, detection_type=1, max_results=10) #Label detection
    with open(input_filename, 'rb') as input_file:
        generatejson.main(input_file, request_filename)
    readoutput.main(request_filename, response_filename)
    # not exactly usefull for the final usage
    showresults.main(image_filename, response_filename, result_filename)
