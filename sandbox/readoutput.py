import requests
import sys
import json
import config

def main(input_filename, output_filename):
    #Get the setting description
    data = open(input_filename, 'rb').read()

    # Post the request
    response = requests.post(
        url='https://vision.googleapis.com/v1/images:annotate?key=%s'%config.secrets['API_KEY'],
        data = data,
        headers={'Content-Type': 'application/json'}
    )

    # Writre the detections to file
    with open(output_filename, 'w') as output_filename:
        response_list = response.text
        json.dump(response_list, output_filename)
    
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
