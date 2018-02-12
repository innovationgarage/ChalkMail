import sys

def main(image_filename, input_filename, detection_type, max_results):
    with open(input_filename, 'w') as f:
        f.write('%s %d:%d'%(image_filename, int(detection_type), int(max_results)))
        
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

