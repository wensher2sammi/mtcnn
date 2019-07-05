#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import optparse
import os.path
from PIL import Image
from mtcnn.mtcnn import MTCNN

def MTCNN_benchmark(detector):
    numOfImage = 0
    total_detection = 0

    for root, dirs, files in os.walk('/home/local/KLASS/wensher.ong/Downloads/datasets/lfw'):
        for file in files:
            if file.endswith('.jpg'):
                numOfImage = numOfImage + 1
                full_imagepath = os.path.join(root, file)
                face_detected = MTCNN_detect(detector, full_imagepath)
                if (face_detected > 1):
                    #print("More than 1 face detected in : {}" .format(file))
                    face_detected = 1
                elif (face_detected == 0):
                    print("Huh? No face detected in : {}" .format(file))

            total_detection = total_detection + face_detected
            #print("Current total detection : {}" .format(total_detection))

    print("Base number of faces : {}" .format(numOfImage))
    print("Actual detected faces : {}" .format(total_detection))            

def crop_face(imagefile, coords, savelocation):
    image_obj = Image.open(imagefile)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(savelocation)
    cropped_image.show()


def MTCNN_detect(detector, imagefile):

    image = cv2.imread(imagefile)    

    result = detector.detect_faces(image)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    indexing = 0
    for i_result in result:
        bounding_box = i_result['box']
        keypoints = i_result['keypoints']
        indexing = indexing + 1
        cv2.rectangle(image,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0,155,255),
                2)

        #crop_face(imagefile, (bounding_box[0], bounding_box[1],bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), 'test_{}.jpg' .format(indexing))

        cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
        

    # Prints all the keypoints and bounding box onto the image.
    #cv2.imwrite("output.jpg", image)

    #print(result)
    return indexing


if __name__ == '__main__':
    opt_parser = optparse.OptionParser()

    opt_parser.add_option('-f', '--file', dest='filename', 
                            help="Image filename", default="avengers.jpg")
    opt_parser.add_option('-b', '--benchmark', action="store_true", dest='benchmark', 
                            help="Select mode to run on this engine : Benchmark/...",
                            default=False)

    (opt_args, args) = opt_parsed = opt_parser.parse_args()

    if len(args) > 0:
        opt_parser.print_help()
        exit()

    if (opt_args.filename is not None):
        imagefile = opt_args.filename
        # Check if file exist
        if (not os.path.exists(imagefile)):
            print("File does not exist!")

    detector = MTCNN()

    if (opt_args.benchmark is True):
        # Perform benchmark to see how much can it detect faces in LFW database
        MTCNN_benchmark(detector)
    else:
        MTCNN_detect(detector, imagefile)
