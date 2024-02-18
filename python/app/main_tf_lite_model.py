#!/usr/bin/python

import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import efficientdet_classes
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ImageFile:
    name : str
    image : cv2.Mat

IMG_SIZE = [640, 640]
IMAGE_EXTENSIONS = [".jpg",".gif",".png",".tga"]

def read_params(path):
    # Opening JSON file
    f = open(path)
    # returns JSON object as a dictionary
    data = json.load(f)
    # Iterating through the json list
    model = data["model_path"]
    images = data["images_path"]
    saved_images = data["image_save_path"]
    score_threshold = data["score_threshold"]

    # Closing file
    f.close()

    return model, images, saved_images, score_threshold

def load_images(dirname):
    test_images = []
    for file in os.listdir(dirname):
        ext = os.path.splitext(file)[1]
        if ext.lower() not in IMAGE_EXTENSIONS:
                continue
        print("File: ", file)
        img = cv2.imread(f"{dirname}/{file}")

        img = cv2.resize(img, IMG_SIZE)
        image_file = ImageFile( Path(file).stem, img)

        test_images.append(image_file)

    return test_images

def draw_rectangle_over_objects(img, boxes, scores, num_detections, model_classes, score_threshold=0.65):
    out_image = img #image
    thickness = 2
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    for i in range(num_detections):
        if scores[i] >= score_threshold:
            box = boxes[i]
            y_min, x_min, y_max, x_max = box * IMG_SIZE[0]
            start_point = (int(x_min), int(y_min))
            end_point = (int(x_max), int(y_max))

            out_image = cv2.rectangle(out_image, start_point, end_point, color, thickness)
            out_image = cv2.putText(out_image, str(efficientdet_classes.class_list[int(model_classes[i])]), start_point, font, fontScale, color, int(thickness/2), cv2.LINE_AA)

    return out_image

def get_output_tensor(interpreter, index):
  """Return the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def draw_image(image):
    plt.imshow(image)
    plt.show()

def save_image(image, filename):
    cv2.imwrite(filename, image)

def delete_images_from_dir(dirname):
    for file in os.listdir(dirname):
        ext = os.path.splitext(file)[1]
        if ext.lower() not in IMAGE_EXTENSIONS:
                continue
        os.remove(dirname + "/" + file)


def main():
    params_path = sys.argv[1]
    if not os.path.exists(params_path):
        print("File doesn't exist!")
        return

    MODEL_DIRNAME, IMAGES_DIRNAME, SAVED_IMAGES_DIRNAME, SCORE_THRESHOLD,  = read_params(params_path)

    # delete eventual images from SAVED_IMAGES_DIRNAME
    delete_images_from_dir(SAVED_IMAGES_DIRNAME)

    # load images
    test_images = load_images(IMAGES_DIRNAME)

    # Load the TFLite model and allocate tensors.
    # interpreter = tf.lite.Interpreter(model_path="../python/efficientdet_lite4/2.tflite")
    interpreter = tf.lite.Interpreter(model_path=MODEL_DIRNAME)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print("input_details : ", input_details)
    # print("output_details : ", output_details)

    for i in range(len(test_images)):
        # Test the model on random input data.
        sample_image_t = tf.constant(test_images[i].image, dtype=tf.uint8)
        sample_image_t = tf.expand_dims(sample_image_t, axis=0)
        image_np = sample_image_t.numpy()

        # input_shape = input_details[0]['shape']
        # print("input_shape: ", input_shape)
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data = image_np

        interpreter.set_tensor(input_details[0]['index'], input_data)

        # model inference time
        start = time.time()
        interpreter.invoke()
        end = time.time()
        print("Inference time[ms] : ", end - start)

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # tensor_details = interpreter.get_tensor_details()
        # Get all outputs from the model
        classes = get_output_tensor(interpreter, 1)
        # print("classes", classes)
        boxes = get_output_tensor(interpreter, 0)
        # print("boxes", boxes)
        count = int(get_output_tensor(interpreter, 3))
        # print("count", count)
        scores = get_output_tensor(interpreter, 2)
        # print("scores", scores)
        # print("output data {output_data}", output_data)
        # print("output_data[0] ", output_data[0])
        # print("output_data[1]", output_data[0][1])
        # print("output_data[2]", output_data[0][2])
        # print("output_data[3]", output_data[0][3])
        # print(output_data)

        #print(test_images[index].shape)
        image = draw_rectangle_over_objects(test_images[i].image, boxes, scores, count, classes, SCORE_THRESHOLD)

        # print(objects)
        # draw_image(image)

        filename = SAVED_IMAGES_DIRNAME + test_images[i].name + "_processed.jpg"
        save_image(image, filename)


if __name__ == "__main__":
    main()