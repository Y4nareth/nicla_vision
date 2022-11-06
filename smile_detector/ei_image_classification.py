# Edge Impulse - OpenMV Image Classification Example
#
# Name: age_detector.py
#
# Author: Andrés Calandín Vega
# Institution: Universitat Politécncia de València
#
# ML model that identifies if a person is smiling or not.

import sensor, image, time, os, tf, uos, gc

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240,240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.
sensor.set_vflip(True)                 # Flips the image vertically
sensor.set_hmirror(True)               # Mirrors the image horizontally

net = None
labels = None

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = tf.load("age_detector/trained.tflite", load_to_fb=uos.stat('age_detector/trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("age_detector/labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()

    # default settings just do one detection... change them to search the image...
    for obj in net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        img.draw_rectangle(obj.rect())
        # This combines the labels and confidence values into a list of tuples
        predictions_list = list(zip(labels, obj.output()))

        for i in range(len(predictions_list)):
            print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

        if (predictions_list[1][1] > 0.5):
            categoria = "smile"
        else:
            categoria = "no-smile"

        img.draw_string(10, 10, categoria, color = (51, 204, 51), scale = 2, mono_space = False,
                        char_rotation = 0, char_hmirror = False, char_vflip = False,
                        string_rotation = 0, string_hmirror = False, string_vflip = False)

    print(clock.fps(), "fps")
