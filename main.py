"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

from argparse import ArgumentParser
from inference import Network

import logging as log
import paho.mqtt.client as mqtt

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.4,
                        help="Probability threshold for detections filtering"
                             "(0.4 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    network_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Live feed check
    if args.input == 'CAM':
        input_stream = 0
        single_image_mode = False

    # Input image check
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Video check
    else:
        single_image_mode = False
        input_stream = args.input
        assert os.path.isfile(args.input), "File doesn't exist"

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)

    w = int(cap.get(3))
    h = int(cap.get(4))

    # Init variables
    
    counter=0
    total_duration=0
    req_id=0
    
    entry_time=0
    leave_time=0
    
    input_shape = network_shape
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        input_image = cv2.resize(frame, (input_shape[3], input_shape[2]))
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape(1, *input_image.shape)
  

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': input_image, 'image_info': input_image.shape[1:]}
#         duration_report = None
        infer_network.exec_net(req_id, net_input)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            net_output = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
         
            detected_people = 0
            probs = net_output[0, 0, :, 2]
            for i, j in enumerate(probs):
                if j > prob_threshold:
                    detected_people += 1
                    bbox = net_output[0, 0, i, 3:]
                    j1 = (int(bbox[0] * w), int(bbox[1] * h))
                    j2 = (int(bbox[2] * w), int(bbox[3] * h))
                    
                    frame = cv2.rectangle(frame, j1, j2, (125, 0, 0), 3)
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if detected_people > counter:
                timer_enter = time.time()
                previous_off = entry_time - leave_time
                counter = detected_people
                if previous_off > 10:
                    total_duration = total_duration  + (detected_people - counter)
                    client.publish("person", json.dumps({"total": total_duration}))
                                
            if detected_people < counter:
                leave_time = time.time()
                previous_on = leave_time - entry_time
                counter = detected_people
                if previous_on > 10:
                    client.publish("person/duration", json.dumps({"duration": int(previous_on)}))
            
            client.publish("person", json.dumps({"count": counter}))

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
            
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
