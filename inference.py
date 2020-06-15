"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log

from openvino.inference_engine import IENetwork, IECore

DEVICE = 'CPU'
class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model_xml, CPU_EXTENSION, DEVICE):
        ### TODO: Load the model ###
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.plugin = IECore()
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        ### TODO: Check for supported layers ###
        layers_supported = self.plugin.query_network(self.network, DEVICE)
        layers = self.network.layers.keys()

        total_supported = True
        for i in layers:
            if i not in layers_supported:
                total_supported = False
        if not total_supported:
            print('Not all model layers supported, please specify an appropriate cpu_extension with --cpu_extension')
            self.plugin.add_extension(CPU_EXTENSION, DEVICE)
            
        self.exec_network = self.plugin.load_network(self.network, DEVICE)
        
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return 
    

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        input_shape = self.network.inputs[self.input_blob].shape
        return input_shape
    

    def exec_net(self, request_id, net_input):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        execute_net = self.infer_request_handle = self.exec_network.start_async(request_id, net_input)

        return execute_net


    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.infer_request_handle.wait()
    

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.infer_request_handle.outputs[self.output_blob]
