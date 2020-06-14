<h2 align="left"> Deploying People Counter App at the Edge Using Intel OpenVINO Toolkit </h2>

---

![Terminals](/terminal.png)
![Detection](/detection.png)

---

This project utilizes the Intel® Distribution of the OpenVINO™ Toolkit to build a People Counter app.
Including performing inference on an input video, the app extracts and analyzes the output data, then sends that data to a server.

The app detects people in the frame, provides people count as well as average duration.

---

For this project, I used [faster_rcnn_inception_v2_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)  model from the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

---

### Explaining Custom Layers in OpenVINO™:

When a layer isn’t supported by the Model Optimizer, one potential solution is to use of custom layers. The handling is different for each framework. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

Read the documentation: https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html

> <h4> The Process: </h4>
1. Download the model from Tensorflow Model Zoo:

  `wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz`

2. Extract the file:

`tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz`

3. Get into the extracted folder:

`cd faster_rcnn_inception_v2_coco_2018_01_28`

4. Convert the TensorFlow model to Intermediate Representation:

`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json`

---

### Compare Model Performance:
I compared the faster R-CNN Inception model's performance in terms of Inference Time and Model Size before and after conversion with the inference engine.


Comparison   | Latency (ms) | Size (MB) |
:-------------------------:|:-------------------------:|:-------------------------:|
Faster R-CNN Inception (TensorFlow) |1255 | 531|
Faster R-CNN Inception (OpenVINO) |919 | 293|

---

### Model Use Cases:
The model can can be used in situations where counting people is important. For example, in shops, only the required number of people can can be allowed to enter at a time. The model can automate and simplify this task. It can also be used in schools or colleges for attendance purposes.

---

### Effects on End User Needs:
Lighting, model accuracy, and camera focal length/image size can have big effects on end user needs. Poor lighting can interfere with model inference. Having properly lit environment as per the model requirement is of utmost importance.
Disturbed focul length can cause model difficulty infering the image which can icrease further difficulty.
According to the end user, some factors need to be maintained. Improving size, maintaining quality light can improve performance and model accuracy but it can increase cost of the hardware too.

---

### How to Set the Environment:
> You need four separate terminals open. New terminal for each task.
1. **Start the Mosca server:**

Run:

`cd webservice/server/node-server`

`node ./server.js`

You will see the following message, if successful:


`Mosca server started.`

2. **Start the GUI:**

Run:

`cd webservice/ui`

`npm run dev`

You should see the following message in the terminal.

`webpack: Compiled successfully`

3. **FFmpeg Server:**

Run:

`sudo ffserver -f ./ffmpeg/server.conf`

4. **Run the code:**

Initialize the OpenVINO Source Environment:

`source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5`

Execute the following command:

`python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`
