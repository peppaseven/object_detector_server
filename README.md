# object_classify_server
This server use tensorflow as main tool, try to tell my iPeppaAIcar what something is seen,then let iPeppaAIcar speak it.

# object_detector_server
This server use tensorflow object detection model as original codes, just for verify pre-trained models in common I3 PC server. you can download other pre-trained models&modify microdefine&test it.
Deploy tensorflow object detection model as web service running on gunicorn/flask instead of grpc.

# Copyright
This is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; version 2 of the License.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
Except for google tensorflow models,some codes from https://github.com/hetaoaoao/tensorflow_web_deploy, many thanks！

# how to make this work
Assume that you have installed tensorflow, flask, gunicorn. There may be other python module dependencies, pip install them.
1. run classify.sh(detector.sh has very similar steps as classify)
2. Open your browser, navigate to "http://localhost:30000", the page should be simple enough to understand.
3. Use as webservice. Modify the interface for your need.   
  > curl --request POST --data-binary "@sample.jpg" http://localhost:30000

