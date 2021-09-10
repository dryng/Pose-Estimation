# Pose-Estimation

This project is broken up into 4 differnt folders:
  1. data <br>
     &nbsp;&nbsp;&nbsp;&nbsp; data.py: create the dataset from the MPII Human Pose Dataset matlab file
  2. models <br>
     &nbsp;&nbsp;&nbsp;&nbsp; model.py: contains various flavors of CNNs to experiment with <br>
     &nbsp;&nbsp;&nbsp;&nbsp; saved_models: folder of trained model weights <br>
  3. training <br>
     &nbsp;&nbsp;&nbsp;&nbsp; training.py: functions to train and evaluate the chosen model from model.py <br>
  4. utils <br>
     &nbsp;&nbsp;&nbsp;&nbsp; anotate.py: annotate images based on model output <br>
     &nbsp;&nbsp;&nbsp;&nbsp; utils.py: utlity functions, i.e calculate single epoch time <br>
     &nbsp;&nbsp;&nbsp;&nbsp; video.py: read and then annotate (based on model output) images from a video stream <br>
     &nbsp;&nbsp;&nbsp;&nbsp; visualizations.py: functions to visulaize training process <br>
