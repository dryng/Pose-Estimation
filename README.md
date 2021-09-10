# Pose-Estimation

This project is broken up into 4 differnt folders:
  1. data
     data.py: create the dataset from the MPII Human Pose Dataset matlab file
  2. models
     model.py: contains various flavors of CNNs to experiment with
     saved_models: folder of trained model weights
  3. training
     training.py: functions to train and evaluate the chosen model from model.py
  4. utils
     anotate.py: annotate images based on model output
     utils.py: utlity functions, i.e calculate single epoch time
     video.py: read and then annotate (based on model output) images from a video stream
     visualizations.py: functions to visulaize training process
