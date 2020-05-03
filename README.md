# Description  
From images, this model detects locations of faces and recognizes emotions that appear in the face. This model is based on Faster R-CNN for object detection.

# Run
* Training the model: python train.py (Images should be placed in "data/train/img")
* Inference: python test.py (Images should be placed in "data/test/img")  
  
# For light-weight model
* We used the mobilinet_v2 pretrained with imagenet dataset. 
* We have tried to convert the model precision from float32 into float16, which lead to reducing the size of model (329M --> 164MB). However, it leads to divergence of the model weights.

 