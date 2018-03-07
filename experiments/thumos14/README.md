
### R-C3D code on THUMOS'14 dataset 

### Please change the path in the corresponding file, and follow the instrucitons for the activityNet demo to run the code.



### Training:
    
1. Download C3D classification pretrain model to ./pretrain/ .

   The C3D model weight pretrained on UCF101 dataset (released by [1]) is provided in: [caffemodel](https://drive.google.com/file/d/1OlmcuaJbLjDYKQAPJi8b3TqkuGGWcDb0/view)


[1] Tran, Du, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar
Paluri. "Learning spatiotemporal features with 3d convolutional networks."
In Proceedings of the IEEE international conference on computer vision,
pp. 4489-4497. 2015. 



### Testing:

1. Download one sample R-C3D model to ./snapshot/ .

   One R-C3D model on THUMOS'14 dataset is provided in: [caffemodel](https://drive.google.com/file/d/1Mc0pWlbRsOKkUsV5F0ZCvu_F3Aq2O_Lk/view)

   The provided R-C3D model has the mAP@0.5 29.9% on the test set.


