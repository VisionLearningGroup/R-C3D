import sys
sys.path.insert(0, '/home/gpuadmin/Documents/segmentation/tdcnn/caffe3d/python')
import caffe
import numpy as np

blob = caffe.proto.caffe_pb2.BlobProto()
arr = np.load('ucf101_mean.npy')
blob = caffe.io.array_to_blobproto(arr)
data = open('ucf101_train_mean.binaryproto', 'wb')
data.write(blob.SerializeToString())
data.close()


data = open('ucf101_train_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr1 = np.array(caffe.io.blobproto_to_array(blob))

