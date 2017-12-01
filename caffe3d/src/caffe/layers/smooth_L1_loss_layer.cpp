// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// modified by huijuan
// ------------------------------------------------------------------

#include "caffe/layers/smooth_L1_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SmoothL1LossParameter loss_param = this->layer_param_.smooth_l1_loss_param();
  sigma2_ = loss_param.sigma() * loss_param.sigma();
  has_weights_ = (bottom.size() >= 3);
  if (has_weights_) {
    CHECK_EQ(bottom.size(), 4) << "If weights are used, must specify both "
      "inside and outside weights";
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  for (int i=1; i<bottom[0]->num_axes(); i++)
    CHECK_EQ(bottom[0]->shape()[i], bottom[1]->shape()[i]);
  if (has_weights_) {
    for (int i=1; i<bottom[0]->num_axes(); i++) {
      CHECK_EQ(bottom[0]->shape()[i], bottom[2]->shape()[i]);
      CHECK_EQ(bottom[0]->shape()[i], bottom[3]->shape()[i]);
    }
  }
  diff_.Reshape(bottom[0]->shape());
  errors_.Reshape(bottom[0]->shape());
  // vector of ones used to sum
  ones_.Reshape(bottom[0]->shape()); 
  caffe_set(bottom[0]->count(), Dtype(1), ones_.mutable_cpu_data());
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe

