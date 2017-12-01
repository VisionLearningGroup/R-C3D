// ------------------------------------------------------------------
// R-C3D
// Copyright (c) 2017 Boston University
// Licensed under The MIT License [see LICENSE for details]
// Written by Huijuan Xu
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/roi_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_l(), 0)
      << "pooled_l must be > 0";
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_length_ = roi_pool_param.pooled_l();
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
  temporal_scale_ = roi_pool_param.temporal_scale();
  spatial_scale_ = roi_pool_param.spatial_scale();
  LOG(INFO) << "Temporal scale: " << temporal_scale_;
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->shape()[1];
  length_ = bottom[0]->shape()[2];
  height_ = bottom[0]->shape()[3];
  width_ = bottom[0]->shape()[4];
  vector<int> top_shape(5);
  top_shape[0] = bottom[1]->shape()[0];
  top_shape[1] = channels_;
  top_shape[2] = pooled_length_;
  top_shape[3] = pooled_height_;
  top_shape[4] = pooled_width_;
  top[0]->Reshape(top_shape);
  max_idx_.Reshape(top_shape);
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->shape()[0];
  int batch_size = bottom[0]->shape()[0];
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_l = round(bottom_rois[1] * temporal_scale_);
    int roi_start_w = 0;
    int roi_start_h = 0;
    int roi_end_l = round(bottom_rois[2] * temporal_scale_);
    int roi_end_w = 6;
    int roi_end_h = 6;
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_length = max(roi_end_l - roi_start_l + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const Dtype bin_size_l = static_cast<Dtype>(roi_length)
                             / static_cast<Dtype>(pooled_length_);
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int pl = 0; pl < pooled_length_; ++pl) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            // Compute pooling region for this output unit:
            //  start (included) = floor(ph * roi_height / pooled_height_)
            //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
            int lstart = static_cast<int>(floor(static_cast<Dtype>(pl)
                                      * bin_size_l));
            int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                      * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                                * bin_size_w));
            int lend = static_cast<int>(ceil(static_cast<Dtype>(pl + 1)
                                             * bin_size_l));
            int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                             * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                             * bin_size_w));
            lstart = min(max(lstart + roi_start_l, 0), length_);
            lend = min(max(lend + roi_start_l, 0), length_);
            hstart = min(max(hstart + roi_start_h, 0), height_);
            hend = min(max(hend + roi_start_h, 0), height_);
            wstart = min(max(wstart + roi_start_w, 0), width_);
            wend = min(max(wend + roi_start_w, 0), width_);

            bool is_empty = (hend <= hstart) || (wend <= wstart) ||
                            (lend <= lstart);

            const int pool_index = (pl * pooled_length_ + ph) * pooled_width_ + pw;
            if (is_empty) {
              top_data[pool_index] = 0;
              argmax_data[pool_index] = -1;
            }

            for (int l = lstart; l < lend; ++l) {
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = (l * length_ + h) * width_ + w;
                  if (batch_data[index] > top_data[pool_index]) {
                    top_data[pool_index] = batch_data[index];
                    argmax_data[pool_index] = index;
                  }
                }
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIPoolingLayer);
REGISTER_LAYER_CLASS(ROIPooling);

}  // namespace caffe

