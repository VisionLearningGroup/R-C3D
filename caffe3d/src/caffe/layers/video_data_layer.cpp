#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>&
      bottom, const vector<Blob<Dtype>*>& top) {
  const int new_length = this->layer_param_.video_data_param().new_length();
  const int new_height = this->layer_param_.video_data_param().new_height();
  const int new_width  = this->layer_param_.video_data_param().new_width();
  const bool is_color  = this->layer_param_.video_data_param().is_color();
  string root_folder = this->layer_param_.video_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.video_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int frame_num, label, stride;
  while (infile >> filename >> frame_num >> label >> stride) {
    triplet video_and_label;
    video_and_label.first = filename;
    video_and_label.second = frame_num;
    video_and_label.third = label;
    video_and_label.fourth = stride;
    lines_.push_back(video_and_label);
  }

  if (this->layer_param_.video_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleVideos();
  }
  LOG(INFO) << "A total of " << lines_.size() << " video chunks.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.video_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.video_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a video clip, and use it to initialize the top blob.
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result = ReadVideoToCVMat(root_folder +
                                            lines_[lines_id_].first,
                                            lines_[lines_id_].second,
                                            new_length, new_height, new_width,
                                            is_color,
                                            &cv_imgs,
                                            lines_[lines_id_].fourth);
  CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
                              " at frame " << lines_[lines_id_].second << ".";
  CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
                                          lines_[lines_id_].first <<
                                          " at frame " <<
                                          lines_[lines_id_].second <<
                                          " correctly.";
  // Use data_transformer to infer the expected blob shape from a cv_image.
  const bool is_video = true;
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_imgs,
                                                                  is_video);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.video_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->shape(0) << ","
      << top[0]->shape(1) << "," << top[0]->shape(2) << ","
      << top[0]->shape(3) << "," << top[0]->shape(4);
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  VideoDataParameter video_data_param = this->layer_param_.video_data_param();
  const int batch_size = video_data_param.batch_size();
  const int new_length = video_data_param.new_length();
  const int new_height = video_data_param.new_height();
  const int new_width = video_data_param.new_width();
  const bool is_color = video_data_param.is_color();
  string root_folder = video_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result = ReadVideoToCVMat(root_folder +
                                             lines_[lines_id_].first,
                                             lines_[lines_id_].second,
                                             new_length, new_height, new_width,
                                             is_color,
                                             &cv_imgs,
                                             lines_[lines_id_].fourth);
  CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
                              " at frame " << lines_[lines_id_].second << ".";
  CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
                                          lines_[lines_id_].first <<
                                          " at frame " <<
                                          lines_[lines_id_].second <<
                                          " correctly.";
  // Use data_transformer to infer the expected blob shape from a cv_imgs.
  bool is_video = true;
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_imgs,
                                                                  is_video);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  vector<int> blob_offset(5);
  blob_offset[1] = 0;
  blob_offset[2] = 0;
  blob_offset[3] = 0;
  blob_offset[4] = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    std::vector<cv::Mat> cv_imgs;
    bool read_video_result = ReadVideoToCVMat(root_folder +
                                               lines_[lines_id_].first,
                                               lines_[lines_id_].second,
                                               new_length, new_height,
                                               new_width, is_color, &cv_imgs,
                                               lines_[lines_id_].fourth);
    CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
                                " at frame " << lines_[lines_id_].second << ".";
    CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
                                             lines_[lines_id_].first <<
                                            " at frame " <<
                                            lines_[lines_id_].second <<
                                            " correctly.";
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    blob_offset[0] = item_id;
    int offset = batch->data_.offset(blob_offset);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    const bool is_video = true;
    this->data_transformer_->Transform(cv_imgs, &(this->transformed_data_),
                                       is_video);
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].third;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.video_data_param().shuffle()) {
        ShuffleVideos();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

}  // namespace caffe
#endif  // USE_OPENCV
