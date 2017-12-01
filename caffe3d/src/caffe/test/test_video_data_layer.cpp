#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class VideoDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  VideoDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
      outfile <<
        CMAKE_SOURCE_DIR "caffe/test/test_data/UCF-101_Rowing_g16_c03.avi " <<
        "0 " << i;
    }
    outfile.close();
    // Create test input file for images of distinct sizes.
    MakeTempFilename(&filename_reshape_);
    std::ofstream reshapefile(filename_reshape_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_reshape_;
    reshapefile <<
        CMAKE_SOURCE_DIR "caffe/test/test_data/UCF-101_Rowing_g16_c03.avi " <<
        "0 0";
    reshapefile <<
        CMAKE_SOURCE_DIR "caffe/test/test_data/UCF-101_Rowing_g16_c03.avi " <<
        "0 1";
    reshapefile.close();
  }

  virtual ~VideoDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  string filename_reshape_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(VideoDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(VideoDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();
  const int num = 5;
  const int channels = 3;
  const int length = 16;
  video_data_param->set_batch_size(num);
  video_data_param->set_new_length(length);
  video_data_param->set_source(this->filename_.c_str());
  video_data_param->set_shuffle(false);
  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), num);
  EXPECT_EQ(this->blob_top_data_->shape(1), channels);
  EXPECT_EQ(this->blob_top_data_->shape(2), length);
  EXPECT_EQ(this->blob_top_data_->shape(3), 240);
  EXPECT_EQ(this->blob_top_data_->shape(4), 320);
  EXPECT_EQ(this->blob_top_label_->shape(0), num);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(VideoDataLayerTest, TestCrop) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();
  video_data_param->set_batch_size(5);
  video_data_param->set_source(this->filename_.c_str());
  video_data_param->set_new_length(16);
  video_data_param->set_new_height(132);
  video_data_param->set_new_width(123);
  video_data_param->set_shuffle(false);
  TransformationParameter* transform_param =
      param.mutable_transform_param();
  transform_param->set_crop_size(77);
  transform_param->set_mirror(true);

  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 5);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 77);
  EXPECT_EQ(this->blob_top_data_->shape(4), 77);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(VideoDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();
  video_data_param->set_batch_size(5);
  video_data_param->set_source(this->filename_.c_str());
  video_data_param->set_new_length(16);
  video_data_param->set_new_height(132);
  video_data_param->set_new_width(123);
  video_data_param->set_shuffle(false);
  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 5);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 132);
  EXPECT_EQ(this->blob_top_data_->shape(4), 123);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(VideoDataLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();
  video_data_param->set_batch_size(1);
  video_data_param->set_new_length(16);
  video_data_param->set_source(this->filename_reshape_.c_str());
  video_data_param->set_shuffle(false);
  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  //
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 1);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 240);
  EXPECT_EQ(this->blob_top_data_->shape(4), 320);
  //
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 1);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 240);
  EXPECT_EQ(this->blob_top_data_->shape(4), 320);
}

TYPED_TEST(VideoDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  VideoDataParameter* video_data_param = param.mutable_video_data_param();
  video_data_param->set_batch_size(5);
  video_data_param->set_new_length(16);
  video_data_param->set_source(this->filename_.c_str());
  video_data_param->set_shuffle(true);
  VideoDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 5);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 16);
  EXPECT_EQ(this->blob_top_data_->shape(3), 240);
  EXPECT_EQ(this->blob_top_data_->shape(4), 320);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }
}

}  // namespace caffe
#endif  // USE_OPENCV
