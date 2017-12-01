#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Set the Random number generations given seed
   */
  void SetRandFromSeed(const unsigned int rng_seed);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);

#ifdef USE_OPENCV
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   * @param is_video
   *    A flag to specify if the given mat_vector should be treated as
   *    consecutive video frames (shape=[1, channels, N, w, h]), rather
   *    than a batch of images (shape=[N, channels, w, h]).
   */
  void Transform(const vector<cv::Mat> & mat_vector,
                Blob<Dtype>* transformed_blob,
                const bool is_video = false);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   * @param is_video
   *    A flag to reuse same random seed to replicate croppings/mirrorings and
   *    data augmentations for images within a same video clip
   * @param frame
   *    If is_video is enabled, frame refers to the frame index within a video
   *    clip (usually 0~15).
   * @param rand_mirror
   *    If is_video is enabled, rand_mirror is whether image frames within a
   *    video clip will be flipped.
   * @param rand_h_off
   *    If is_video is enabled, this is crop position in y for image frames
   *    within a video clip will be flipped.
   * @param rand_w_off
   *    If is_video is enabled, this is crop position in x for image frames
   *    within a video clip will be flipped.
   */
  void Transform(const cv::Mat& cv_img,
                  Blob<Dtype>* transformed_blob,
                  const bool is_video = false,
                  const int frame = 0,
                  const bool rand_mirror = false,
                  const int rand_h_off = 0,
                  const int rand_w_off = 0);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a pair of cv::Mat.
   * Ignores mean subtraction for second cv::Mat
   * Useful for an (image, label) pair
   * Appends seg as last channel of image
   *
   * @param cv_img
   *    1-channel cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_seg_data_layer.cpp for an example.
   */
  void TransformImageSeg(const cv::Mat& cv_img, const cv::Mat & cv_seg,
      Blob<Dtype>* transformed_data);
  // ------------------------------------------------------------------
#endif  // USE_OPENCV

  /**
   * @brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param is_video
   *    A flag to specify if the given mat_vector should be treated as
   *    consecutive video frames (shape=[1, channels, N, w, h]), rather
   *    than a batch of images (shape=[N, channels, w, h]).
   */
#ifdef USE_OPENCV
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector,
                             const bool is_video = false);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

  void Transform(const Datum& datum, Dtype* transformed_data);
  // Tranformation parameters
  TransformationParameter param_;


  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
