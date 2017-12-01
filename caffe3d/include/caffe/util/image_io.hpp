#ifndef IMAGE_IO_HPP_
#define IMAGE_IO_HPP_


#include <string>

#include "caffe/blob.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
bool save_blob_to_binary(Blob<Dtype>* blob,
    const string fn_blob, int num_index);

template <typename Dtype>
inline bool save_blob_to_binary(Blob<Dtype>* blob,
    const string fn_blob) {
    return save_blob_to_binary(blob, fn_blob, -1);
}


}  // namespace caffe


#endif /* IMAGE_IO_HPP_ */
