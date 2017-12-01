#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/image_io.hpp"

using std::fstream;
using std::ios;
using std::max;
using std::string;

namespace caffe {

template <>
bool save_blob_to_binary<float>(Blob<float>* blob,
    const string fn_blob, int num_index) {
    FILE *f;
    float *buff;
    int n, c, l, w, h;
    f = fopen(fn_blob.c_str(), "wb");
    if (f == NULL)
        return false;

    if (num_index < 0) {
        n = blob->shape(0);
        buff = blob->mutable_cpu_data();
    } else {
        n = 1;
        buff = blob->mutable_cpu_data() + blob->offset(num_index);
    }
    c = blob->shape(1);
    l = blob->shape(2);
    h = blob->shape(3);
    w = blob->shape(4);

    fwrite(&n, sizeof(int), 1, f);
    fwrite(&c, sizeof(int), 1, f);
    fwrite(&l, sizeof(int), 1, f);
    fwrite(&h, sizeof(int), 1, f);
    fwrite(&w, sizeof(int), 1, f);
    fwrite(buff, sizeof(float), n * c * l * h * w, f);
    fclose(f);
    return true;
}

template <>
bool save_blob_to_binary<double>(Blob<double>* blob,
    const string fn_blob, int num_index) {
    FILE *f;
    double *buff;
    int n, c, l, w, h;
    f = fopen(fn_blob.c_str(), "wb");
    if (f == NULL)
        return false;

    if (num_index < 0) {
        n = blob->shape(0);
        buff = blob->mutable_cpu_data();
    } else {
        n = 1;
        buff = blob->mutable_cpu_data() + blob->offset(num_index);
    }
    c = blob->shape(1);
    l = blob->shape(2);
    h = blob->shape(3);
    w = blob->shape(4);

    fwrite(&n, sizeof(int), 1, f);
    fwrite(&c, sizeof(int), 1, f);
    fwrite(&l, sizeof(int), 1, f);
    fwrite(&h, sizeof(int), 1, f);
    fwrite(&w, sizeof(int), 1, f);
    fwrite(buff, sizeof(double), n * c * l * h * w, f);
    fclose(f);
    return true;
}

}  // namespace caffe
