#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#if CV_MAJOR_VERSION == 3
#include <opencv2/videoio/videoio.hpp>
#endif

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class IOTest : public ::testing::Test {};

bool ReadImageToDatumReference(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  int num_channels = (is_color ? 3 : 1);
  datum->set_channels(num_channels);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  if (is_color) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
          static_cast<char>(cv_img.at<uchar>(h, w)));
        }
      }
  }
  return true;
}

TEST_F(IOTest, TestReadImageToDatum) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TEST_F(IOTest, TestReadImageToDatumReference) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum, datum_ref;
  ReadImageToDatum(filename, 0, 0, 0, true, &datum);
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum.data();

  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}


TEST_F(IOTest, TestReadImageToDatumReferenceResized) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum, datum_ref;
  ReadImageToDatum(filename, 0, 100, 200, true, &datum);
  ReadImageToDatumReference(filename, 0, 100, 200, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum.data();

  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestReadImageToDatumContent) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, &datum);
  cv::Mat cv_img = ReadImageToCVMat(filename);
  EXPECT_EQ(datum.channels(), cv_img.channels());
  EXPECT_EQ(datum.height(), cv_img.rows);
  EXPECT_EQ(datum.width(), cv_img.cols);

  const string& data = datum.data();
  int index = 0;
  for (int c = 0; c < datum.channels(); ++c) {
    for (int h = 0; h < datum.height(); ++h) {
      for (int w = 0; w < datum.width(); ++w) {
        EXPECT_TRUE(data[index++] ==
          static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
}

TEST_F(IOTest, TestReadImageToDatumContentGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, is_color, &datum);
  cv::Mat cv_img = ReadImageToCVMat(filename, is_color);
  EXPECT_EQ(datum.channels(), cv_img.channels());
  EXPECT_EQ(datum.height(), cv_img.rows);
  EXPECT_EQ(datum.width(), cv_img.cols);

  const string& data = datum.data();
  int index = 0;
  for (int h = 0; h < datum.height(); ++h) {
    for (int w = 0; w < datum.width(); ++w) {
      EXPECT_TRUE(data[index++] == static_cast<char>(cv_img.at<uchar>(h, w)));
    }
  }
}

TEST_F(IOTest, TestReadImageToDatumResized) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, 100, 200, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 100);
  EXPECT_EQ(datum.width(), 200);
}


TEST_F(IOTest, TestReadImageToDatumResizedSquare) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, 256, 256, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 256);
  EXPECT_EQ(datum.width(), 256);
}

TEST_F(IOTest, TestReadImageToDatumGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, is_color, &datum);
  EXPECT_EQ(datum.channels(), 1);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TEST_F(IOTest, TestReadImageToDatumResizedGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, 256, 256, is_color, &datum);
  EXPECT_EQ(datum.channels(), 1);
  EXPECT_EQ(datum.height(), 256);
  EXPECT_EQ(datum.width(), 256);
}

TEST_F(IOTest, TestReadImageToCVMat) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestReadImageToCVMatResized) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename, 100, 200);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 100);
  EXPECT_EQ(cv_img.cols, 200);
}

TEST_F(IOTest, TestReadImageToCVMatResizedSquare) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename, 256, 256);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 256);
  EXPECT_EQ(cv_img.cols, 256);
}

TEST_F(IOTest, TestReadImageToCVMatGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  const bool is_color = false;
  cv::Mat cv_img = ReadImageToCVMat(filename, is_color);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestReadImageToCVMatResizedGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  const bool is_color = false;
  cv::Mat cv_img = ReadImageToCVMat(filename, 256, 256, is_color);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 256);
  EXPECT_EQ(cv_img.cols, 256);
}

TEST_F(IOTest, TestCVMatToDatum) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  Datum datum;
  CVMatToDatum(cv_img, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TEST_F(IOTest, TestCVMatToDatumContent) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  Datum datum;
  CVMatToDatum(cv_img, &datum);
  Datum datum_ref;
  ReadImageToDatum(filename, 0, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestCVMatToDatumReference) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  Datum datum;
  CVMatToDatum(cv_img, &datum);
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestReadFileToDatum) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(datum.encoded());
  EXPECT_EQ(datum.label(), -1);
  EXPECT_EQ(datum.data().size(), 140391);
}

TEST_F(IOTest, TestDecodeDatum) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(DecodeDatum(&datum, true));
  EXPECT_FALSE(DecodeDatum(&datum, true));
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestDecodeDatumToCVMat) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  cv::Mat cv_img = DecodeDatumToCVMat(datum, true);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
  cv_img = DecodeDatumToCVMat(datum, false);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestDecodeDatumToCVMatContent) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadImageToDatum(filename, 0, std::string("jpg"), &datum));
  cv::Mat cv_img = DecodeDatumToCVMat(datum, true);
  cv::Mat cv_img_ref = ReadImageToCVMat(filename);
  EXPECT_EQ(cv_img_ref.channels(), cv_img.channels());
  EXPECT_EQ(cv_img_ref.rows, cv_img.rows);
  EXPECT_EQ(cv_img_ref.cols, cv_img.cols);

  for (int c = 0; c < datum.channels(); ++c) {
    for (int h = 0; h < datum.height(); ++h) {
      for (int w = 0; w < datum.width(); ++w) {
        EXPECT_TRUE(cv_img.at<cv::Vec3b>(h, w)[c]==
          cv_img_ref.at<cv::Vec3b>(h, w)[c]);
      }
    }
  }
}

TEST_F(IOTest, TestDecodeDatumNative) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(DecodeDatumNative(&datum));
  EXPECT_FALSE(DecodeDatumNative(&datum));
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, true, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestDecodeDatumToCVMatNative) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  cv::Mat cv_img = DecodeDatumToCVMatNative(datum);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestDecodeDatumNativeGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat_gray.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  EXPECT_TRUE(DecodeDatumNative(&datum));
  EXPECT_FALSE(DecodeDatumNative(&datum));
  Datum datum_ref;
  ReadImageToDatumReference(filename, 0, 0, 0, false, &datum_ref);
  EXPECT_EQ(datum.channels(), datum_ref.channels());
  EXPECT_EQ(datum.height(), datum_ref.height());
  EXPECT_EQ(datum.width(), datum_ref.width());
  EXPECT_EQ(datum.data().size(), datum_ref.data().size());

  const string& data = datum.data();
  const string& data_ref = datum_ref.data();
  for (int i = 0; i < datum.data().size(); ++i) {
    EXPECT_TRUE(data[i] == data_ref[i]);
  }
}

TEST_F(IOTest, TestDecodeDatumToCVMatNativeGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat_gray.jpg";
  Datum datum;
  EXPECT_TRUE(ReadFileToDatum(filename, &datum));
  cv::Mat cv_img = DecodeDatumToCVMatNative(datum);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestDecodeDatumToCVMatContentNative) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  EXPECT_TRUE(ReadImageToDatum(filename, 0, std::string("jpg"), &datum));
  cv::Mat cv_img = DecodeDatumToCVMatNative(datum);
  cv::Mat cv_img_ref = ReadImageToCVMat(filename);
  EXPECT_EQ(cv_img_ref.channels(), cv_img.channels());
  EXPECT_EQ(cv_img_ref.rows, cv_img.rows);
  EXPECT_EQ(cv_img_ref.cols, cv_img.cols);

  for (int c = 0; c < datum.channels(); ++c) {
    for (int h = 0; h < datum.height(); ++h) {
      for (int w = 0; w < datum.width(); ++w) {
        EXPECT_TRUE(cv_img.at<cv::Vec3b>(h, w)[c]==
          cv_img_ref.at<cv::Vec3b>(h, w)[c]);
      }
    }
  }
}

TEST_F(IOTest, TestReadVideoToCVMatBasic) {
  string path = CMAKE_SOURCE_DIR \
                "caffe/test/test_data/youtube_objects_dog_v0002_s006";
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result = ReadVideoToCVMat(path,
                                            1,     // start frame
                                            16,    // length (# frames)
                                            0,     // new height
                                            0,     // new width
                                            true,  // load as color
                                            &cv_imgs, 1);
  EXPECT_EQ(read_video_result, true);
  EXPECT_EQ(cv_imgs.size(), 16);
  EXPECT_EQ(cv_imgs[0].channels(), 3);
  EXPECT_EQ(cv_imgs[0].rows, 720);
  EXPECT_EQ(cv_imgs[0].cols, 1280);
}

TEST_F(IOTest, TestReadVideoToCVMatNotEnoughFrames) {
  string path = CMAKE_SOURCE_DIR \
                "caffe/test/test_data/youtube_objects_dog_v0002_s006";
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result = ReadVideoToCVMat(path,
                                            2,     // start frame
                                            16,    // length (# frames)
                                            0,     // new height
                                            0,     // new width
                                            true,  // load as color
                                            &cv_imgs, 1);
  EXPECT_EQ(read_video_result, false);   // because there are only 16 frames
}

TEST_F(IOTest, TestReadVideoToCVMatResize) {
  string path = CMAKE_SOURCE_DIR \
                "caffe/test/test_data/youtube_objects_dog_v0002_s006";
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result = ReadVideoToCVMat(path,
                                            1,     // start frame
                                            16,    // length (# frames)
                                            80,    // new height
                                            100,   // new width
                                            true,  // load as color
                                            &cv_imgs, 1);
  EXPECT_EQ(read_video_result, true);
  EXPECT_EQ(cv_imgs.size(), 16);
  EXPECT_EQ(cv_imgs[0].channels(), 3);
  EXPECT_EQ(cv_imgs[0].rows, 80);
  EXPECT_EQ(cv_imgs[0].cols, 100);
}

TEST_F(IOTest, TestReadVideoToCVMatFromAviBasic) {
  string path = CMAKE_SOURCE_DIR \
                "caffe/test/test_data/UCF-101_Rowing_g16_c03.avi";
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result = ReadVideoToCVMat(path,
                                            1,     // start frame
                                            19,    // length (# frames)
                                            0,     // new height
                                            0,     // new width
                                            true,  // load as color
                                            &cv_imgs, 1);
  EXPECT_EQ(read_video_result, true);
  EXPECT_EQ(cv_imgs.size(), 19);
  EXPECT_EQ(cv_imgs[0].channels(), 3);
  EXPECT_EQ(cv_imgs[0].rows, 240);
  EXPECT_EQ(cv_imgs[0].cols, 320);
}

TEST_F(IOTest, TestReadVideoToCVMatFromAviResize) {
  string path = CMAKE_SOURCE_DIR \
                "caffe/test/test_data/UCF-101_Rowing_g16_c03.avi";
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result = ReadVideoToCVMat(path,
                                            1,      // start frame
                                            19,     // length (# frames)
                                            123,    // new height
                                            300,    // new width
                                            true,   // load as color
                                            &cv_imgs, 1);
  EXPECT_EQ(read_video_result, true);
  EXPECT_EQ(cv_imgs.size(), 19);
  EXPECT_EQ(cv_imgs[0].channels(), 3);
  EXPECT_EQ(cv_imgs[0].rows, 123);
  EXPECT_EQ(cv_imgs[0].cols, 300);
}

TEST_F(IOTest, TestReadVideoToCVMatFromAviResizeAndGrayscale) {
  string path = CMAKE_SOURCE_DIR \
                "caffe/test/test_data/UCF-101_Rowing_g16_c03.avi";
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result = ReadVideoToCVMat(path,
                                            1,      // start frame
                                            16,     // length (# frames)
                                            80,     // new height
                                            100,    // new width
                                            false,  // load as color
                                            &cv_imgs, 1);
  EXPECT_EQ(read_video_result, true);
  EXPECT_EQ(cv_imgs.size(), 16);
  EXPECT_EQ(cv_imgs[0].channels(), 1);
  EXPECT_EQ(cv_imgs[0].rows, 80);
  EXPECT_EQ(cv_imgs[0].cols, 100);
}

}  // namespace caffe
#endif  // USE_OPENCV
