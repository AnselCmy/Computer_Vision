#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "adaptiveThreshold.h"
#include "boxFilter.h"
#include "GLCM.h"
#include "histEqual.h"
#include "alphaChannel.h"
using namespace std;
using namespace cv;

int main()
{
//    adaptiveThresholdTest();
//    blackFrameTest();
//    histEqualTest();
    calGLCMTest();
//    testGetIntImage();
//    split();


//    String dir = "../img/erode/";
//    String file = "test2";
//    String format = ".png";
//    Mat img = imread(dir+file+format, CV_8UC1);
//    Mat dst_thres, dst_erode, dst;
//    // OTSU
//    threshold(img, dst_thres, 0, 255, CV_THRESH_OTSU);
//    // 腐蚀
//    Mat element = getStructuringElement(MORPH_RECT, Size(11, 11));
//    erode(dst_thres, dst_erode, element);
//    // 使用腐蚀后的二值图像对原图像进行覆盖
//    dst = Mat(img.size(), img.type(), Scalar(0));
//    for(int r=0; r<img.rows; r++)
//    {
//        uchar* img_line = img.ptr(r);
//        uchar* erode_line = dst_erode.ptr(r);
//        uchar* dst_line = dst.ptr(r);
//        for(int c=0; c<img.cols; c++)
//        {
//            // 只取白色的为感兴趣的部分
//            if(erode_line[c] == 255)
//            {
//                dst_line[c] = img_line[c];
//            }
//        }
//    }
//
//    imwrite(dir+file+"_OTSU"+format, dst_thres);
//    imwrite(dir+file+"_erode"+format, dst_erode);
//    imwrite(dir+file+"_dst"+format, dst);

//    waitKey();
    return 0;
}