//
// Created by Chen on 2018/1/21.
//
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


#ifndef COMPUTER_VISION_ALPHACHANNEL_H
#define COMPUTER_VISION_ALPHACHANNEL_H

void readPng()
{
    Mat img = imread("../img/git.png", CV_LOAD_IMAGE_UNCHANGED);
    imshow("alpha", img);
//    cout << img.channels();
    cout << img.rows << " " << img.cols << endl;
    int row = 15;
    int col = 15;
    Vec4b* line = img.ptr<Vec4b>(row);
    cout << int(line[col][0]) << endl;
    cout << int(line[col][1]) << endl;
    cout << int(line[col][2]) << endl;
    cout << int(line[col][3]) << endl;
//    waitKey();
}

void makePng(InputArray _src, OutputArray _dst)
{
    // src
    Mat src = _src.getMat();
    Size size = src.size();
    // dst
    _dst.create(size, CV_8UC2);
    Mat dst = _dst.getMat();
    // thres
    Mat thres;
//    adaptiveThreshold(src, thres, )

}

#endif //COMPUTER_VISION_ALPHACHANNEL_H
