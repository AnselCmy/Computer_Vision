//
// Created by Chen on 2017/12/19.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
using namespace std;
using namespace cv;

#ifndef COMPUTER_VISION_HISTOGRAMEQUALIZATION_H
#define COMPUTER_VISION_HISTOGRAMEQUALIZATION_H

void getHist(InputArray _src, int* hist)
{
    Mat src = _src.getMat();
    int pixelCnt = src.cols * src.rows;
    uchar* imgData = src.data;
    for(int i=0; i<255; i++)
    {
        hist[i] = 0;
    }
    for(int i=0; i<pixelCnt; i++)
    {
        hist[imgData[i]]++;
    }
}

void histEqual(InputArray _src, OutputArray _dst)
{
    // src img
    Mat src = _src.getMat();
    // pixel count
    int pixelCnt = src.cols * src.rows;
    // dst img
    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();
    // histogram
    auto* hist = new int[256];
    getHist(src, hist);
    // conversion function
    uchar trans[256];
    // temp sum
    int sum = 0;
    for(int i = 0; i <= 255; ++i)
    {
        sum += hist[i];
        trans[i] = (uchar)(256 * sum / pixelCnt);
    }
    uchar* srcData = src.data;
    uchar* dsrData = dst.data;
    for(int j = 0; j < pixelCnt; ++j)
    {
        dsrData[j] = trans[srcData[j]];
    }
}

void histEqualTest()
{
    Mat img = imread("../img/film.jpg", CV_8UC1);
    Mat rst;
    histEqual(img, rst);
//    equalizeHist(img, rst);
    namedWindow("before histogram equalization", WINDOW_AUTOSIZE);
    imshow("before histogram equalization", img);
    namedWindow("after histogram equalization", WINDOW_AUTOSIZE);
    imshow("after histogram equalization", rst);
    waitKey();
}

#endif //COMPUTER_VISION_HISTOGRAMEQUALIZATION_H
