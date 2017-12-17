//
// Created by Chen on 2017/12/9.
//
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
using namespace std;
using namespace cv;

#ifndef CV_CALGLCM_H
#define CV_CALGLCM_H

void calGLCM(InputArray _src, OutputArray _GLCM, int GLCM_class = 256, int angle = 0, int offset = 1)
{
    // src
    Mat src = _src.getMat();
    Size srcSize = src.size();
    // matrix temp to store the pixel in GLCM_class-gray
    Mat temp(srcSize, CV_8UC1);
    // GLCM
    _GLCM.create(Size(GLCM_class, GLCM_class), CV_8UC1);
    Mat GLCM = _GLCM.getMat();
    GLCM = Scalar_<uchar>(0);
    // find max pixel in src
    uchar max = 0;
    for(int h = 0; h < srcSize.height; ++h)
    {
        for(int w = 0; w < srcSize.width; ++w)
        {
            if(src.at<uchar>(h, w) > max)
                max = src.at<uchar>(h, w);
        }
    }
    // zip into GLCM_class
    for(int h = 0; h < srcSize.height; ++h)
    {
        for(int w = 0; w < srcSize.width; ++w)
        {
            temp.at<uchar>(h, w) = (uchar)(src.at<uchar>(h, w) * GLCM_class / max);
        }
    }

    // calculate the matrix
    int row = 0, col = 0;
    if(angle == 0)
    {
        for(int h = 0; h < srcSize.height; ++h)
        {
            for(int w = 0; w < srcSize.width - offset; ++w)
            {
                row = temp.at<uchar>(h, w);
                col = temp.at<uchar>(h, w + offset);
                GLCM.at<uchar>(row, col)++;
                GLCM.at<uchar>(col, row)++;
            }
        }
    }
    else if(angle == 90)
    {
        for(int h = 0; h < srcSize.height - offset; ++h)
        {
            for(int w = 0; w < srcSize.width; ++w)
            {
                row = temp.at<uchar>(h, w);
                col = temp.at<uchar>(h + offset, w);
                GLCM.at<uchar>(row, col)++;
                GLCM.at<uchar>(col, row)++;
            }
        }
    }
    else if(angle == 45)
    {
        for(int h = 0; h < srcSize.height - offset; ++h)
        {
            for(int w = 0; w < srcSize.width - offset; ++w)
            {
                row = temp.at<uchar>(h, w);
                col = temp.at<uchar>(h + offset, w + offset);
                GLCM.at<uchar>(row, col)++;
                GLCM.at<uchar>(col, row)++;
            }
        }
    }
    else if(angle == 135)
    {
        for(int h = 0; h < srcSize.height-offset; ++h)
        {
            for(int w = 1; w < srcSize.width; ++w)
            {
                row = temp.at<uchar>(h, w);
                col = temp.at<uchar>(h + offset, w - offset);
                GLCM.at<uchar>(row, col)++;
                GLCM.at<uchar>(col, row)++;
            }
        }
    }
}

void getFeature(InputArray _GLCM, double& entropy, double& homogeneity, double& contrast)
{
    entropy = 0;
    homogeneity = 0;
    contrast = 0;
    Mat GLCM = _GLCM.getMat();
    Size size = GLCM.size();
    uchar currVal = 0;
    for(int h = 0; h < size.height; ++h)
    {
        for(int w = 0; w < size.width; ++w)
        {
            currVal = GLCM.at<uchar>(h, w);
            if(GLCM.at<uchar>(h, w) > 0)
                entropy -= currVal * log(currVal);
            contrast += currVal * pow(h-w, 2);
            homogeneity += currVal * (1/(1+pow(h-w, 2)));
        }
    }
}

void GLCM(const String path)
{
    Mat img = imread(path, CV_8UC1);
    Mat GLCM;
    double entropy = 0, homogeneity = 0, contrast = 0;
    int angels[] = {0, 90, 45, 135};
    cout << "-------------------------------------------------------------------" << endl;
    cout << path << endl;
    cout << "-------------------------------------------------------------------" << endl;
    cout << setiosflags(ios::left) << setw(14) << "Angle"
                                   << setw(14) << "Entropy"
                                   << setw(14) << "Homogeneity"
                                   << setw(14) << "Contrast" << endl;
    cout << "-------------------------------------------------------------------" << endl;
    cout.setf(ios::fixed);
    for(int a = 0; a < sizeof(angels)/ sizeof(angels[0]); ++a)
    {
        calGLCM(img, GLCM, 7, angels[a], 7);
        getFeature(GLCM, entropy, homogeneity, contrast);
        cout << setprecision(2) << setw(14) << angels[a]
                                << setw(14) << entropy
                                << setw(14) << homogeneity
                                << setw(14) << contrast << endl;
    }
//    cout << GLCM << endl;
    cout << "-------------------------------------------------------------------" << endl << endl;
}

void calGLCMTest()
{
    GLCM("../img/GLCM/GLCM_6.png");
    GLCM("../img/GLCM/GLCM_11.png");
    GLCM("../img/GLCM/GLCM_12.png");
    GLCM("../img/GLCM/GLCM_7.png");
    GLCM("../img/GLCM/GLCM_9.png");
}

#endif //CV_CALGLCM_H
