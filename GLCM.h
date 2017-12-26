//
// Created by Chen on 2017/12/9.
//
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
using namespace std;
using namespace cv;

#ifndef CV_CALGLCM_H
#define CV_CALGLCM_H

void calGLCM(InputArray _src, OutputArray _GLCM, int GLCM_class = -1,
             int angle = 0, int offset = 1, bool norm = true)
{
    // src
    Mat src = _src.getMat();
    Size srcSize = src.size();
    // matrix temp to store the pixel in GLCM_class-gray
    Mat temp(srcSize, CV_8UC1);
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
    // GLCM
    if(GLCM_class == -1)
    {
        GLCM_class = max+1;
    }
    _GLCM.create(Size(GLCM_class, GLCM_class), CV_32FC1);
    Mat GLCM = _GLCM.getMat();
    GLCM = Scalar_<float>(0);
    // zip into GLCM_class
    for(int h = 0; h < srcSize.height; ++h)
    {
        for(int w = 0; w < srcSize.width; ++w)
        {
            temp.at<uchar>(h, w) = (uchar)(src.at<uchar>(h, w) * GLCM_class / (max+1));
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
                GLCM.at<float>(row, col)++;
                GLCM.at<float>(col, row)++;
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
                GLCM.at<float>(row, col)++;
                GLCM.at<float>(col, row)++;
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
                GLCM.at<float>(row, col)++;
                GLCM.at<float>(col, row)++;
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
                GLCM.at<float>(row, col)++;
                GLCM.at<float>(col, row)++;
            }
        }
    }

    // normalization
    if(norm)
    {
        float sum = 0;
        for(int i = 0; i < GLCM_class; ++i)
        {
            for(int j = 0; j < GLCM_class; ++j)
            {
                sum += GLCM.at<float>(i, j);
            }
        }
        for(int i = 0; i < GLCM_class; ++i)
        {
            for(int j = 0; j < GLCM_class; ++j)
            {
                GLCM.at<float>(i, j) /= (sum * 1.0);
            }
        }
    }
}

void getFeature(InputArray _GLCM, double& entropy, double& homogeneity, double& contrast,
                double& ASM)
{
    entropy = 0;
    homogeneity = 0;
    contrast = 0;
    Mat GLCM = _GLCM.getMat();
    Size size = GLCM.size();
    float currVal = 0;
    for(int h = 0; h < size.height; ++h)
    {
        for(int w = 0; w < size.width; ++w)
        {
            currVal = GLCM.at<float>(h, w);
            // Entropy
            if(currVal > 0)
                entropy += (pow(h-w, 2) * currVal) / log(currVal);
            // Contrast
            contrast += currVal * pow(h-w, 2);
            // Homogeneity
            homogeneity += currVal / (1+pow(h-w, 2));
            // Angular Second Moment
            ASM += pow(currVal, 2);
        }
    }
}

void GLCM(const String path)
{
    Mat img = imread(path, CV_8UC1);
    Mat GLCM;
    double entropy = 0, homogeneity = 0, contrast = 0, ASM = 0;
    int angels[] = {0, 90, 45, 135};
    cout << "-------------------------------------------------------------------" << endl;
    cout << path << endl;
    cout << "-------------------------------------------------------------------" << endl;
    cout << setiosflags(ios::left) << setw(14) << "Angle"
                                   << setw(14) << "Entropy"
                                   << setw(14) << "Homogeneity"
                                   << setw(14) << "Contrast"
                                   << setw(14) << "ASM" << endl;
    cout << "-------------------------------------------------------------------" << endl;
    cout.setf(ios::fixed);
    for(int a = 0; a < sizeof(angels)/ sizeof(angels[0]); ++a)
    {
        calGLCM(img, GLCM, 16, angels[a], 1);
        getFeature(GLCM, entropy, homogeneity, contrast, ASM);
        cout << setprecision(2) << setw(14) << angels[a]
                                << setw(14) << entropy
                                << setw(14) << homogeneity
                                << setw(14) << contrast
                                << setw(14) << ASM << endl;
    }
//    cout << GLCM << endl;
    cout << "-------------------------------------------------------------------" << endl << endl;
}

void GLCM(String imgs[], int len)
{
    // out file
    ofstream out;
    out.open("../GLCM_result.csv", ios::out);
    out << "Picture,Angle,Entropy,Homogeneity,Contrast,ASM\n";
    // GLCM Mat
    Mat GLCM;
    Mat img;
    double entropy = 0, homogeneity = 0, contrast = 0, ASM = 0;
    int angles[] = {0, 90, 45, 135};
    for(int i = 0; i < len; ++i)
    {
        img = imread(imgs[i], CV_8UC1);
        for(int a = 0; a < sizeof(angles)/ sizeof(angles[0]); ++a)
        {
            calGLCM(img, GLCM, 8, angles[a]);
            getFeature(GLCM, entropy, homogeneity, contrast, ASM);
            out << imgs[i] << "," << angles[a] << ","
                << entropy << "," << homogeneity << ","
                << contrast << "," << ASM << "\n";
            cout.setf(ios::fixed);
            cout << setprecision(4) << setw(14) << imgs[i]
                                    << setw(14) << angles[a]
                                    << setw(14) << entropy
                                    << setw(14) << homogeneity
                                    << setw(14) << contrast
                                    << setw(14) << ASM << endl;
        }
//        cout << GLCM << endl;
    }
    out.close();
}

void calGLCMTest()
{
    String imgs[] = {"../img/GLCM/E1.png", "../img/GLCM/E2.png",
                     "../img/GLCM/E3.png", "../img/GLCM/E4.png",
                     "../img/GLCM/R1.png", "../img/GLCM/R2.png",
                     "../img/GLCM/R3.png", "../img/GLCM/R4.png",
                     "../img/GLCM/R5.png", "../img/GLCM/R6.png",
                     "../img/GLCM/R7.png", "../img/GLCM/R8.png",
                     "../img/GLCM/R9.png", "../img/GLCM/R10.png",
                     "../img/GLCM/R11.png", "../img/GLCM/R12.png"};
    GLCM(imgs, sizeof(imgs)/ sizeof(imgs[0]));
}

void checkGLCM()
{
    uchar m[3][3] = {{0,1,2}, {1,2,0}, {2,1,0}};
    Mat img = Mat(3, 3, CV_8UC1, m);
    cout << img << endl;
    Mat GLCM;
    cout << "------------------------------------------------------" << endl;
    calGLCM(img, GLCM, -1, 0, 1, false);
    cout << GLCM << endl;
    calGLCM(img, GLCM, -1, 0, 1, true);
    cout << GLCM << endl;
    cout << "------------------------------------------------------" << endl;
    calGLCM(img, GLCM, -1, 45, 1, false);
    cout << GLCM << endl;
    calGLCM(img, GLCM, -1, 45, 1, true);
    cout << GLCM << endl;
    cout << "------------------------------------------------------" << endl;
    calGLCM(img, GLCM, -1, 90, 1, false);
    cout << GLCM << endl;
    calGLCM(img, GLCM, -1, 90, 1, true);
    cout << GLCM << endl;
    cout << "------------------------------------------------------" << endl;
    calGLCM(img, GLCM, -1, 135, 1, false);
    cout << GLCM << endl;
    calGLCM(img, GLCM, -1, 135, 1, true);
    cout << GLCM << endl;
}

#endif //CV_CALGLCM_H
