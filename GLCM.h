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
                double& ASM, double& correlation)
{
    entropy = 0;
    homogeneity = 0;
    contrast = 0;
    Mat GLCM = _GLCM.getMat();
    Size size = GLCM.size();
    float currVal = 0;
    // correlation
    double mean_i = 0, mean_j = 0, var_i = 0, var_j = 0;
    for(int i = 0; i < size.height; ++i)
    {
        for(int j = 0; j < size.width; ++j)
        {
            currVal = GLCM.at<float>(i, j);
            mean_i += currVal * i;
            mean_j += currVal * j;
            var_i += currVal * pow((i - mean_i), 2);
            var_j += currVal * pow((j - mean_j), 2);
        }
    }
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
            // correlation
            correlation += currVal * (h-mean_i) * (w-mean_j);
        }
    }
    correlation /= sqrt(var_i*var_j);
}

void GLCM(const String path)
{
    Mat img = imread(path, CV_8UC1);
    Mat GLCM;
    double entropy = 0, homogeneity = 0, contrast = 0, ASM = 0, correlation = 0;
    int angels[] = {0, 90, 45, 135};
    cout << "-------------------------------------------------------------------" << endl;
    cout << path << endl;
    cout << "-------------------------------------------------------------------" << endl;
    cout << setiosflags(ios::left) << setw(14) << "Angle"
                                   << setw(14) << "Entropy"
                                   << setw(14) << "Homogeneity"
                                   << setw(14) << "Contrast"
                                   << setw(14) << "ASM"
                                   << setw(14) << "Correlation" << endl;
    cout << "-------------------------------------------------------------------" << endl;
    cout.setf(ios::fixed);
    for(int a = 0; a < sizeof(angels)/ sizeof(angels[0]); ++a)
    {
        calGLCM(img, GLCM, 16, angels[a], 1);
        getFeature(GLCM, entropy, homogeneity, contrast, ASM, correlation);
        cout << setprecision(2) << setw(14) << angels[a]
                                << setw(14) << entropy
                                << setw(14) << homogeneity
                                << setw(14) << contrast
                                << setw(14) << ASM
                                << setw(14) << correlation << endl;
    }
//    cout << GLCM << endl;
    cout << "-------------------------------------------------------------------" << endl << endl;
}

void GLCM(String imgs[], int len)
{
    // out file
    ofstream out;
    out.open("../GLCM_result.csv", ios::out);
    out << "Picture,Angle,Entropy,Homogeneity,Contrast,ASM,Correlation\n";
    // GLCM Mat
    Mat GLCM;
    Mat img;
    double entropy = 0, homogeneity = 0, contrast = 0, ASM = 0, correlation = 0;
    int angles[] = {0, 90, 45, 135};
    for(int i = 0; i < len; ++i)
    {
        img = imread(imgs[i], CV_8UC1);
        for(int a = 0; a < sizeof(angles)/ sizeof(angles[0]); ++a)
        {
            calGLCM(img, GLCM, 8, angles[a]);
            entropy = 0, homogeneity = 0, contrast = 0, ASM = 0, correlation = 0;
            getFeature(GLCM, entropy, homogeneity, contrast, ASM, correlation);
            out << imgs[i] << "," << angles[a] << ","
                << entropy << "," << homogeneity << ","
                << contrast << "," << ASM <<","
                << correlation << "\n";
            cout.setf(ios::fixed);
            cout << setprecision(4) << setw(14) << imgs[i]
                                    << setw(14) << angles[a]
                                    << setw(14) << entropy
                                    << setw(14) << homogeneity
                                    << setw(14) << contrast
                                    << setw(14) << ASM
                                    << setw(14) << correlation << endl;
        }
//        cout << GLCM << endl;
    }
    out.close();
}

void calGLCMTest()
{
    String imgs[] = {
                     "../img/GLCM_SVM/E1.bmp", "../img/GLCM_SVM/E2.bmp",
                     "../img/GLCM_SVM/E3.bmp", "../img/GLCM_SVM/E4.bmp",
                     "../img/GLCM_SVM/E5.bmp", "../img/GLCM_SVM/E6.bmp",
                     "../img/GLCM_SVM/E7.bmp", "../img/GLCM_SVM/E8.bmp",
                     "../img/GLCM_SVM/E9.bmp", "../img/GLCM_SVM/E10.bmp",
                     "../img/GLCM_SVM/E11.bmp", "../img/GLCM_SVM/E12.bmp",
                     "../img/GLCM_SVM/E13.bmp","../img/GLCM_SVM/E14.bmp",
                     "../img/GLCM_SVM/E15.bmp","../img/GLCM_SVM/E16.bmp",
                     "../img/GLCM_SVM/E17.bmp","../img/GLCM_SVM/E18.bmp",
                     "../img/GLCM_SVM/E19.bmp","../img/GLCM_SVM/E20.bmp",
                     "../img/GLCM_SVM/E21.bmp", "../img/GLCM_SVM/E22.bmp",
                     "../img/GLCM_SVM/E23.bmp","../img/GLCM_SVM/E24.bmp",
                     "../img/GLCM_SVM/E25.bmp","../img/GLCM_SVM/E26.bmp",
                     "../img/GLCM_SVM/R1.bmp", "../img/GLCM_SVM/R2.bmp",
                     "../img/GLCM_SVM/R3.bmp", "../img/GLCM_SVM/R4.bmp",
                     "../img/GLCM_SVM/R5.bmp", "../img/GLCM_SVM/R6.bmp",
                     "../img/GLCM_SVM/R7.bmp", "../img/GLCM_SVM/R8.bmp",
                     "../img/GLCM_SVM/R9.bmp", "../img/GLCM_SVM/R10.bmp",
                     "../img/GLCM_SVM/R11.bmp", "../img/GLCM_SVM/R12.bmp",
                     "../img/GLCM_SVM/R13.bmp", "../img/GLCM_SVM/R14.bmp",
                     "../img/GLCM_SVM/R15.bmp","../img/GLCM_SVM/R16.bmp",
                     "../img/GLCM_SVM/R17.bmp","../img/GLCM_SVM/R18.bmp",
                     "../img/GLCM_SVM/R19.bmp","../img/GLCM_SVM/R20.bmp",
                     "../img/GLCM_SVM/R21.bmp", "../img/GLCM_SVM/R22.bmp",
                     "../img/GLCM_SVM/R23.bmp", "../img/GLCM_SVM/R24.bmp",
                     "../img/GLCM_SVM/R25.bmp","../img/GLCM_SVM/R26.bmp",
                    };
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
