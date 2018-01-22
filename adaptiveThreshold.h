//
// Created by Chen on 2017/12/6.
//
#include <iostream>
#include <fstream>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "boxFilter.h"
using namespace std;
using namespace cv;

/*
 * This is the adaptiveThreshold from openCV source code
 */
void adaptiveThreshold_( InputArray _src, OutputArray _dst, double maxValue,
                         int method, int type, int blockSize, double delta )
{
    // 获取原图片的矩阵
    Mat src = _src.getMat();
    CV_Assert( src.type() == CV_8UC1 );
    CV_Assert( blockSize % 2 == 1 && blockSize > 1 );
    // 获取原图片的size
    Size size = src.size();

    // 构造一个和原图一样的图片
    _dst.create( size, src.type() );
    Mat dst = _dst.getMat();

    if( maxValue < 0 )
    {
        dst = Scalar(0);
        return;
    }

    // 用于比较的平均值
    Mat mean;

    if( src.data != dst.data )
        mean = dst;

    if (method == ADAPTIVE_THRESH_MEAN_C)
    {
        // 用boxFilter计算均值
        boxFilter(src, mean, src.type(), Size(blockSize, blockSize),
                  Point(-1, -1), true, BORDER_REPLICATE);
    }
    else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        Mat srcfloat,meanfloat;
        src.convertTo(srcfloat,CV_32F);
        meanfloat=srcfloat;
        GaussianBlur(srcfloat, meanfloat, Size(blockSize, blockSize), 0, 0, BORDER_REPLICATE);
        meanfloat.convertTo(mean, src.type());
    }
    else
        CV_Error( CV_StsBadFlag, "Unknown/unsupported adaptive threshold method" );

    int i, j;
    // 把maxValue限制在0～255
    uchar imaxval = saturate_cast<uchar>(maxValue);
    // 根据二值化的类型计算idelta的值，
    int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);

    // 构造要一个每个像素的映射表，方便之后直接查表运算
    uchar tab[768];
    if( type == CV_THRESH_BINARY )
        for( i = 0; i < 768; i++ )
            tab[i] = (uchar)(i - 255 > -idelta ? imaxval : 0);
    else if( type == CV_THRESH_BINARY_INV )
        for( i = 0; i < 768; i++ )
            tab[i] = (uchar)(i - 255 <= -idelta ? imaxval : 0);
    else
        CV_Error( CV_StsBadFlag, "Unknown/unsupported threshold type" );

    // 查看是否连续，如果连续，那么可以都加到一行，加快运算的速度
    if( src.isContinuous() && mean.isContinuous() && dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( i = 0; i < size.height; i++ )
    {
        // 分别获取到src, mean, dst中第i行的元素
        const uchar* sdata = src.ptr(i);
        const uchar* mdata = mean.ptr(i);
        uchar* ddata = dst.ptr(i);

        for( j = 0; j < size.width; j++ )
            ddata[j] = tab[sdata[j] - mdata[j] + 255];
    }
}


/*
 * The is the adaptiveThreshold by hand
 */
void adaptiveThresholdByHand(InputArray _src, OutputArray _dst, OutputArray _thresholdMat, double maxValue=255,
                             int blockSize=11, int delta=5)
{
    assert(blockSize % 2 == 1);

    // 获取输入的mat和size
    Mat src = _src.getMat();
    Size size = src.size();
    // 构造一个和src相同的dst
    _dst.create(size, src.type());
    Mat dst = _dst.getMat();
    Mat temp;
    // thresholdMat
    _thresholdMat.create(size, src.type());
    Mat thresholdMat = _thresholdMat.getMat();
    // cover
    Mat cover;
    threshold(src, cover, 0, 255, CV_THRESH_OTSU);
    // 中间的mean作为计算平均值的图像结果
    Mat mean;
    boxFilter(src, mean, src.type(), Size(blockSize, blockSize),
              Point(-1, -1), true, BORDER_REPLICATE);
    mean.copyTo(thresholdMat);
    // GaussianBlur(src, mean, Size(blockSize, blockSize), 0, 0, BORDER_DEFAULT);
    // 循环遍历进行二值化
    for(int h = 0; h < size.height; h++)
    {
        const uchar* srcLine  = src.ptr(h);
        const uchar* meanLine = mean.ptr(h);
        const uchar* coverLine = cover.ptr(h);
        uchar* dstLine = dst.ptr(h);
        for(int w = 0; w < size.width; w++)
        {
            if(coverLine[w] == 0)
            {
                dstLine[w] = 255;
            }
            else
            {
                dstLine[w] = (uchar)(srcLine[w] > meanLine[w] - delta ? maxValue : 0);
            }
        }
    }
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(dst, temp, element);
    erode(temp, dst, element);
}

void getIntegralImage(InputArray _src, OutputArray _intImg, int power = 1)
{
    // src
    Mat src = _src.getMat();
    Size size = src.size();
    // integral image
    _intImg.create(size, CV_64FC1);
    Mat intImg = _intImg.getMat();

    double sum;
    for(int w = 0; w < size.width; ++w)
    {
        sum = 0;
        for(int h = 0; h < size.height; ++h)
        {
            if(power == 1)
                sum += src.at<uchar>(h, w);
            else
                sum += pow(src.at<uchar>(h, w), power);
            if(w == 0)
                intImg.at<double>(h, w) = sum;
            else
                intImg.at<double>(h, w) = intImg.at<double>(h, w-1) + sum;
        }
    }
}

void testGetIntImage()
{
    uchar m[3][4] = {{3,1,2,5}, {1,4,2,1}, {3,1,6,0}};
    Mat img = Mat(3, 4, CV_8UC1, m);
    Mat intImg;
    getIntegralImage(img, intImg);
    cout << "Origin Image" << endl;
    cout << img << endl;
    cout << "Integral Image" << endl;
    cout << intImg << endl;
}

void bradleyThreshold(InputArray _src, OutputArray _dst, OutputArray _thresholdMat, double maxValue = 255,
                                int blockSize = 9, double subPercent = 0.09, bool padding = false)
{
    // src
    Mat src = _src.getMat();
    Size size = src.size();
    // dst
    _dst.create(size, src.type());
    Mat dst = _dst.getMat();
    // integral image
    Mat intImg;
    // _threshold
    _thresholdMat.create(size, src.type());
    Mat thresholdMat = _thresholdMat.getMat();

    int x1, y1, x2, y2;
    int count;
    double sum;
    int width;
    int height;
    int pad = (blockSize - 1) / 2;
    if(padding)
    {
        assert(blockSize % 2 == 1);
        Mat srcPad;
        copyMakeBorder(src, srcPad, pad, pad, pad, pad, BORDER_REPLICATE, 0);
        width = srcPad.size().width;
        height = srcPad.size().height;
        getIntegralImage(srcPad, intImg);
        for(int w = pad; w < size.width+pad; ++w)
        {
            for(int h = pad; h < size.height+pad; ++h)
            {
                x1 = w - pad;
                x2 = w + pad;
                y1 = h - pad;
                y2 = h + pad;

                count = (x2 - x1 + 1) * (y2 - y1 + 1);
                if(x1 == 0 && y1 == 0)
                {
                    sum = intImg.at<double>(y2, x2);
                } else if(x1 == 0 && y1 > 0)
                {
                    sum = intImg.at<double>(y2, x2) - intImg.at<double>(y1 - 1, x2);
                } else if(y1 == 0 && x1 > 0)
                {
                    sum = intImg.at<double>(y2, x2) - intImg.at<double>(y2, x1 - 1);
                } else
                {
                    sum = intImg.at<double>(y2, x2) + intImg.at<double>(y1 - 1, x1 - 1)
                          - intImg.at<double>(y1 - 1, x2) - intImg.at<double>(y2, x1 - 1);
                }
                // do threshold
                auto threshold = (uchar) (sum * (1.0 - subPercent) / count);
//                auto threshold = (uchar)(sum/count - 5);
                thresholdMat.at<uchar>(h-pad, w-pad) = (uchar)threshold;
                dst.at<uchar>(h-pad, w-pad) = (uchar) (src.at<uchar>(h-pad, w-pad) > threshold ? maxValue : 0);
            }
        }
    }
    else
    {
        width = size.width;
        height = size.height;
        getIntegralImage(src, intImg);
        for(int w = 0; w < size.width; ++w)
        {
            for(int h = 0; h < size.height; ++h)
            {
                x1 = max(w - pad, 0);
                x2 = min(w + pad, width - 1);
                y1 = max(h - pad, 0);
                y2 = min(h + pad, height - 1);

                count = (x2 - x1 + 1) * (y2 - y1 + 1);
                if(x1 == 0 && y1 == 0)
                {
                    sum = intImg.at<double>(y2, x2);
                } else if(x1 == 0)
                {
                    sum = intImg.at<double>(y2, x2) - intImg.at<double>(y1 - 1, x2);
                } else if(y1 == 0)
                {
                    sum = intImg.at<double>(y2, x2) - intImg.at<double>(y2, x1 - 1);
                } else
                {
                    sum = intImg.at<double>(y2, x2) + intImg.at<double>(y1 - 1, x1 - 1)
                          - intImg.at<double>(y1 - 1, x2) - intImg.at<double>(y2, x1 - 1);
                }
                // do threshold
                auto threshold = (uchar) (sum * (1.0 - subPercent) / count);
                thresholdMat.at<uchar>(h, w) = (uchar)threshold;
//                auto threshold = (uchar)(sum/count - 5);
                dst.at<uchar>(h, w) = (uchar) (src.at<uchar>(h, w) > threshold ? maxValue : 0);
            }
        }
    }
}

void sauvolaThreshold(InputArray _src, OutputArray _dst, OutputArray _thresholdMat,
                      double maxValue=255, int blockSize=11)
{
    // src image
    Mat src = _src.getMat();
    Size size = src.size();
    // dst image
    _dst.create(size, src.type());
    Mat dst = _dst.getMat();
    // integral image
    Mat intImg, intImgSq;
    getIntegralImage(_src, intImg, 1);
    getIntegralImage(_src, intImgSq, 2);
    // _thresholdMat to store the value of each block
    _thresholdMat.create(size, src.type());
    Mat thresholdMat = _thresholdMat.getMat();

    int x1, y1, x2, y2;
    int count;
    double sum, sumSq;
    double k = 0.1;
    double mean, stdVariance, threshold;
    for(int w = 0; w < size.width; ++w)
    {
        for(int h = 0; h < size.height; ++h)
        {
            x1 = max(w - blockSize / 2, 0);
            x2 = min(w + blockSize / 2, size.width - 1);
            y1 = max(h - blockSize / 2, 0);
            y2 = min(h + blockSize / 2, size.height - 1);

            count = (x2 - x1 + 1) * (y2 - y1 + 1);
            if(x1 == 0 && y1 == 0)
            {
                sum = intImg.at<double>(y2, x2);
                sumSq = intImgSq.at<double>(y2, x2);
            }
            else if(x1 == 0 && y1 > 0)
            {
                sum = intImg.at<double>(y2, x2) - intImg.at<double>(y1-1, x2);
                sumSq = intImgSq.at<double>(y2, x2) - intImgSq.at<double>(y1-1, x2);
            }
            else if(y1 == 0 && x1 > 0)
            {
                sum = intImg.at<double>(y2, x2) - intImg.at<double>(y2, x1-1);
                sumSq = intImgSq.at<double>(y2, x2) - intImgSq.at<double>(y2, x1-1);
            }
            else
            {
                sum = intImg.at<double>(y2, x2) + intImg.at<double>(y1-1, x1-1)
                      - intImg.at<double>(y1-1, x2) - intImg.at<double>(y2, x1-1);
                sumSq = intImgSq.at<double>(y2, x2) + intImgSq.at<double>(y1-1, x1-1)
                      - intImgSq.at<double>(y1-1, x2) - intImgSq.at<double>(y2, x1-1);
            }

            mean = sum/count;
//            stdVariance = sqrt((sumSq - pow(sum, 2)/count)/(count-1));
//            stdVariance = sqrt((sumSq - sum*sum/count)/count);
            stdVariance = sqrt((sumSq/count - mean*mean));
            threshold = mean*(1+k*((stdVariance/128)-1));
            thresholdMat.at<uchar>(h, w) = (uchar)threshold;
            dst.at<uchar>(h, w) = (uchar)(src.at<uchar>(h, w) > threshold ? maxValue : 0);
        }
    }
}

/************************ Track Bar Start ************************/
int blockSize = 11;
int subPercent = 15;

void adaptiveThresholdByIntImg_adjust(int, void*)
{
    Mat img, rst, thresholdMap;
    img = imread("../img/threshold/threshold_6.bmp", CV_8UC1);
    resize(img, rst, Size(img.cols/4, img.rows/4), 0, 0, INTER_LINEAR);
    img = rst;
    if(blockSize % 2 != 1)
    {
        blockSize++;
    }
//    adaptiveThreshold(img, rst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, subPercent);
    bradleyThreshold(img, rst, thresholdMap, 255, blockSize, subPercent*0.01, true);
//    sauvolaThreshold(img, rst, 255, blockSize);
    cout << "Block Size: " << blockSize << "\n" << "Sub Percent: " << subPercent*0.01 << endl;
    imshow("threshold", rst);
}

void adjust()
{
    String winName = "threshold";
    namedWindow(winName, WINDOW_AUTOSIZE);
    createTrackbar("Sub Percent(*0.01)", winName, &subPercent, 100, adaptiveThresholdByIntImg_adjust, NULL);
    createTrackbar("Block Size", winName, &blockSize, 20, adaptiveThresholdByIntImg_adjust, NULL);
    adaptiveThresholdByIntImg_adjust(0, NULL);
    waitKey();
}
/************************ Track Bar End ************************/

void adaptiveThresholdTest()
{
//    adjust();
    Mat img, rst, rst1, thresholdMat;
    clock_t start, end;
    ofstream file;
    String path = "../img/threshold/threshold_1";
    String format = ".bmp";
    img = imread(path+format, CV_8UC1);
//    threshold(img, rst, 75, 255, THRESH_BINARY);
//    imwrite(path+"_0"+format, rst);
    file.open((path+".pixel.txt").c_str(), ios_base::out);
    file << img;

    start = clock();
    adaptiveThreshold(img, rst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 5);
    end = clock();
    printf("the running time of cv::adaptiveThreshold():   %f\n", double(end-start)/CLOCKS_PER_SEC);
    imwrite(path+"_1"+format, rst);
    file.open((path+"_1.pixel.txt").c_str(), ios_base::out);
    file << rst;
    file.close();

    start = clock();
    adaptiveThresholdByHand(img, rst, thresholdMat, 255, 11, 5);
    end = clock();
    printf("the running time of adaptiveThresholdByHand(): %f\n", double(end-start)/CLOCKS_PER_SEC);
    imwrite(path+"_2"+format, rst);
    file.open((path+"_2.pixel.txt").c_str(), ios_base::out);
    file << rst;
    file.close();
    file.open((path+"_2.thres.txt").c_str(), ios_base::out);
    file << thresholdMat;
    file.close();

    start = clock();
    bradleyThreshold(img, rst, thresholdMat, 255, 11, 0.09, false);
    end = clock();
    printf("the running time of bradleyThreshold():        %f\n", double(end-start)/CLOCKS_PER_SEC);
    imwrite(path+"_3"+format, rst);
    file.open((path+"_3.pixel.txt").c_str(), ios_base::out);
    file << rst;
    file.close();
    file.open((path+"_3.thres.txt").c_str(), ios_base::out);
    file << thresholdMat;
    file.close();

    start = clock();
    sauvolaThreshold(img, rst, thresholdMat, 255, 11);
    end = clock();
    printf("the running time of sauvolaThreshold():        %f\n", double(end-start)/CLOCKS_PER_SEC);
    imwrite(path+"_4"+format, rst);
    file.open((path+"_4.pixel.txt").c_str(), ios_base::out);
    file << rst;
    file.close();
    file.open((path+"_4.thres.txt").c_str(), ios_base::out);
    file << thresholdMat;
    file.close();
}

void split()
{
    Mat img = imread("../img/threshold/threshold_1.bmp", CV_8UC1);
    Mat dst_thres;
    Mat dst = Mat(img.size(), CV_8UC4);
    threshold(img, dst_thres, 0, 255, CV_THRESH_OTSU);
    for(int r=0; r<img.rows; r++)
    {
        uchar* thres_line = dst_thres.ptr<uchar>(r);
        uchar* img_line = img.ptr<uchar>(r);
        Vec4b* dst_line = dst.ptr<Vec4b>(r);
        for(int c=0; c<img.cols; c++)
        {
            // 白色作为前景保留
            if(int(thres_line[c]) == 255)
            {
                dst_line[c][0] = 255;
                dst_line[c][1] = 255;
                dst_line[c][2] = 255;
                dst_line[c][3] = 255;
            }
            else if(int(thres_line[c]) == 0)
            {
                dst_line[c][0] = 0;
                dst_line[c][1] = 0;
                dst_line[c][2] = 0;
                dst_line[c][3] = 0;
            }
        }
//        cout << int(thres_line[100]);
    }
//    waitKey();
    imwrite("../img/threshold/threshold_1.thres.png", dst_thres);
    imwrite("../img/threshold/threshold_1.split.png", dst);
}
