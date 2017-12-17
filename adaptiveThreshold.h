//
// Created by Chen on 2017/12/6.
//
#include <iostream>
#include <opencv2/opencv.hpp>
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
void adaptiveThresholdByHand (InputArray _src, OutputArray _dst, double maxValue, int blockSize, int delta)
{
    assert(blockSize % 2 == 1);

    // 获取输入的mat和size
    Mat src = _src.getMat();
    Size size = src.size();
    // 构造一个和src相同的dst
    _dst.create(size, src.type());
    Mat dst = _dst.getMat();
    // 中间的mean作为计算平均值的图像结果
    Mat mean;
    boxFilter(src, mean, src.type(), Size(blockSize, blockSize),
              Point(-1, -1), true, BORDER_REPLICATE);
    // GaussianBlur(src, mean, Size(blockSize, blockSize), 0, 0, BORDER_DEFAULT);
    // 循环遍历进行二值化
    for(int h = 0; h < size.height; h++)
    {
        const uchar* srcLine  = src.ptr(h);
        const uchar* meanLine = mean.ptr(h);
        uchar* dstLine = dst.ptr(h);
        for(int w = 0; w < size.width; w++)
        {
            dstLine[w] = (uchar)(srcLine[w] > meanLine[w] - delta ? maxValue : 0);
        }
    }
}

void adaptiveThresholdByIntImg(InputArray _src, OutputArray _dst)
{
    // src
    Mat src = _src.getMat();
    Size size = src.size();
    // dst
    _dst.create(size, src.type());
    Mat dst = _dst.getMat();
    // integral image
    Mat intImg(size, CV_32SC1);

    int s = 11;
    float t = 0.15;
    // calculate integral image
    int sum;
    for(int w = 0; w < size.width; ++w)
    {
        sum = 0;
        for(int h = 0; h < size.height; ++h)
        {
            sum += src.at<uchar>(w, h);
            if(w == 0)
                intImg.at<int>(w, h) = sum;
            else
                intImg.at<int>(w, h) = intImg.at<int>(w-1, h) + sum;
        }
    }

    int x1, y1, x2, y2;
    int count;
    for(int w = 0; w < size.width; ++w)
    {
        for(int h = 0; h < size.height; ++h)
        {
            x1 = w - s / 2 > 0 ? w - s / 2 : 0;
            x2 = w + s / 2 < size.width ? w + s / 2 : size.width;
            y1 = h - s / 2 > 0 ? h - s / 2 : 0;
            y2 = h + s / 2 < size.height ? h + s / 2 : size.height;
            count = (x2 - x1) * (y2 - y1);
            sum = intImg.at<int>(x2, x1) + intImg.at<int>(x1 - 1, y1 - 1)
                  - intImg.at<int>(x2, y1 - 1) - intImg.at<int>(x1 - 1, y2);
            // do threshold
            auto threshold = (uchar)(sum*(1.0-t)/count);
            dst.at<uchar>(w, h) = (uchar)(src.at<uchar>(w, h) > threshold ? 255 : 0);
        }
    }
}