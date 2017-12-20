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
void adaptiveThresholdByHand(InputArray _src, OutputArray _dst, double maxValue=255,
                             int blockSize=11, int delta=5)
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

void adaptiveThresholdByIntImg(InputArray _src, OutputArray _dst, double maxValue=255,
                               int blockSize=9, double subPercent=0.06)
{
    // src
    Mat src = _src.getMat();
    Size size = src.size();
    // dst
    _dst.create(size, src.type());
    Mat dst = _dst.getMat();
    // integral image
    Mat intImg(size, CV_32SC1);

    // calculate integral image
    int sum;
    for(int w = 0; w < size.width; ++w)
    {
        sum = 0;
        for(int h = 0; h < size.height; ++h)
        {
            sum += src.at<uchar>(h, w);
            if(w == 0)
                intImg.at<int>(h, w) = sum;
            else
            {
                intImg.at<int>(h, w) = intImg.at<int>(h, w-1) + sum;
            }
        }
    }

    int x1, y1, x2, y2;
    int count;
    for(int w = 0; w < size.width; ++w)
    {
        for(int h = 0; h < size.height; ++h)
        {
            x1 = w - blockSize / 2 > 0 ? w - blockSize / 2 : 0;
            x2 = w + blockSize / 2 < size.width ? w + blockSize / 2 : size.width-1;
            y1 = h - blockSize / 2 > 0 ? h - blockSize / 2 : 0;
            y2 = h + blockSize / 2 < size.height ? h + blockSize / 2 : size.height-1;
            count = (x2 - x1) * (y2 - y1);
            sum = intImg.at<int>(y2, x2) + intImg.at<int>(y1, x1)
                  - intImg.at<int>(y1, x2) - intImg.at<int>(y2, x1);
            // do threshold
            auto threshold = (uchar)(sum*(1.0-subPercent)/count);
            dst.at<uchar>(h, w) = (uchar)(src.at<uchar>(h, w) > threshold ? maxValue : 0);
        }
    }
}

/************************ Track Bar Start ************************/
int blockSize = 11;
int subPercent = 15;

void adaptiveThresholdByIntImg_adjust(int, void*)
{
    Mat img, rst;
    img = imread("../img/threshold/threshold_1.bmp", CV_8UC1);
    adaptiveThresholdByIntImg(img, rst, 255, blockSize, subPercent*0.01);
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
    Mat img, rst;
    String path = "../img/threshold/threshold_1";
    String format = ".bmp";
    img = imread(path+format, CV_8UC1);

    threshold(img, rst, 75, 255, THRESH_BINARY);
    imwrite(path+"_0"+format, rst);

    adaptiveThreshold(img, rst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, 5);
    imwrite(path+"_1"+format, rst);

    adaptiveThresholdByHand(img, rst, 255, 3, 5);
    imwrite(path+"_2"+format, rst);

    adaptiveThresholdByIntImg(img, rst);
    imwrite(path+"_3"+format, rst);
}