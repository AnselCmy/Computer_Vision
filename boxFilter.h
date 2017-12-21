//
// Created by Chen on 2017/12/6.
//
#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#ifndef CV_BOXFILTER_H
#define CV_BOXFILTER_H

void boxFilterByHand(InputArray _src, OutputArray _dst, Size ksize)
{
    // 作为boxFilter的kernel，边长必须是奇数并且大于1
    assert(ksize.height % 2 == 1 && ksize.height > 1);
    assert(ksize.height == ksize.width);

    // 获得_src的mat
    Mat src = _src.getMat();
    Size srcSize = src.size();
    // 对src的图形做拓展padding
    Mat src_pad;
    int padding = (ksize.width - 1) / 2;
    copyMakeBorder(src, src_pad, padding, padding, padding, padding, BORDER_REPLICATE, 0);
    // 初始化dst，作为结果的mat
    _dst.create(srcSize, src.type());
    Mat dst = _dst.getMat();
    // 开始用四层for循环进行卷积的操作
    // 首先用两层for循环遍历原图中的每一个点
    for (int h = padding; h < srcSize.height+padding; ++h)
    {
        for (int w = padding; w < srcSize.width+padding; ++w)
        {
            int tempVal = 0;
            // 再用两层for循环遍历一次kernel的大小，对所选到的点和这点周围的点求和
            for (int kh = -padding; kh <= padding; ++kh)
            {
                for (int kw = -padding; kw <= padding; ++kw)
                {
                    tempVal += src_pad.at<uchar>(h+kh, w+kw);
                }
            }
            // 最后求平均，也就是对求和后的值tempVal除以kernel的大小
            dst.at<uchar>(h-padding, w-padding) = uchar(tempVal/(ksize.width*ksize.height));
        }
    }
}

void boxFilterByHandFast(InputArray _src, OutputArray _dst, Size ksize)
{
    assert(ksize.height % 2 == 1 && ksize.height > 1);
    assert(ksize.height == ksize.width);

    // 获得_src的mat
    Mat src = _src.getMat();
    Size srcSize = src.size();
    // 对src的图形做拓展padding
    Mat src_pad;
    int padding = (ksize.width - 1) / 2;
    copyMakeBorder(src, src_pad, padding, padding, padding, padding, BORDER_REPLICATE, 0);
    // 初始化dst，作为结果的mat
    _dst.create(srcSize, src.type());
    Mat dst = _dst.getMat();
    // temp1是将原始图像对每一行进行卷积的操作之后的结果
    Mat temp1(src_pad.size().height, dst.size().width, CV_32SC1);
    // temp2是temp1的转置
    Mat temp2(dst.size().width, src_pad.size().height, CV_32SC1);
    // temp3是将temp2对每一行进行卷积操作之后的结果
    Mat temp3(dst.size().width, dst.size().height, CV_32SC1);
    // 对原图的每一行进行卷积的操作，并且在窗口的移动过程，每次并不用全部重新计算
    // 只需要减去窗口现在最左边的像素值，加上最右边的下一个像素值，这样减少了计算量
    for(int h = 0; h < temp1.size().height; h++)
    {
        int temp_value = 0;
        // row
        for (int k = 0; k < ksize.width; k++)
        {
            temp_value += (int)src_pad.at<uchar>(h, k);
        }
        temp1.at<int>(h, 0) = temp_value;
        for (int w = 1; w < temp1.size().width; w++)
        {
            temp_value += (int)(src_pad.at<uchar>(h, w + ksize.width - 1) - src_pad.at<uchar>(h, w - 1));
            temp1.at<int>(h, w) = temp_value;
        }
    }
    // 对每一列进行卷积操作
    // 这里我们将之前对行操作的结果进行转置再次重复该操作，就相当于对列操作
    transpose(temp1, temp2);
    for (int h = 0; h < temp3.size().height; h++)
    {
        int temp_value = 0;
        // row
        for (int k = 0; k < ksize.width; k++)
        {
            temp_value += (int)temp2.at<int>(h, k);
        }
        temp3.at<int>(h, 0) = temp_value;
        for (int w = 1; w < temp3.size().width; w++)
        {
            temp_value += (int)(temp2.at<int>(h, w + ksize.height - 1) - temp2.at<int>(h, w - 1));
            temp3.at<int>(h, w) = temp_value;
        }
    }
    // 最后需要将卷积后得到的值除以kernel的大小
    for (int h = 0; h < temp3.size().height; ++h)
    {
        for (int w = 0; w < temp3.size().width; ++w)
        {
            dst.at<uchar>(w, h) = uchar(temp3.at<int>(h, w) / (ksize.width * ksize.height));
        }
    }
}

void boxFilterTest()
{
    clock_t start, end;
    Mat image, result1, result2, result3;
    image = imread("../img/film.jpg", 0);
    namedWindow("image", 1);
    imshow("image", image);
    start = clock();
    boxFilterByHand(image, result1, Size(21, 21));
    end = clock();
    printf("the running time of boxFilterByHand():     %f\n", double(end-start)/CLOCKS_PER_SEC);

    start = clock();
    boxFilterByHandFast(image, result2, Size(21, 21));
    end = clock();
    printf("the running time of boxFilterByHandFast(): %f\n", double(end-start)/CLOCKS_PER_SEC);

    start = clock();
    boxFilter(image, result3, -1, Size(21, 21));
    end = clock();
    printf("the running time of cv::boxFilter():       %f\n", double(end-start)/CLOCKS_PER_SEC);
    namedWindow("result1", 1);
    imshow("result1", result1);
    namedWindow("result2", 1);
    imshow("result2", result2);
    namedWindow("result3", 1);
    imshow("result3", result3);
    waitKey();
}

#endif //CV_BOXFILTER_H
