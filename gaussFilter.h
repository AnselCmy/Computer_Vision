//
// Created by Chen on 2017/12/21.
//
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

#ifndef COMPUTER_VISION_GAUSSFILTER_H
#define COMPUTER_VISION_GAUSSFILTER_H

void guassFilter(InputArray _src, OutputArray _dst, Size ksize)
{
    assert(ksize.height % 2 == 1 && ksize.height > 1);
    assert(ksize.height == ksize.width);
}

#endif //COMPUTER_VISION_GAUSSFILTER_H
