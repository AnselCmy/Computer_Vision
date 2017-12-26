#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "adaptiveThreshold.h"
#include "boxFilter.h"
#include "GLCM.h"
#include "histEqual.h"
using namespace std;
using namespace cv;

int main()
{
//    adaptiveThresholdTest();
//    histEqualTest();
    calGLCMTest();
    return 0;
}