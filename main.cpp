#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "adaptiveThreshold.h"
#include "boxFilter.h"
#include "calGLCM.h"
#include "histEqual.h"
using namespace std;
using namespace cv;

int main()
{
    adaptiveThresholdTest();
    return 0;
}