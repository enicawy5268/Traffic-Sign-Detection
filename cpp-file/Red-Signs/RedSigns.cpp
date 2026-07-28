
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <vector>
#include "Supp.h"

using namespace cv;
using namespace std;

// --- Color mask ---
vector<Mat> colorMask(Mat src) {
    Mat hsv, blur;
    GaussianBlur(src, blur, Size(3, 3), 3);
    cvtColor(blur, hsv, COLOR_BGR2HSV);

    Mat redMask, redMask1, redMask2;
    Scalar red_lower1(0, 120, 0);
    Scalar red_upper1(10, 255, 255);
    inRange(hsv, red_lower1, red_upper1, redMask1);
    Scalar red_lower2(170, 120, 0);
    Scalar red_upper2(220, 255, 255);
    inRange(hsv, red_lower2, red_upper2, redMask2);
    redMask = redMask1 | redMask2;
    threshold(redMask, redMask, 0, 255, THRESH_BINARY + THRESH_OTSU);

    return { redMask };
}

// --- Shape detection ---
void detectShape(vector<Point> contour, Mat& show) {
    vector<Point> approx;
    vector<Point> hull;
    convexHull(contour, hull, false);
    approxPolyDP(hull, approx, 0.06 * arcLength(hull, true), true);

    if (approx.size() == 3) {
        fillPoly(show, hull, Scalar(255, 255, 255)); // 三角形
    }
    else {
        approxPolyDP(hull, approx, 0.01 * arcLength(hull, true), true);
        if (approx.size() > 10) {
            fillPoly(show, hull, Scalar(255, 255, 255)); // 圆形
        }
        else {
            fillPoly(show, contour, Scalar(255, 255, 255)); // 其他形状
        }
    }
}

// --- Safe crop ---
static Rect safeExpandRect(const Rect& r, int pad, const Size& sz) {
    int x = max(0, r.x - pad);
    int y = max(0, r.y - pad);
    int w = min(sz.width - x, r.width + 2 * pad);
    int h = min(sz.height - y, r.height + 2 * pad);
    return Rect(x, y, max(1, w), max(1, h));
}

int main() {
    String imgPattern("Inputs/Red signs/*.png");
    String outputDir("Outputs/Red segmentation/");
    vector<string> imageNames;
    glob(imgPattern, imageNames, true);

    if (imageNames.empty()) {
        cout << "No image found in: " << imgPattern << endl;
        return -1;
    }

    if (!cv::utils::fs::exists(outputDir)) {
        cv::utils::fs::createDirectories(outputDir);
    }

    for (size_t i = 0; i < imageNames.size(); ++i) {
        Mat srcI = imread(imageNames[i]);
        if (srcI.empty()) continue;

       
        const int noOfImagePerCol = 2, noOfImagePerRow = 3;
        Mat detailWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
        createWindowPartition(srcI, detailWin, win, legend, noOfImagePerCol, noOfImagePerRow);

        putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend[1], "Red Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend[2], "Contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend[3], "Best contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend[4], "Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend[5], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

        srcI.copyTo(win[0]);

        
        const int noOfImagePerCol2 = 1, noOfImagePerRow2 = 2;
        Mat resultWin, win2[noOfImagePerRow2 * noOfImagePerCol2], legend2[noOfImagePerRow2 * noOfImagePerCol2];
        createWindowPartition(srcI, resultWin, win2, legend2, noOfImagePerCol2, noOfImagePerRow2);

        putText(legend2[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend2[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

        srcI.copyTo(win2[0]);

        // --- Color mask ---
        vector<Mat> masks = colorMask(srcI);
        Mat redMask = masks[0];
        cvtColor(redMask, win[1], COLOR_GRAY2BGR);

        // --- Find contours ---
        vector<vector<Point>> contours;
        findContours(redMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (contours.empty()) continue;

        int bestIdx = -1;
        double maxArea = 0;
        double imgArea = srcI.rows * srcI.cols;

        for (int j = 0; j < (int)contours.size(); ++j) {
            double a = contourArea(contours[j]);
            if (a < 0.001 * imgArea || a > 0.5 * imgArea) continue;

            Rect rb = boundingRect(contours[j]);
            double aspect = (double)rb.width / max(1, rb.height);
            if (aspect < 0.5 || aspect > 2.0) continue;

            if (a > maxArea) { maxArea = a; bestIdx = j; }
        }

        if (bestIdx < 0) continue;

        // --- Show best contour ---
        Mat canvasGray = Mat::zeros(srcI.size(), CV_8U);
        drawContours(canvasGray, contours, bestIdx, Scalar(255), 2);
        cvtColor(canvasGray, win[3], COLOR_GRAY2BGR);

        // --- Shape detection ---
        Mat shapeMask = Mat::zeros(srcI.size(), CV_8U);
        detectShape(contours[bestIdx], shapeMask);
        cvtColor(shapeMask, win[4], COLOR_GRAY2BGR);

        // --- Segment sign ---
        Mat segmented = Mat::zeros(srcI.size(), CV_8UC3);
        srcI.copyTo(segmented, shapeMask);
        segmented.copyTo(win[5]);
        segmented.copyTo(win2[1]); 

        // --- Crop ---
        vector<Point> nz;
        findNonZero(shapeMask, nz);
        Rect cropBox = boundingRect(nz);
        cropBox = safeExpandRect(cropBox, 10, srcI.size());
        Mat croppedSegmented = segmented(cropBox).clone();

        // --- Save result ---
        String outName = outputDir + "segmented_" + to_string(i) + ".png";
        imwrite(outName, croppedSegmented);

        // --- Show results ---
        imshow("Traffic sign segmentation (6 windows)", detailWin);
        imshow("Traffic sign segmentation (Left-Right)", resultWin);
        waitKey(0);
        destroyAllWindows();
    }

    return 0;
}
