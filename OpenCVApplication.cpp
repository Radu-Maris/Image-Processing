// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <fstream>

wchar_t* projectPath;

using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))	//INVOCA APARITIA "browse for file"
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);	//ACCESAEZA pixelul (i,j)
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void add50()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))	//INVOCA APARITIA "browse for file"
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);	//ACCESAEZA pixelul (i,j)
				int plus50;
				if (val + 50 < 255) {
					plus50 = val + 50;
				}
				else {
					plus50 = 255;
				}
				dst.at<uchar>(i, j) = plus50;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("add 50 image", dst);
		waitKey();
	}
}

void minus50()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))	//INVOCA APARITIA "browse for file"
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);	//ACCESAEZA pixelul (i,j)
				int minus50;
				if (val - 50 > 0) {
					minus50 = val - 50;
				}
				else {
					minus50 = 0;
				}
				dst.at<uchar>(i, j) = minus50;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("minus 50 image", dst);
		waitKey();
	}
}

void fourColors()
{
	int height = 500;
	int width = 500;
	Mat_<Vec3b> dst(height, width, CV_8UC3);


	for (int i = 0; i < height / 2; i++)
	{
		for (int j = 0; j < width / 2; j++)
		{
			dst(i, j) = Vec3b(255, 255, 255);
		}
	}

	for (int i = height / 2; i < height; i++)
	{
		for (int j = 0; j < width / 2; j++)
		{
			dst(i, j) = Vec3b(0, 255, 0);
		}
	}

	for (int i = 0; i < height / 2; i++)
	{
		for (int j = width / 2; j < width; j++)
		{
			dst(i, j) = Vec3b(0, 0, 255);
		}
	}

	for (int i = height / 2; i < height; i++)
	{
		for (int j = width / 2; j < width; j++)
		{
			dst(i, j) = Vec3b(0, 255, 255);
		}
	}

	imshow("Four Colors", dst);
	waitKey();

}

void inverseOfaMatrix() {
	float vals[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 10 };
	Mat M(3, 3, CV_32FC1, vals);	//4 parameter constructor
	std::cout << M << std::endl;

	Mat x = M.inv();

	std::cout << x << std::endl;

	getchar();
	getchar();
}

void rgbSplitGrayscale() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat dstRed = Mat(height, width, CV_8UC1);
		Mat dstGreen = Mat(height, width, CV_8UC1);
		Mat dstBlue = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b val = src.at<Vec3b>(i, j);

				dstBlue.at<uchar>(i, j) = val[0];
				dstGreen.at<uchar>(i, j) = val[1];
				dstRed.at<uchar>(i, j) = val[2];
			}
		}

		imshow("image", src);
		imshow("Red", dstRed);
		imshow("Blue", dstBlue);
		imshow("Green", dstGreen);
		imshow("Green", dstGreen);

		waitKey();
	}
}

void blackWhiteConversion() {
	int threshold;
	std::cin >> threshold;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dstBW = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);

				if (threshold > val) {
					dstBW.at<uchar>(i, j) = 0;
				}
				else {
					dstBW.at<uchar>(i, j) = 255;
				}
			}
		}

		imshow("image", src);
		imshow("Black-White", dstBW);
		waitKey();
	}
}

void conversionRGB_HCV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat dstHue = Mat(height, width, CV_8UC1);
		Mat dstValue = Mat(height, width, CV_8UC1);
		Mat dstSaturation = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b val = src.at<Vec3b>(i, j);

				float b = (float)val[0] / 255;
				float g = (float)val[1] / 255;
				float r = (float)val[2] / 255;

				float M = max(r, max(g, b));	//Value = M
				float m = min(r, min(g, b));

				float C = M - m;
				float S;						//Saturation = S

				if (M != 0) {
					S = C / M;
				}
				else {
					S = 0;
				}

				float H;						//Hue = H
				if (C != 0) {
					if (M == r) {
						H = 60 * (g - b) / C;
					}
					if (M == g) {
						H = 120 + 60 * (b - r) / C;
					}
					if (M == b) {
						H = 240 + 60 * (r - g) / C;
					}
				}
				else {
					H = 0;
				}
				if (H < 0) {
					H = H + 360;
				}

				dstHue.at<uchar>(i, j) = H * 255 / 360;
				dstValue.at<uchar>(i, j) = M * 255;
				dstSaturation.at<uchar>(i, j) = S * 255;
			}
		}

		imshow("image", src);
		imshow("Hue", dstHue);
		imshow("Value", dstValue);
		imshow("Saturation", dstSaturation);
		waitKey();
	}
}

int inImage(Mat src, int a, int b) {
	int height = src.rows;
	int width = src.cols;

	if (height > a && width > b && a > 0 && b > 0) {
		return 1;
	}
	return 0;
}

void hystogramComputation() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		int h[256] = { 0 };

		Mat dstHystogram = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);
				h[val]++;
			}
		}

		imshow("image", src);
		showHistogram("Hystogram", h, 256, 500);
		waitKey();
	}
}

void normalizedHystogram() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		int M = height * width;
		int h[256] = { 0 };
		float p[256];

		Mat dstHystogram = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);
				h[val]++;
			}
		}

		for (int i = 0; i < 255; i++) {
			p[i] = float(h[i]) / float(M);
		}

		imshow("image", src);
		waitKey();
	}
}

void multiLevelTresholding(int WH, float TH) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		int M = height * width;
		int h[256] = { 0 };
		float p[256];
		float MLT[256] = { 0 };
		float average;
		int aux = 1;
		int ok = 0;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);
				h[val]++;
			}
		}

		for (int i = 0; i <= 255; i++) {
			p[i] = float(h[i]) / float(M);
		}

		for (int i = WH; i <= 255 - WH; i++) {
			average = 0;
			ok = 0;

			for (int j = i - WH; j < i + WH + 1; j++) {
				if (i != j && p[i] < p[j]) {
					ok = 1;
				}
				average = average + p[j];
			}

			average = average / (float)(2 * WH + 1);
			if (ok == 0 && p[i] > (average + TH)) {
				MLT[aux] = i;
				aux++;
			}

			MLT[aux] = 255;
			MLT[aux + 1] = 300;
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);
				aux = 0;
				while (val >= MLT[aux]) {
					aux++;
				}
				aux--;
				if (MLT[aux + 1] - (int)val > (int)val - MLT[aux]) {
					dst.at<uchar>(i, j) = (uchar)MLT[aux];
				}
				else {
					dst.at<uchar>(i, j) = (uchar)MLT[aux + 1];
				}
			}
		}

		imshow("image", src);
		imshow("Multilevel Tresholding", dst);
		waitKey();
	}
}

int saturate(int value) {
	if (value < 0) {
		return 0;
	}
	if (value > 255) {
		return 255;
	}
	return value;
}

void floydSteinberg(int WH, float TH) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		int M = height * width;
		int h[256] = { 0 };
		float p[256];
		float MLT[256] = { 0 };
		float average;
		int aux = 1;
		int ok = 0;

		Mat dst = src.clone();


		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);
				h[val]++;
			}
		}

		for (int i = 0; i <= 255; i++) {
			p[i] = float(h[i]) / float(M);
		}

		for (int i = WH; i <= 255 - WH; i++) {
			average = 0;
			ok = 0;

			for (int j = i - WH; j < i + WH + 1; j++) {
				if (i != j && p[i] < p[j]) {
					ok = 1;
				}
				average = average + p[j];
			}

			average = average / (float)(2 * WH + 1);
			if (ok == 0 && p[i] > (average + TH)) {
				MLT[aux] = i;
				aux++;
			}

			MLT[aux] = 255;
			MLT[aux + 1] = 300;
		}


		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				int floydError;
				uchar val = dst.at<uchar>(i, j);
				aux = 0;
				while (val >= MLT[aux]) {
					aux++;
				}
				aux--;
				if (MLT[aux + 1] - (int)val > (int)val - MLT[aux]) {
					dst.at<uchar>(i, j) = (uchar)MLT[aux];
				}
				else {
					dst.at<uchar>(i, j) = (uchar)MLT[aux + 1];
				}

				floydError = val - dst.at<uchar>(i, j);

				dst.at<uchar>(i, j + 1) = saturate(dst.at<uchar>(i, j + 1) + (7 / 16.f * floydError));
				dst.at<uchar>(i + 1, j - 1) = saturate(dst.at<uchar>(i + 1, j - 1) + (3 / 16.f * floydError));
				dst.at<uchar>(i + 1, j) = saturate(dst.at<uchar>(i + 1, j) + (5 / 16.f * floydError));
				dst.at<uchar>(i + 1, j + 1) = saturate(dst.at<uchar>(i + 1, j + 1) + (1 / 16.f * floydError));

			}
		}

		imshow("image", src);
		imshow("Floyd-Steinberg", dst);
		waitKey();
	}
}

void geometricalFeaturesComputation(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	Mat dst = src->clone();
	Mat dst1 = Mat(src->rows, src->cols, CV_8UC3);
	Mat dst2 = Mat(src->rows, src->cols, CV_8UC3);
	Mat dst3 = src->clone();
	if (event == EVENT_LBUTTONDBLCLK)
	{
		int Area = 0;
		int r = 0, c = 0;
		float R = 0, C = 0;

		Vec3b color = src->at<Vec3b>(y, x);
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);

		for (int i = 0; i < src->rows; i++) {
			for (int j = 0; j < src->cols; j++) {
				if (src->at<Vec3b>(i, j) == color) {
					Area = Area + 1;
					r = r + i;
					c = c + j;
				}
			}
		}

		R = (float)1 / Area * r;
		C = (float)1 / Area * c;

		float aux1 = 0, aux2 = 0, aux3 = 0;
		float Fi = 0;
		float FiDegree = 0;

		for (int i = 0; i < src->rows; i++) {
			for (int j = 0; j < src->cols; j++) {
				if (src->at<Vec3b>(i, j) == color) {
					aux1 = aux1 + (i - R) * (j - C);
					aux2 = aux2 + (j - C) * (j - C);
					aux3 = aux3 + (i - R) * (i - R);
				}
			}
		}

		Fi = 1 / float(2) * atan2(2 * aux1, (aux2 - aux3));
		if (Fi < 0) {
			Fi = Fi + CV_PI;
		}
		FiDegree = Fi * float(180) / CV_PI;

		float Perimeter = 0;
		int alb = 0;
		int NP = 0;
		float ThicknessRatio = 0;
		float AspectRatio = 0;
		int Cmax = 0, Rmax = 0, Cmin = INT_MAX, Rmin = INT_MAX;

		for (int i = 1; i < src->rows - 1; i++) {
			for (int j = 1; j < src->cols - 1; j++) {
				if (src->at<Vec3b>(i, j) == color) {
					alb = 0;

					for (int i1 = i - 1; i1 <= i + 1; i1++) {
						for (int j1 = j - 1; j1 <= j + 1; j1++) {

							if (i1 != i || j1 != j) {
								if (src->at<Vec3b>(i1, j1) != color) {
									alb = 1;
								}
							}

						}
					}
					if (alb == 1) {
						NP++;
						dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					}
					if (Cmax < j) {
						Cmax = j;
					}
					if (Cmin > j) {
						Cmin = j;
					}
					if (Rmax < i) {
						Rmax = i;
					}
					if (Rmin > i) {
						Rmin = i;
					}
				}
			}
		}

		Perimeter = NP * CV_PI / 4;
		ThicknessRatio = 4 * CV_PI * Area / (Perimeter * Perimeter);
		AspectRatio = (float)(Cmax - Cmin + 1) / (Rmax - Rmin + 1);

		int iH = 0, jH = 0;
		for (int i = 0; i < src->rows; i++) {
			for (int j = 0; j < src->cols; j++) {
				if (src->at<Vec3b>(i, j) == color) {
					dst1.at<Vec3b>(i, jH) = color;
					jH++;
				}
			}
			jH = 0;
		}

		iH = src->rows - 1;
		for (int j = 0; j < src->cols; j++) {
			for (int i = 0; i < src->rows; i++) {
				if (src->at<Vec3b>(i, j) == color) {
					dst2.at<Vec3b>(iH, j) = color;
					iH--;
				}
			}
			iH = src->rows - 1;
		}

		Point PointA, PointB;

		int ra, rb;
		ra = R + tan(Fi) * (Cmin - C);
		rb = R + tan(Fi) * (Cmax - C);

		PointA = { Cmin, ra };
		PointB = { Cmax, rb };

		line(dst3, PointA, PointB, Scalar(0, 0, 0), 2);

		imshow("Perimeter", dst);
		imshow("Horizaontal Projection", dst1);
		imshow("Vertical Projection", dst2);
		imshow("Ax of elongation", dst3);
		printf("Area = %d\n", Area);
		printf("Center of mass = %f, %f\n", R, C);
		printf("The Ax of elongation = %f\n", FiDegree);
		printf("Perimeter = %f\n", Perimeter);
		printf("Thickness Ration = %f\n", ThicknessRatio);
		printf("Aspect Ratio = %f\n", AspectRatio);
	}
}

void geometricalFeatures() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_COLOR);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", geometricalFeaturesComputation, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);

	}
}

void objectsGraytoColor() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("image", src);
		int height = src.rows;
		int width = src.cols;
		
		Mat dst = Mat(height, width, CV_8UC3,Scalar(255,255,255));

		Mat labels(height, width, CV_32SC1, Scalar(0));
		std::queue<Point> Q;
		int label = 0;

		int di[8] = {0,-1,-1,-1, 0, 1, 1, 1};
		int dj[8] = {1, 1, 0,-1,-1,-1, 0, 1};

		for (int i = 1; i < height-1; i++) {
			for (int j = 1; j < width-1; j++) {
				Vec3b color(rand() % 256, rand() % 256, rand() % 256);
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					label++;
					labels.at<int>(i, j) = label;
					
					Q.push(Point(i, j));
					while (!Q.empty()) {
						Point q = Q.front();
						Q.pop();
						for (int k = 0; k < 8; k++) {
							if (src.at<uchar>(q.x + di[k], q.y + dj[k]) == 0 && labels.at<int>(q.x + di[k], q.y + dj[k])==0) {
								labels.at<int>(q.x + di[k], q.y + dj[k]) = label;
								Q.push(Point(q.x + di[k], q.y + dj[k]));
							}
						}
						dst.at<Vec3b>(q.x, q.y) = color;
					}
					waitKey();
					imshow("Colored Objects", dst);
				}
			}
		}
		waitKey();
	}
}

void two_pass_Component_Labeling() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));

		Mat labels(height, width, CV_32SC1, Scalar(0));
		std::queue<int> Q;
		int label = 0;

		int x;

		std::vector<std::vector<int>> edges(1000);

		int di[8] = { 0,-1,-1,-1, 0, 1, 1, 1 };
		int dj[8] = { 1, 1, 0,-1,-1,-1, 0, 1 };

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					
					std::vector<int> L;

					for (int k = 0; k < 8; k++) {
						if (labels.at<int>(i + di[k], j + dj[k]) > 0) {
							L.push_back(labels.at<int>(i + di[k], j + dj[k]));
						}
					}

					if (L.size() == 0) {
						label++;
						labels.at<int>(i, j) = label;
					}
					else {
						x = *std::min_element(L.begin(), L.end());
						labels.at<int>(i, j) = x;
						for (int y = 0; y < L.size(); y++) {
							if (L[y] != x) {
								edges[x].push_back(L[y]);
								edges[L[y]].push_back(x);
							}
						}
					}
					
				}
			}
		}

		int newlabel = 0;
		std::vector<int> newlabels(label + 1,0);

		Vec3b colors[1000];

		for (int i = 1; i <= label; i++) {

			Vec3b new_color(rand() % 256, rand() % 256, rand() % 256);
			colors[i] = new_color;

			if (newlabels[i] == 0) {
				newlabel++;
				newlabels[i] = newlabel;
				
				Q.push(i);
				while (!Q.empty()) {
					x = Q.front();
					Q.pop();
					for (int y = 0; y < edges[x].size(); y++) {
						if (newlabels[edges[x][y]] == 0) {
							newlabels[edges[x][y]] = newlabel;
							Q.push(edges[x][y]);
						}
					}
				}
				
			}
		}

		for (int i = 0; i < height - 1; i++) {
			for (int j = 0; j < width - 1; j++) {
				if (labels.at<int>(i, j) != 0) {
					labels.at<int>(i, j) = newlabels[labels.at<int>(i, j)];
				}
			}
		}
       
		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				dst.at<Vec3b>(i, j) = colors[labels.at<int>(i, j)];
			}
		}


		imshow("image", src);
		imshow("Colored Objects", dst);
		waitKey();
	}
}

void lab6_border(){

	Mat_<uchar> img = imread("Images/triangle_up.bmp", 0);
	Mat_<uchar> dst(img.rows, img.cols);
	dst.setTo(255);

	int di[] = { 0, -1, -1, -1,  0,  1, 1, 1};
	int dj[] = { 1,  1,  0, -1, -1, -1, 0, 1};

	std::vector<int> dirs;						//vector pentru directii
	std::vector<std::pair<int, int>> pts;		//vector pentru punctele de pe contur

	std::vector<int> derivate;
	int derivate_aux;

	int firsti=0, firstj=0;

	for (int i = 0; i < img.rows;i++) {
		for (int j = 0; j < img.cols;j++) {
			if (img(i,j)==0) {
				firsti = i;
				firstj = j;
				pts.push_back({ i,j });
				dst(i, j) = 0;
				goto et;
			}
		}
	}

et:
	int dir = 7;
	while (1) {
		if (dir % 2 == 0) {
			dir = (dir + 7) % 8;
		}
		else {
			dir = (dir + 6) % 8;
		}

		for (int k = 0; k < 8; k++) {			//parcurgem vecinii incepand de la dir
			int dirnow = (dir + k) % 8;
			int i2 = pts.back().first + di[dirnow];
			int j2 = pts.back().second + dj[dirnow];

			if (firsti == i2 && firstj == j2) {
				goto et2;
			}

			if (img(i2, j2) == 0) {
				pts.push_back({ i2,j2 });
				dst(i2, j2) = 0;
				dir = dirnow;
				break;
			}
		}
		dirs.push_back(dir);

	}
et2:
	printf("Directions: \n");
	for (int i = 0; i < dirs.size(); i++) {
		printf("%d ", dirs[i]);
	}

	for (int i = 1; i < dirs.size(); i++) {
		derivate.push_back((dirs[i] - dirs[i-1] + 8)%8);
	}


	printf("\nDerivate: \n");
	for (int i = 0; i < derivate.size(); i++) {
		printf("%d ", derivate[i]);
	}

	imshow("image", img);
	imshow("border", dst);
	waitKey();
}

void lab6_reconstruct() {
	string myText;

	std::ifstream file("Images/reconstruct.txt");

	int starti, startj, nr, dir;
	file >> starti >> startj >> nr;

	Mat_<uchar> dst(500,800);
	dst.setTo(255);

	dst(starti, startj) = 0;

	int di[] = { 0, -1, -1, -1,  0,  1, 1, 1 };
	int dj[] = { 1,  1,  0, -1, -1, -1, 0, 1 };

	while (file) {
		file >> dir;
		starti = starti + di[dir];
		startj = startj + dj[dir];
		dst(starti, startj) = 0;
	}

	imshow("border", dst);
	waitKey();
}

Mat_<uchar> lab7_dilate(Mat_<uchar> img, Mat_<uchar> strel) {
	Mat_<uchar> dst(img.rows,img.cols);
	dst.setTo(255);
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				for (int u =0; u < strel.rows; u++) {
					for (int v = 0; v < strel.cols; v++) {
						if (!strel(u, v)) {
							int i2 = i + u - strel.rows / 2;
							int j2 = j + v - strel.cols / 2;
							
							if (j2 >= 0 && i2 < img.rows-1 && j2>=0 && img.cols-1) {
								dst(i, j) = 0;
							}
							
						}
					}
				}
			}
		}
	}
	return dst;
}

void n_dilate(int n) {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", src);
	Mat_<uchar> strel(3, 3);
	Mat aux = src;
	for (int i = 0; i < n; i++)aux = lab7_dilate(aux,strel);
	imshow("dilated", aux);
	waitKey(0);
}

Mat erode(Mat src) {
	Mat aux;
	src.copyTo(aux);
	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {
				if (src.at<uchar>(i - 1, j - 1) == 255 ||
					src.at<uchar>(i - 1, j) == 255 ||
					src.at<uchar>(i, j - 1) == 255 ||
					src.at<uchar>(i + 1, j) == 255 ||
					src.at<uchar>(i, j + 1) == 255 ||
					src.at<uchar>(i + 1, j + 1) == 255 ||
					src.at<uchar>(i + 1, j - 1) == 255 ||
					src.at<uchar>(i - 1, j + 1) == 255)
				{
					aux.at<uchar>(i, j) = 255;
				}
			}
		}
	}
	return aux;
}

void n_erode(int n) {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", src);
	Mat aux = src;
	for (int i = 0; i < n; i++)aux = erode(aux);
	imshow("eroded", aux);
	waitKey(0);
}

Mat opening(Mat src) {
	Mat aux;
	src.copyTo(aux);
	aux = erode(aux);
	Mat_<uchar> strel(3, 3);
	aux = lab7_dilate(aux,strel);
	return aux;
}

Mat closing(Mat src) {
	Mat aux;
	src.copyTo(aux);
	Mat_<uchar> strel(3, 3);
	aux = lab7_dilate(aux,strel);
	aux = erode(aux);
	return aux;
}

void opening_n(int n) {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", src);
	Mat aux = src;
	for (int i = 0; i < n; i++)aux = opening(aux);
	imshow("opened", aux);
	waitKey(0);
}

void closing_n(int n) {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", src);
	Mat aux = src;
	for (int i = 0; i < n; i++)aux = closing(aux);
	imshow("closed", aux);
	waitKey(0);
}

void extract() {
	Mat src;
	Mat dst;
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", src);
	Mat aux;
	src.copyTo(aux);
	aux = erode(aux);
	bitwise_xor(src, aux, dst);
	bitwise_not(dst, dst);
	imshow("extracted", dst);
	waitKey(0);
}

bool equal(Mat a, Mat b) {
	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			if (a.at<uchar>(i, j) != b.at<uchar>(i, j)) return false;
		}
	}
	return true;
}

Mat intersect(Mat a, Mat b) {
	Mat intersect(a.rows, a.cols, CV_8UC1);
	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			if (a.at<uchar>(i, j) == b.at<uchar>(i, j) && a.at<uchar>(i, j) == 0)
				intersect.at<uchar>(i, j) = 0;
			else intersect.at<uchar>(i, j) = 255;
		}
	}
	return intersect;
}

void region_fill(int start_x, int start_y) {
	Mat src;
	Mat_<uchar> strel(3, 3);
	char fname[MAX_PATH];
	openFileDlg(fname);
	src = imread(fname, IMREAD_GRAYSCALE);
	imshow("original", src);
	Mat aux(src.rows, src.cols, CV_8UC1);
	Mat complement(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255)complement.at<uchar>(i, j) = 0;
			else complement.at<uchar>(i, j) = 255;
		}
	}
	aux.at<uchar>(start_x, start_y) = 0;
	Mat aux1 = lab7_dilate(aux,strel);
	aux1 = intersect(aux1, complement);
	while (!equal(aux1, aux)) {
		aux = aux1;
		aux1 = lab7_dilate(aux,strel);
		aux1 = intersect(aux1, complement);
	}
	imshow("fill", aux);
	waitKey(0);
}

void wrapper() {
	Mat_<uchar> strel(3, 3);
	strel.setTo(0);
	Mat_<uchar> img = imread("Images/Morphological_Op_Images/1_Dilate/wdg2thr3_bw.bmp", 0);
	Mat_<uchar> dst = lab7_dilate(img, strel);
	imshow("dilation", dst);
	waitKey();
}

float meanIntensity(Mat_<uchar> img) {
	int nrOfPixels = img.rows * img.cols;

	int h[256] = {0};
	int aux = 0;
	float meanI = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uchar val = img.at<uchar>(i, j);
			h[val]++;
		}
	}

	for (int g = 0; g <= 255; g++) {
		aux += g * h[g];
	}

	meanI = 1 / (float)nrOfPixels * aux;

	return meanI;

}

std::vector<float> normalizedHystogram2(Mat_<uchar> src) {
	
	int height = src.rows;
	int width = src.cols;

	int M = height * width;
	int h[256] = { 0 };
	std::vector<float> p(256);

	Mat dstHystogram = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar val = src.at<uchar>(i, j);
			h[val]++;
		}
	}

	for (int i = 0; i < 255; i++) {
		p[i] = float(h[i]) / float(M);
	}

	return p;
}

float standardDeviation(Mat_<uchar> img) {
	float stdDev = 0;
	float aux = 0;
	float meanI = meanIntensity(img);
	std::vector<float> p(256);
	p = normalizedHystogram2(img);
	for (int i = 0; i <= 255; i++) {
		aux += (i - meanI) * (i - meanI) * p[i];
	}

	stdDev = sqrt(aux);

	return stdDev;
}

std::vector<int> cummulativeHistogram(Mat_<uchar> img) {
	
	std::vector<int> h(256);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uchar val = img.at<uchar>(i, j);
			h[val]++;
		}
	}

	for (int i = 1; i < 256; i++) {
		h[i] = h[i] + h[i-1];
	}

	return h;
}

void globalTresholding(Mat_<uchar> img) {
	std::vector<int> h(256);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uchar val = img.at<uchar>(i, j);
			h[val]++;
		}
	}

	int auxMax = -1, auxMin = 999999;
	int Imax, Imin;
	int N1=0, N2=0;
	int aux1=0, aux2=0;

	for (int i = 0; i < 256; i++) {
		if (auxMax < h[i]) {
			auxMax = h[i];
			Imax = i;
		}
		if (auxMin > h[i]) {
			auxMin = h[i];
			Imin = i;
		}
	}

	float T = (Imin + Imax) / 2.0;
	float Tnew = T;
	do {
		T = Tnew;
		for (int i = Imin; i <= T; i++) {
			N1 += h[i];
		}

		for (int i = T + 1; i <= Imax; i++) {
			N2 += h[i];
		}

		for (int i = Imin; i <= T; i++) {
			aux1 += i * h[i];
		}

		for (int i = T + 1; i <= Imax; i++) {
			aux2 += i * h[i];
		}

		float mean1 = 0, mean2 = 0;
		mean1 = 1 / (float)N1 * aux1;
		mean2 = 1 / (float)N2 * aux2;

		Tnew = (mean1 + mean2) / 2.0;
	} while ((Tnew-T)>0.1);

	Mat dst = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j)<T) {
				dst.at<uchar>(i, j) = 0;
			}
			else {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
	imshow("Global Threshold", dst);
}

void hystogramStreching(Mat_<uchar> img, int gOutMin, int gOutMax) {
	int h[256] = { 0 };
	int g[256] = { 0 };

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uchar val = img.at<uchar>(i, j);
			h[val]++;
		}
	}

	std::vector<int> hNew(256);

	int ok = 0;
	int gInMin = 0, gInMax = 0;

	for (int i = 0; i < 256; i++) {
		if (h[i] > 0 && ok == 0) {
			gInMin = i;
			ok = 1;
		}
		if (h[i] > 0) {
			gInMax = i;
		}
	}

	Mat_<uchar> dst;
	dst.setTo(255);

	for (int i = 0; i < 256; i++) {
		g[i] = gOutMin + (h[i] - gInMin) * (gOutMax - gOutMin) / (gInMax - gInMin);
	}

	imshow("Destinatie", dst);
	waitKey();
}

void gamma(float gamma) {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	int h[256] = { 0 };
	src = imread(fname, IMREAD_GRAYSCALE);
	
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			h[src.at<uchar>(i, j)]++;
		}
	}
	Mat src1 = src.clone();

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			src1.at<uchar>(i, j) = max(0, min(255.0 * pow((float)(src.at<uchar>(i, j) / 255.0), gamma), 255.0));
		}
	}
	imshow("gammaimg", src1);
	waitKey(0);
}

void equalizeHisto() {
	Mat src;
	char fname[MAX_PATH];
	openFileDlg(fname);
	
	int h[256] = { 0 };
	src = imread(fname, IMREAD_GRAYSCALE);
	
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			h[src.at<uchar>(i, j)]++;
		}
	}
	
	int cumm_hist[256] = { 0 };
	for (int i = 1; i < 256; i++) {
		cumm_hist[i] += cumm_hist[i - 1] + h[i];
	}
	
	float cpdf[256] = { 0.0 };
	for (int i = 0; i < 256; i++) {
		cpdf[i] = cumm_hist[i] / (float)(src.rows * src.cols);
	}
	
	Mat src1 = src.clone();
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			src1.at<uchar>(i, j) = 255 * cpdf[src.at<uchar>(i, j)];
		}
	}
	
	int h1[256] = { 0 };
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			h1[src1.at<uchar>(i, j)]++;
		}
	}
	imshow("good", src1);
	showHistogram("equal", h1, 256, 256);
	waitKey(0);
}

void wrapper1() {
	Mat_<uchar> img = imread("Images/balloons.bmp", 0);
	float meanI = meanIntensity(img);
	printf("%f", meanI);
	imshow("image", img);
	waitKey();
}

void wrapper2() {
	Mat_<uchar> img = imread("Images/balloons.bmp", 0);
	float stdDev = standardDeviation(img);
	printf("\n%f", stdDev);
	imshow("image", img);
	waitKey();
}

void wrapper3() {
	Mat_<uchar> img = imread("Images/balloons.bmp", 0);
	std::vector<int> h = cummulativeHistogram(img);
	showHistogram("Histogram", h.data(), 256, 256);
	imshow("image", img);
	waitKey();
}

void wrapper4() {
	Mat_<uchar> img = imread("Images/eight.bmp", 0);
	globalTresholding(img);
	imshow("image", img);
	waitKey();
}

void wrapper5() {
	Mat_<uchar> img = imread("Images/eight.bmp", 0);
	hystogramStreching(img, 10,250);
	waitKey();
}

bool isInside(int height, int width, int row, int column) {
	if (row < 0 || column < 0) return false;
	return row < height&& column < width;
}

Mat_<float> conv(Mat_<uchar> src, Mat_<float> H) {
	
	Mat_<float> dst(src.rows, src.cols);
	dst.setTo(0);

	int startU, startV, stopU, stopV;
	int i2, j2;
	float sumAux = 0;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			
			startU = 0;
			startV = 0;
			stopU = H.rows - 1;
			stopV = H.cols - 1;

			sumAux = 0;
			
			for (int u = 0; u < H.rows; u++) {
				for (int v = 0; v < H.cols; v++) {
					i2 = i + u - H.rows / 2;
					j2 = j + v - H.cols / 2;
					if (isInside(src.rows, src.cols, i2, j2) == 1) {
						sumAux = sumAux + src(i2, j2) * H(u, v);
					}
				}
			}

			dst(i, j) = sumAux;
		}
	}
	return dst;
}

Mat_<uchar> norm(Mat_<float> dst, Mat_<float> H) {
	int a, b;
	int c = 0, d = 255;
	int sumNegative = 0, sumPositive=0;

	for (int i = 0; i < H.rows; i++) {
		for (int j = 0; j < H.cols; j++) {
			if (H(i, j) < 0) {
				sumNegative += H(i, j);
			}
			else {
				sumPositive += H(i, j);
			}
		}
	}

	a = sumNegative * 255;
	b = sumPositive * 255;

	Mat_<uchar> dstn(dst.rows, dst.cols);
	dstn.setTo(255);

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			dstn(i, j) = (dst(i, j) - a) * (d - c) / (b - a) + c;
		}
	}

	return dstn;
}

Mat_<float> createTest(int w) {

	Mat_<float> H(1, 2*w+1, 1.f);

	for (int i = 0; i < w; i++) {
		H(0, i) = i + 1.f;
	}

	H(0, w) = 0.f;

	for (int i = 1; i <= w; i++) {
		H(0, w+i) = -w+i-1 + 0.f;
	}

	for (int i = 0; i < 2*w+1; i++) {
		cout << H(0, i) << ", ";
	}

	return H;
}

Mat_<uchar> testC(Mat_<float> dst, Mat_<float> H) {
	int a, b;
	int c = 0, d = 255;
	int sumNegative = 0, sumPositive = 0;

	for (int i = 0; i < H.rows; i++) {
		for (int j = 0; j < H.cols; j++) {
			if (H(i, j) < 0) {
				sumNegative += H(i, j);
			}
			else {
				sumPositive += H(i, j);
			}
		}
	}

	a = sumNegative * 255;
	b = sumPositive * 255;

	Mat_<uchar> afisareMuchii(dst.rows, dst.cols);
	afisareMuchii.setTo(0);


	Mat_<uchar> dstn(dst.rows, dst.cols);
	dstn.setTo(255);

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			dstn(i, j) = (dst(i, j) - a) * (c - d) / (b - a) + d;
			if (abs(a) + b > 100) {
				afisareMuchii(i, j) = 255;
			}
		}
	}

	imshow("afisareMuchii", afisareMuchii);

	return dstn;
}

void wrapper6() {
	Mat_<uchar> src = imread("Images/cameraman.bmp", 0);
	Mat_<float> H(3, 3, 1.f);
	H(0, 0) = 0.f; H(0, 1) = -1.f; H(0, 2) = 0.f;
	H(1, 0) = -1.f; H(1, 1) = 5.f; H(1, 2) = -1.f;
	H(2, 0) = 0.f; H(2, 1) = -1.f; H(2, 2) = 0.f;
	auto dst = conv(src, H);
	auto dstn = norm(dst, H);
	imshow("src", src);
	imshow("dstn", dstn);
	waitKey();
}

void wrapper7() {
	Mat_<uchar> src = imread("Images/cameraman.bmp", 0);
	Mat_<float> H = createTest(3);
	auto dst = conv(src, H);
	auto dstn = testC(dst, H);
	imshow("src", src);
	imshow("dstn", dstn);
	waitKey();
}

/*
	filtru general in domeniul frecventa

	1. convertim din uchar in float
	2. operatia de centrare (sa avem in centru frecventele joase
	3. aplicam DFT -> o imagine cu numere complexe
				   -> o imagine cu 2 canale flotante = parte reala / imaginara
	4. transformam in reprezentare polara
	   magnitudinea = sqrt(parteReala^2 + parteImaginara^2)
	   phi = atan2(parteImaginara, parteReala)
	
	5. filtram imaginea de magnitudine
	o singura functie care dicteaza daca e filtru trece sus sau trece jos, cu 2 parametri
	-parametru daca e trece sus sau trece jos
	-parametru daca e ideal sau gausian

	6. transformam magnitudinea filtrata inapoi in reprezentare carteziana
	7. aplicam DFT invers
	8. operatie de centrare (corectie)
	9. transformam din float in uchar

	pentru vizualizare:
		- la Mat_<float>	valori <= 0		--->	negru
							valori >= 1		--->	alb
							valori (0,1)	--->	nuante de gri
		- aplicati log(1 + mag)

		magnitudinea input trebuie sa fie intre [0, 1]
		(gasesti minumu si maximu si min=0 si max=1)
*/

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat generic_frequency_domain_filter(Mat src, int low_pass, int gaussian, int select, int R) {
	//convert input image to float image
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	//centering transformation
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
	
	//split into real and imaginary channels
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
	
	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);
	
	//display the phase and magnitude images here
	// ......
	Mat mag2;
	log(mag+1, mag2);

	Mat magNorm;
	normalize(mag2, magNorm, 0, 255, NORM_MINMAX, CV_8UC1);

	Mat phi2;
	log(phi+1, phi2);
	
	Mat phiNorm;
	normalize(phi, phiNorm, 0, 255, NORM_MINMAX, CV_8UC1);
	
	imshow("normalized", magNorm);
	imshow("phase", phiNorm);
	
	waitKey(0);
	
	//insert filtering operations on Fourier coefficients here
	// ......
	
	int aux1 = channels[0].rows;
	int aux2 = channels[0].cols;

	if (select == 0) {
		if (low_pass == 0) {
			for (int i = 0; i < aux1; i++) {
				for (int j = 0; j < aux2; j++) {
					if (((aux1/2 - i) * (aux1/2 - i)) + ((aux2/2 - j) * (aux2/2 - j)) < R)
						channels[0].at<float>(i, j) = channels[1].at<float>(i, j) = 0.0;;
				}
			}
		}
		else{
			for (int i = 0; i < aux1; i++) {
				for (int j = 0; j < aux2; j++) {
					if (((aux1 / 2 - i) * (aux1 / 2 - i)) + ((aux2 / 2 - j) * (aux2 / 2 - j)) >= R)
						channels[0].at<float>(i, j) = channels[1].at<float>(i, j) = 0.0;
				}
			}
		}
	}
	else {
		if (gaussian == 0) {
			for (int i = 0; i < aux1; i++) {
				for (int j = 0; j < aux2; j++) {
					float nr = ((aux1 / 2 - i) * (aux1 / 2 - i)) + ((aux2 / 2 - j) * (aux2 / 2 - j));
					nr /= R;
					channels[0].at<float>(i, j) = channels[0].at<float>(i, j) * exp(-nr);
					channels[1].at<float>(i, j) = channels[1].at<float>(i, j) * exp(-nr);
				}
			}
		}
		else{
			for (int i = 0; i < aux1; i++) {
				for (int j = 0; j < aux2; j++) {
					float nr = ((aux1 / 2 - i) * (aux1 / 2 - i)) + ((aux2 / 2 - j) * (channels[0].cols / 2 - j));
					nr /= R;
					channels[0].at<float>(i, j) = channels[0].at<float>(i, j) * (1 - exp(-nr));
					channels[1].at<float>(i, j) = channels[1].at<float>(i, j) * (1 - exp(-nr));
				}
			}
		}
	}

	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	
	//inverse centering transformation
	centering_transform(dstf);
	
	//normalize the result and put in the destination image
	
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	//Note: normalizing distorts the resut while enhancing the image display in the range [0,255].
	//For exact results (see Practical work 3) the normalization should be replaced with convertion:
	//dstf.convertTo(dst, CV_8UC1);
	return dst;
}

void wrapper8() {
	Mat_<uchar> src = imread("Images/cameraman.bmp", 0);
	Mat dst = generic_frequency_domain_filter(src,1,1,0,400);
	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}

void suprapunere_imagini() {
	Mat_<uchar> img1 = imread("Images/raduIP.jpg", 0);
	Mat_<uchar> img2 = imread("Images/amongus.jpg", 0);
	//imshow("img1", img1);
	//imshow("img2", img2);

	resize(img2, img2, img1.size());

	Mat dst1 = generic_frequency_domain_filter(img1, 1, 0, 0, 1000);
	Mat dst2 = generic_frequency_domain_filter(img2, 0, 1, 0, 1000);

	imshow("dst1", dst1);
	imshow("dst2", dst2);

	Mat dst3 = generic_frequency_domain_filter(img1, 1, 1, 0, 400);
	for (int i = 0; i < dst1.rows; i++) {
		for (int j = 0; j < dst1.cols; j++) {
			dst3.at<uchar>(i, j) = (dst1.at<uchar>(i, j)*0.3 + dst2.at<uchar>(i, j)*0.7);
		}
	}

	imshow("dst3", dst3);
	waitKey();
}

/*
	
	Mat_<uchar> median_filter(Mat_<uchar> img, int w)

	- pt fiecare pixel consideram vecinii dintr-o fereastra w x w
	- sortam vecinii
	- inlocuim pixelul central cu mediana
	
	2) Mat_<uchar> Gauss2d(Mat_<uchar> img, int w)

	- w = 6sigma => sigma = w/6.0
	- construim nucleul gausian w x w
	- aplicam convolutie cu acest nucleu
	- dst = G o img

	3) Mat_<uchar> Gauss_1d(Mat_<uchar> img, int w)

	- aplicati 10.8
	- filtram img cu Gy, apoi rezultatul cu Gx
	- dst = Gx o (Gy o img)

*/

Mat_<uchar> median_filter(Mat_<uchar> img, int w) {

	int a[100];
	int nrPixels = 0;

	Mat_<uchar> dst(img.rows, img.cols);
	dst.setTo(255);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (isInside(img.rows, img.cols, i-w/2, j-w/2) && isInside(img.rows, img.cols, i - w / 2, j + w / 2) && isInside(img.rows, img.cols, i + w / 2, j - w / 2) && isInside(img.rows, img.cols, i + w / 2, j + w / 2)) {
				for (int k = i - w / 2; k <= i + w / 2; k++) {
					for (int l = j - w / 2; l <= j + w / 2; l++) {
						a[nrPixels] = img.at<uchar>(k, l);
						nrPixels++;
					}
				}

				sort(a,a + nrPixels);

				dst.at<uchar>(i, j) = a[nrPixels / 2];

				nrPixels = 0;
			}
		}
	}
	return dst;
}

void wrapper9() {
	Mat_<uchar> src = imread("Images/balloons_Salt&Pepper.bmp", 0);
	Mat dst = median_filter(src, 5);
	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}

Mat_<uchar> Gauss2d(Mat_<uchar> img, int w) {
	double t = (double)getTickCount();
	
	float sigma = w / 6.0;

	Mat_<float> G(w,w);

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			G.at<float>(i, j) = 1 / (2 * PI * sigma * sigma) * exp((- (i - w / 2) * (i - w / 2) - (j - w / 2) * (j - w / 2)) / (2 * sigma * sigma));
		}
	}
	auto aux = conv(img,G);
	Mat_<uchar> dst;

	aux.convertTo(dst,CV_8UC1);

	t = ((double)getTickCount() - t) / getTickFrequency();
	// Display (in the console window) the processing time in [ms]
	printf("Time = %.3f [ms]\n", t * 1000);

	return dst;
}

void wrapper10() {
	Mat_<uchar> src = imread("Images/portrait_Gauss2.bmp", 0);
	Mat dst = Gauss2d(src, 5);
	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}

Mat_<uchar> Gauss_1d(Mat_<uchar> img, int w) {
	double t = (double)getTickCount();
	
	float sigma = w / 6.0;

	Mat_<float> Gx(1, w);
	Mat_<float> Gy(w, 1);

	for (int i = 0; i < w; i++) {
		Gx.at<float>(0, i) = 1 / (sqrt(2 * PI) * sigma) * exp((-(i - w / 2) * (i - w / 2)) / (2 * sigma * sigma));
		Gy.at<float>(i, 0) = 1 / (sqrt(2 * PI) * sigma) * exp((-(i - w / 2) * (i - w / 2)) / (2 * sigma * sigma));
	}

	auto aux = conv(img, Gx);
	Mat_<uchar> dst_aux;
	aux.convertTo(dst_aux, CV_8UC1);

	auto aux2 = conv(dst_aux, Gy);
	Mat_<uchar> dst;
	aux2.convertTo(dst, CV_8UC1);

	t = ((double)getTickCount() - t) / getTickFrequency();
	// Display (in the console window) the processing time in [ms]
	printf("Time = %.3f [ms]\n", t * 1000);
	
	return dst;
}

void wrapper11() {
	Mat_<uchar> src = imread("Images/portrait_Gauss2.bmp", 0);
	Mat dst = Gauss_1d(src, 5);
	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}

/*
	1. Create Sobel kernels Sx and Sy (float)

	2. Calculate the derivative in x and y direction
	
	dx = src o Sx
	dy = src o Sy

	Floating point image, whitout normalization
	- for visualisation imshow("dx", abs(dx)/255)

	3. calculate magnitude and angle

	mag = sqrt(dx*dx + dy*dy)
	angle = atan2(dy, dx)
	-for visualisation imshow("mag", abs(mag)/255)

	4. non-maximum supression = thining

	quantize angles q = ((int)round(angle(2pi)*8))%8
	if mag(i, j) > than neighbour its neighbour in direction q and (q+4)%8
		mag2(i,j) = mag(i,j)
	otherwise erase (mag2(i,j) = 0)

*/

void imageGradient(Mat_<uchar> src) {
	Mat_<float> Sx = (Mat_<float>(3, 3) <<
		-1.0, 0.0, 1.0,
		-2.0, 0.0, 2.0,
		-1.0, 0.0, 1.0
		);

	Mat_<float> Sy = (Mat_<float>(3, 3) <<
		1.0, 2.0, 1.0,
		0.0, 0.0, 0.0,
		-1.0, -2.0, -1.0
		);

	auto dx = conv(src, Sx);
	auto dy = conv(src, Sy);

	imshow("dx", abs(dx) / 255);
	imshow("dy", abs(dy) / 255);

	Mat_<float> mag(src.rows,src.cols);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			mag(i,j) = sqrt(dx(i, j) * dx(i, j) + dy(i, j) * dy(i, j));
		}
	}

	imshow("magnitude", abs(mag) / 255);

	Mat_<float> angle(src.rows, src.cols);
	
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			angle(i, j) = atan2(dy(i, j), dx(i, j));
			if (angle(i, j) < 0) {
				angle(i, j) = angle(i, j) + 2 * PI;
			}
		}
	}

	//imshow("angle", abs(angle) / 255);

	Mat_<float> mag2(src.rows, src.cols);

	int di[] = { 0, -1, -1, -1,  0,  1, 1, 1 };
	int dj[] = { 1,  1,  0, -1, -1, -1, 0, 1 };

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int q = ((int)round(angle(i, j) / (2 * PI) * 8)) % 8;
			int q2 = (q + 4) % 8;
			if (isInside(src.rows, src.cols, i + di[q], j + dj[q])==1 && isInside(src.rows,src.cols, i + di[q2], j + dj[q2])==1) {
				if (mag(i, j) > mag(i + di[q], j + dj[q]) && mag(i, j) > mag(i + di[q2], j + dj[q2])) {
					mag2(i, j) = mag(i, j);
				}
				else {
					mag2(i, j) = 0;
				}
			}

		}
	}
	imshow("mag2", abs(mag2) / 255);

	Mat_<uchar> magNew(src.rows, src.cols);
	
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			magNew(i, j) = mag2(i,j) / (4 * sqrt(2));
		}
	}

	int h[256] = { 0 };

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int val = (int)magNew(i,j);
			if (val >= 0) {
				h[val]++;
			}
		}
	}

	for (int i = 0; i < 255; i++) {
		printf("% d \n", h[i]);
	}
	
	float p = 0.2;

	float NoNonEdge = (1 - p) * (src.rows * src.cols - h[0]);

	int sum = 0;
	float t_high = 0.0;
	for (int i = 1; i < 255; i++) {
		sum = sum + h[i];
		if (sum > NoNonEdge) {
			t_high = i;
			break;
		}
	}

	float t_low = 0.4 * t_high;
	
	printf("NoNonEdge = %f\n", NoNonEdge);
	printf("t_high = %f\n", t_high);
	printf("t_low  = %f\n", t_low);
	Mat_<float> mag_strong_edge(src.rows, src.cols);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (magNew(i, j) > t_high) {
				mag_strong_edge(i, j) = 256;
			}
			else {
				mag_strong_edge(i, j) = 0;
			}
		}
	}
	
	Mat_<float> mag_weak_edge(src.rows, src.cols);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (magNew(i, j) > t_low && magNew(i,j) < t_high) {
				mag_weak_edge(i, j) = 256;
			}
			else {
				mag_weak_edge(i, j) = 0;
			}
		}
	}


	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (magNew(i, j) > t_high) {
				magNew(i, j) = 255;
			}
			else if (magNew(i, j) > t_low && magNew(i,j) < t_high) {
				magNew(i, j) = 128;
			}
			else {
				magNew(i, j) = 0;
			}
		}
	}

	int ok = 0;
	do {
		ok = 0;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				for (int k = 0; k < 8; k++) {
					if(isInside(src.rows, src.cols, i + di[k], j + dj[k]) == 1)
						if (magNew(i, j) == 256 && magNew(i + di[k], j + dj[k]) == 128) {
							magNew(i + di[k], j + dj[k]) = 255;
							ok = 1;
						}
				}
			}
		}
	} while (ok == 1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (magNew(i, j) == 128) {
				magNew(i, j) = 0;
			}
		}
	}

	imshow("mag_New",magNew);
	imshow("mag_weak_edge", abs(mag_weak_edge) / 255);
	imshow("mag_strong_edge", abs(mag_strong_edge) / 255);

	waitKey();

}

void wrapper12() {
	Mat_<uchar> src = imread("Images/cameraman.bmp", 0);
	imageGradient(src);
	imshow("src", src);
	waitKey();
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		//system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - Add 50 (ligher photo)\n");
		printf(" 14 - Minus 50 (darker photo)\n");
		printf(" 15 - Four Colors\n");
		printf(" 16 - Inverse of a matrix\n");
		printf(" 17 - RGB split grayscale\n");
		printf(" 18 - Black & White conversion\n");
		printf(" 19 - RGB to HSV\n");
		printf(" 20 - InImage\n");
		printf(" 21 - Gray Image Hystogram\n");
		printf(" 22 - Normalized Hystogram\n");
		printf(" 23 - Multilevel Treshold\n");
		printf(" 24 - Floyd-Steinberg\n");
		printf(" 25 - Geometrical Features\n");
		printf(" 26 - Color Objects form gray image\n");
		printf(" 27 - Two pass\n");
		printf(" 28 - Border Tracing\n");
		printf(" 29 - Reconstruct Image\n");
		printf(" 30 - Dialate Image\n");
		printf(" 31 - Mean Intensity\n");
		printf(" 32 - Standard Deviation\n");
		printf(" 33 - Cummulative Histogram\n");
		printf(" 34 - Thresholding\n");
		printf(" 35 - Hystogram streaching\n");
		printf(" 38 - Gaussian and Ideal high/low filtering\n");
		printf(" 39 - 2 images on top of one another\n");
		printf(" 40 - Noise removal Median Filter\n");
		printf(" 41 - Gauss 2D \n");
		printf(" 42 - Gauss 1D \n");
		printf(" 43 - Image Gradient \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testNegativeImage();
			break;
		case 4:
			testNegativeImageFast();
			break;
		case 5:
			testColor2Gray();
			break;
		case 6:
			testImageOpenAndSave();
			break;
		case 7:
			testBGR2HSV();
			break;
		case 8:
			testResize();
			break;
		case 9:
			testCanny();
			break;
		case 10:
			testVideoSequence();
			break;
		case 11:
			testSnap();
			break;
		case 12:
			testMouseClick();
			break;
		case 13:
			add50();
			break;
		case 14:
			minus50();
			break;
		case 15:
			fourColors();
			break;
		case 16:
			inverseOfaMatrix();
			break;
		case 17:
			rgbSplitGrayscale();
			break;
		case 18:
			blackWhiteConversion();
			break;
		case 19:
			conversionRGB_HCV();
			break;
		case 20:
			char fname[MAX_PATH];
			while (openFileDlg(fname)) {
				Mat src = imread(fname, IMREAD_COLOR);
				std::cout << inImage(src, 10, 10);
			}
		case 21:
			hystogramComputation();
			break;
		case 22:
			normalizedHystogram();
			break;
		case 23:
			multiLevelTresholding(5, 0.0003);
			break;
		case 24:
			floydSteinberg(5, 0.0003);
			break;
		case 25:
			geometricalFeatures();
			break;
		case 26:
			objectsGraytoColor();
			break;
		case 27:
			two_pass_Component_Labeling();
			break;
		case 28:
			lab6_border();
			break;
		case 29:
			lab6_reconstruct();
			break;
		case 30:
			wrapper();
			break;
		case 31:
			wrapper1();
			break;
		case 32:
			wrapper2();
			break;
		case 33:
			wrapper3();
			break;
		case 34:
			wrapper4();
			break;
		case 35:
			wrapper5();
			break;
		case 36:
			wrapper6();
			break;
		case 37:
			wrapper7();
			break;
		case 38:
			wrapper8();
			break;
		case 39:
			suprapunere_imagini();
			break;
		case 40:
			wrapper9();
			break;
		case 41:
			wrapper10();
			break;
		case 42:
			wrapper11();
			break;
		case 43:
			wrapper12();
			break;
		}
	} while (op != 0);
	return 0;
}