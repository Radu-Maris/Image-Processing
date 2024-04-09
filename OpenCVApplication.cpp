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
		}
	} while (op != 0);
	return 0;
}