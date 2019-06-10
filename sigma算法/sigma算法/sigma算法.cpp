// 自写sigma平滑滤波.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<iostream>
#include<cstdlib>
#include<opencv.hpp>
#include<cmath>

using namespace std;
using namespace cv;

//构造sigma平滑函数
void sigma(Mat src, Mat dst)

{
	if (src.empty())
	{
		cout << "传图失败，请检查!" << endl;
		return;
	}

	for (int i = 0; i < src.rows; i++)
	{

		for (int j = 0; j < src.cols; j++)
		{

			//对非边缘范围进行处理
			if (i - 1 >= 1 && i + 2 < src.rows && j - 1 >= 1 && j + 2 < src.cols)
			{
				//定义数组
				double a[25] =
				{
					src.at<uchar>(i,j),src.at<uchar>(i,j - 1),src.at<uchar>(i,j - 2),src.at<uchar>(i,j + 1),src.at<uchar>(i,j + 2),
					src.at<uchar>(i - 1,j),src.at<uchar>(i - 1,j - 1),src.at<uchar>(i - 1,j - 2),src.at<uchar>(i - 1,j + 1),src.at<uchar>(i - 1,j + 2),
					src.at<uchar>(i - 2,j),src.at<uchar>(i - 2,j - 1),src.at<uchar>(i - 2,j - 2),src.at<uchar>(i - 2,j + 1),src.at<uchar>(i - 2,j + 2),
					src.at<uchar>(i + 1,j),src.at<uchar>(i + 1,j - 1),src.at<uchar>(i + 1,j - 2),src.at<uchar>(i + 1,j + 1),src.at<uchar>(i + 1,j + 2),
					src.at<uchar>(i + 2,j),src.at<uchar>(i + 2,j - 1),src.at<uchar>(i + 2,j - 2),src.at<uchar>(i + 2,j + 1),src.at<uchar>(i + 2,j + 2)
				};


				//求和，平均值
				double sum1 = 0;
				double X = 0;
				for (int k = 0; k < 25; k++)
					sum1 = sum1 + a[k];             //求和

				X = sum1 / 25;               //求平均数

					//求标准差
				double bzc = 0;
				double sum2 = 0;
				for (int k = 0; k < 25; k++)
					sum2 = sum2 + pow(a[k] - X, 2);

				bzc = sqrt(sum2 / 25);

				//求出置信区间
				float max = a[0] + (2 * bzc);
				float min = a[0] - (2 * bzc);

				//判断元素是否在置信区间内，如果是，累加求和，并计算最终数量，求得最终替换的均值
				double sum3 = 0;
				int number = 0;
				for (int k = 0; k < 25; k++)
				{
					if (a[k] <= max && a[k] >= min)
					{
						sum3 = sum3 + a[k];
						number++;
					}
				}
				double jingdu = sum3 / number;
				dst.at<uchar>(i, j) = round(jingdu);

			}

			//边缘不进行处理
			else
				dst.at<uchar>(i, j) = src.at<uchar>(i, j);
		}

	}
}

//构造高斯分布，用BoxMuller
double generateGaussian(double mu, double sig)
{

	//定义极小值
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	double u1, u2;

	//构造随机变量
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);	    //取【0-1】范围内的均匀分布U1
		u2 = rand() * (1.0 / RAND_MAX);     //取【0-1】范围内的均匀分布U2
	} 
	while (u1 <= epsilon);     //当U1小于等于极小值时候舍弃，再进入do重新循环一次取新值，否则一次即可；


	//构造随机变量Z0，可以使两均匀分布为基础，经过变换使服从标准高斯分布,其中均值为0，方差为1(标准);
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);

	/*正态值 Z0 是服从标准正态分布的均值=0，标准差=1；
	怎么用标准正态分布（均值0，标准差1）生成一个自定义的非标准正态分布呢？(均值mu，标准差sigma）
	可使用以下等式将Z0标准正态分布变换为一个均值为 mu、标准差为 sigma 的自定义正态分布;    */

	return z0*sig + mu;           //标准分布（均值0，标准差1） 乘上sigma（决定分布高度，上下移动），加上mu（决定分布中心，左右移动）
}

//为图像加入高斯噪声
Mat addGaussianNoise(Mat srcImag, double mu, double sig, double l)
{
	Mat dstImage = srcImag.clone();
	for (int i = 0; i < dstImage.rows; i++)
	{
		for (int j = 0; j < dstImage.cols; j++)
		{
			//加入高斯噪声原理：随机产生符合高斯分布的随机数，然后将该值乘上系数l再和图像原有的像素值相加
			int val = dstImage.at<uchar>(i, j) + generateGaussian(mu, sig) * l;   //l越大，乘系数越多，偏离原数值越多，变化越明显
																				  //最后将得到的和压缩到[0,255]区间内，得到新的像素值
			if (val < 0)
				val = 0;
			if (val>255)
				val = 255;
			dstImage.at<uchar>(i, j) = (uchar)val;
		}
	}
	return dstImage;
}


//椒盐函数
void salt(Mat saltsrc, int num)
{
	//检测是否传入空图
	if (saltsrc.empty())
	{
		cout << "传图失败，请检查！" << endl;
		return;
	}

	int i, j;
	srand(time(0));

	//盐
	for (int x = 0; x < num; x++)
	{
		i = rand() % saltsrc.rows;
		j = rand() % saltsrc.cols;
		saltsrc.at<uchar>(i, j) = 255;

	}

	//椒
	for (int x = 0; x < num; x++)
	{
		i = rand() % saltsrc.rows;
		j = rand() % saltsrc.cols;
		saltsrc.at<uchar>(i, j) = 0;

	}

}


int main()
{
	Mat image = imread("D:\\marks.jpg", 0);    //读图

	int number;
	cout << "请输入椒盐噪声中噪声数的值:" << endl;
	cin >> number;
	Mat saltimage = image.clone();
	salt(saltimage, number);

	double mu, sig, l;
	cout << "请输入高斯噪声中mu,sig,l的值" << endl;
	cin >> mu >> sig >> l;
	Mat gsimage = addGaussianNoise(image, mu, sig, l);

	Mat image1 = saltimage.clone();
	Mat image2 = gsimage.clone();

	sigma(saltimage, image1);
	sigma(gsimage, image2);

	imshow("原图", image);
	imshow("椒盐噪声", saltimage);
	imshow("高斯噪声", gsimage);
	imshow("sigma平滑椒盐噪声", image1);
	imshow("sigma平滑高斯噪声", image2);
	waitKey();
	return 0;
}

