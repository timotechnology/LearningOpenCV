// 自写KNN平滑滤波.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<iostream>
#include<cstdlib>
#include<opencv.hpp>

using namespace std;
using namespace cv;

void KNN(Mat src, Mat dst)
{
	if (src.empty())
	{
		cout << "传图失败，请检查！" << endl;
		return;

	}

	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; j++)
		{

			//非边缘部分进行KNN处理
			if (i - 1 >= 0 && i + 1 < src.rows && j - 1 >= 0 && j + 1 < src.cols)
			{
				int a[9] = { src.at<uchar>(i,j),
					src.at<uchar>(i - 1,j - 1),src.at<uchar>(i - 1,j),src.at<uchar>(i - 1,j + 1),
					src.at<uchar>(i,j - 1),src.at<uchar>(i,j + 1),
					src.at<uchar>(i + 1,j - 1), src.at<uchar>(i + 1,j),src.at<uchar>(i + 1,j + 1) };


				//对距离中间像素的绝对值远近进行冒泡排序
				int s = a[0];                       //定义中值
				for (int m = 0; m < 8; m++)         //冒泡排序外循环,循环N-1次；外循环控制"整体的"比较执行的次数
				{
					for (int n = 0; n < 8 - m; n++)   //冒泡排序内循环，循环N-1-p次;内循环控制"内部的"每次多少个数之间比较
					{
						if (abs(a[n] - s) > abs(a[n + 1] - s))    //如果前面的大于后面的，交换。即按绝对差值从小到大排
						{
							int change = a[n];
							a[n] = a[n + 1];
							a[n + 1] = change;
						}
					}
				}
				//冒泡排序结束，现在数组a[9]按绝对差值排序好了；
				double jingdu = (a[0] + a[1] + a[2] + a[3] + a[4]) / 5;  //取前五个，求平均值
				dst.at<uchar>(i, j) = round(jingdu);                   //四舍五入取精度值

			}

			//边缘不进行处理
			else
				dst.at<uchar>(i, j) = src.at<uchar>(i, j);

		}
	}

}

//构造高斯分布，用BoxMuller
double generateGaussian(double mu, double sigma)
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
	} while (u1 <= epsilon);     //当U1小于等于极小值时候舍弃，再循环一次取新值


								 //构造随机变量Z0，可以使两均匀分布为基础，经过变换使服从标准高斯分布,其中均值为0，方差为1(标准);
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);

	/*正态值 Z0 是服从标准正态分布的均值=0，标准差=1；
	怎么用标准正态分布（均值0，标准差1）生成一个自定义的非标准正态分布呢？(均值mu，标准差sigma）
	可使用以下等式将Z0标准正态分布变换为一个均值为 mu、标准差为 sigma 的自定义正态分布;    */

	return z0*sigma + mu;           //标准分布（均值0，标准差1） 乘上sigma（决定分布高度，上下移动），加上mu（决定分布中心，左右移动）
}

//为图像加入高斯噪声
Mat addGaussianNoise(Mat srcImag, double mu, double sigma, double l)
{
	Mat dstImage = srcImag.clone();
	for (int i = 0; i < dstImage.rows; i++)
	{
		for (int j = 0; j < dstImage.cols; j++)
		{
			//加入高斯噪声原理：随机产生符合高斯分布的随机数，然后将该值乘上系数l再和图像原有的像素值相加
			int val = dstImage.at<uchar>(i, j) + generateGaussian(mu, sigma) * l;   //l越大，乘系数越多，偏离原数值越多，变化越明显
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
	cout << "请输入椒盐噪声中噪声数：" << endl;
	cin >> number;
	Mat saltimage = image.clone();
	salt(saltimage, number);

	double mu, sigma, l;
	cout << "请输入mu,sigma,l的值" << endl;
	cin >> mu >> sigma >> l;
	Mat gsimage = addGaussianNoise(image, mu, sigma, l);


	Mat image1 = saltimage.clone();
	Mat image2 = gsimage.clone();
	KNN(saltimage, image1);
	KNN(gsimage, image2);
	for (int i = 0;i < 15;i++) 
	{
		KNN(image1, image1);
		KNN(image2, image2);


	}
	imshow("原图像", image);
	imshow("椒盐噪声", saltimage);
	imshow("高斯噪声", gsimage);
	imshow("KNN平滑椒盐噪声", image1);
	imshow("KNN平滑高斯噪声", image2);

	waitKey();  
	return 0;
}



