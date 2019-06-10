// gsblur.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<iostream>
#include<cmath>
#include<cstdlib>
#include <iomanip>
#include<opencv.hpp>
using namespace std;
using namespace cv;

#define pi 3.1415926

void gskernel(int si ze, double sigma, Mat srcimage,Mat dstimage)

{
	//开辟动态数组，存储数据
	double **a = new double*[size];            //分配一个指针数组，将其首地址保存在a中   、
	for (int m = 0; m <  size; m++)             //为指针数组的每个元素分配一个数组
		a[m] = new double[size];

	int k = (size - 1) / 2;
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			double sigma2 = pow(sigma, 2);
			double A = 1 / (2 * pi * sigma2);   //2pi的sigma方分之一
			double C = pow((i - k), 2) + pow((j - k), 2);
			double B = exp(-(C / (2 * sigma2)));

			a[i][j] = A*B;
			//cout << a[i][j] << "    ";
		}
		    //cout << endl;
	}
	cout << endl << "对应高斯内核函数为：" << endl << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			cout << setw(3) << a[i][j] << "     ";
		}
		cout << endl;
	}
	double guiyi = 1/a[0][0];
	//cout << guiyi << endl;

	cout << endl << "对应整数高斯内核函数为：" << endl << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			a[i][j] *= guiyi;
			a[i][j]=floor(a[i][j]);
			sum += a[i][j];
			cout << setw(3) << a[i][j] << "     ";
		}
		cout << endl;
	}
	cout << endl << "和为：" <<sum<< endl;


	//开始对图像进行操作
	if (srcimage.empty())
	{
		cout << "请检查图片！" << endl;
		return;
	}


	for (int i = 0; i < srcimage.rows; i++)
	{
 		for (int j = 0; j < srcimage.cols; j++)
		{
			//边缘判断
			if (  (i-k) >=0 && (i+k)<srcimage.rows &&  (j - k) >=0 && (j + k) < srcimage.cols)
			{
				     //cout << "i=" << i << "  " << "j=" << j << endl;
				//从(i-k，j-k)到(i+k,j+k)是对应内核的所有元素   

				double juanji = 0;
				double jingdu = 0;
				for (int m = (-k);m <= k;m++)
				{
					for (int n = (-k);n <= k;n++)
					{
						//cout << "m=" << m << "  " << "n=" << n << endl;
						juanji = juanji + (srcimage.at<uchar>(i+m, j+n) * a[k + m][k + n]);
		  //////////此处是动态的高斯内核数组和相图像对应的像素值数组相乘，同一个m，同一个n即可解决。
						//cout << juanji << endl;
					}
				}

				jingdu = juanji / sum;
				//cout << jingdu << endl;
				dstimage.at<uchar>(i, j) = round(jingdu);

			}
			else
			{
				dstimage.at<uchar>(i,j) = srcimage.at<uchar>(i, j);
			}
		}

	}

	
	for (int q = 0; q < size; q++)
		delete[] a[q];
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
	} while (u1 <= epsilon);     //当U1小于等于极小值时候舍弃，再循环一次取新值


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
	if (image.empty())
	{
		cout << "读入图片错误！" << endl;
		system("pause");
		return -1;
	}

	double mu, sig, l;
	cout << "请输入高斯噪声中mu,sig,l的值" << endl;
	cin >> mu >> sig >> l;
	Mat gsimage = addGaussianNoise(image, mu, sig, l);

	
	int size;
	double sigma;
	Mat srcimage= gsimage.clone();
	Mat dstimage= image.clone();
	cout << "请输入内核大小和sigma的值：" << endl;
	cin >> size >> sigma;
	gskernel(size, sigma,srcimage,dstimage);


	imshow("原图像", image);
    imshow("加入高斯噪声后的图像", gsimage);
	imshow("高斯滤波后的图像", dstimage);

	imwrite("F:\\song\\imwirte\\高斯噪声(0,0.8,16).jpg", gsimage);
	imwrite("F:\\song\\imwirte\\高斯滤波滤除高斯噪声(0,0.8,16;7,1.4).jpg", dstimage);
	waitKey();
	system("pause");
	return 0;
}

