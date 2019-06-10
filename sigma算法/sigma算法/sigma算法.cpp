// ��дsigmaƽ���˲�.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include<iostream>
#include<cstdlib>
#include<opencv.hpp>
#include<cmath>

using namespace std;
using namespace cv;

//����sigmaƽ������
void sigma(Mat src, Mat dst)

{
	if (src.empty())
	{
		cout << "��ͼʧ�ܣ�����!" << endl;
		return;
	}

	for (int i = 0; i < src.rows; i++)
	{

		for (int j = 0; j < src.cols; j++)
		{

			//�ԷǱ�Ե��Χ���д���
			if (i - 1 >= 1 && i + 2 < src.rows && j - 1 >= 1 && j + 2 < src.cols)
			{
				//��������
				double a[25] =
				{
					src.at<uchar>(i,j),src.at<uchar>(i,j - 1),src.at<uchar>(i,j - 2),src.at<uchar>(i,j + 1),src.at<uchar>(i,j + 2),
					src.at<uchar>(i - 1,j),src.at<uchar>(i - 1,j - 1),src.at<uchar>(i - 1,j - 2),src.at<uchar>(i - 1,j + 1),src.at<uchar>(i - 1,j + 2),
					src.at<uchar>(i - 2,j),src.at<uchar>(i - 2,j - 1),src.at<uchar>(i - 2,j - 2),src.at<uchar>(i - 2,j + 1),src.at<uchar>(i - 2,j + 2),
					src.at<uchar>(i + 1,j),src.at<uchar>(i + 1,j - 1),src.at<uchar>(i + 1,j - 2),src.at<uchar>(i + 1,j + 1),src.at<uchar>(i + 1,j + 2),
					src.at<uchar>(i + 2,j),src.at<uchar>(i + 2,j - 1),src.at<uchar>(i + 2,j - 2),src.at<uchar>(i + 2,j + 1),src.at<uchar>(i + 2,j + 2)
				};


				//��ͣ�ƽ��ֵ
				double sum1 = 0;
				double X = 0;
				for (int k = 0; k < 25; k++)
					sum1 = sum1 + a[k];             //���

				X = sum1 / 25;               //��ƽ����

					//���׼��
				double bzc = 0;
				double sum2 = 0;
				for (int k = 0; k < 25; k++)
					sum2 = sum2 + pow(a[k] - X, 2);

				bzc = sqrt(sum2 / 25);

				//�����������
				float max = a[0] + (2 * bzc);
				float min = a[0] - (2 * bzc);

				//�ж�Ԫ���Ƿ������������ڣ�����ǣ��ۼ���ͣ�������������������������滻�ľ�ֵ
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

			//��Ե�����д���
			else
				dst.at<uchar>(i, j) = src.at<uchar>(i, j);
		}

	}
}

//�����˹�ֲ�����BoxMuller
double generateGaussian(double mu, double sig)
{

	//���弫Сֵ
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	double u1, u2;

	//�����������
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);	    //ȡ��0-1����Χ�ڵľ��ȷֲ�U1
		u2 = rand() * (1.0 / RAND_MAX);     //ȡ��0-1����Χ�ڵľ��ȷֲ�U2
	} 
	while (u1 <= epsilon);     //��U1С�ڵ��ڼ�Сֵʱ���������ٽ���do����ѭ��һ��ȡ��ֵ������һ�μ��ɣ�


	//�����������Z0������ʹ�����ȷֲ�Ϊ�����������任ʹ���ӱ�׼��˹�ֲ�,���о�ֵΪ0������Ϊ1(��׼);
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);

	/*��ֵ̬ Z0 �Ƿ��ӱ�׼��̬�ֲ��ľ�ֵ=0����׼��=1��
	��ô�ñ�׼��̬�ֲ�����ֵ0����׼��1������һ���Զ���ķǱ�׼��̬�ֲ��أ�(��ֵmu����׼��sigma��
	��ʹ�����µ�ʽ��Z0��׼��̬�ֲ��任Ϊһ����ֵΪ mu����׼��Ϊ sigma ���Զ�����̬�ֲ�;    */

	return z0*sig + mu;           //��׼�ֲ�����ֵ0����׼��1�� ����sigma�������ֲ��߶ȣ������ƶ���������mu�������ֲ����ģ������ƶ���
}

//Ϊͼ������˹����
Mat addGaussianNoise(Mat srcImag, double mu, double sig, double l)
{
	Mat dstImage = srcImag.clone();
	for (int i = 0; i < dstImage.rows; i++)
	{
		for (int j = 0; j < dstImage.cols; j++)
		{
			//�����˹����ԭ������������ϸ�˹�ֲ����������Ȼ�󽫸�ֵ����ϵ��l�ٺ�ͼ��ԭ�е�����ֵ���
			int val = dstImage.at<uchar>(i, j) + generateGaussian(mu, sig) * l;   //lԽ�󣬳�ϵ��Խ�࣬ƫ��ԭ��ֵԽ�࣬�仯Խ����
																				  //��󽫵õ��ĺ�ѹ����[0,255]�����ڣ��õ��µ�����ֵ
			if (val < 0)
				val = 0;
			if (val>255)
				val = 255;
			dstImage.at<uchar>(i, j) = (uchar)val;
		}
	}
	return dstImage;
}


//���κ���
void salt(Mat saltsrc, int num)
{
	//����Ƿ����ͼ
	if (saltsrc.empty())
	{
		cout << "��ͼʧ�ܣ����飡" << endl;
		return;
	}

	int i, j;
	srand(time(0));

	//��
	for (int x = 0; x < num; x++)
	{
		i = rand() % saltsrc.rows;
		j = rand() % saltsrc.cols;
		saltsrc.at<uchar>(i, j) = 255;

	}

	//��
	for (int x = 0; x < num; x++)
	{
		i = rand() % saltsrc.rows;
		j = rand() % saltsrc.cols;
		saltsrc.at<uchar>(i, j) = 0;

	}

}


int main()
{
	Mat image = imread("D:\\marks.jpg", 0);    //��ͼ

	int number;
	cout << "�����뽷����������������ֵ:" << endl;
	cin >> number;
	Mat saltimage = image.clone();
	salt(saltimage, number);

	double mu, sig, l;
	cout << "�������˹������mu,sig,l��ֵ" << endl;
	cin >> mu >> sig >> l;
	Mat gsimage = addGaussianNoise(image, mu, sig, l);

	Mat image1 = saltimage.clone();
	Mat image2 = gsimage.clone();

	sigma(saltimage, image1);
	sigma(gsimage, image2);

	imshow("ԭͼ", image);
	imshow("��������", saltimage);
	imshow("��˹����", gsimage);
	imshow("sigmaƽ����������", image1);
	imshow("sigmaƽ����˹����", image2);
	waitKey();
	return 0;
}

