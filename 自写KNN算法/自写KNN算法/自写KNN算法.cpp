// ��дKNNƽ���˲�.cpp : �������̨Ӧ�ó������ڵ㡣
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
		cout << "��ͼʧ�ܣ����飡" << endl;
		return;

	}

	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; j++)
		{

			//�Ǳ�Ե���ֽ���KNN����
			if (i - 1 >= 0 && i + 1 < src.rows && j - 1 >= 0 && j + 1 < src.cols)
			{
				int a[9] = { src.at<uchar>(i,j),
					src.at<uchar>(i - 1,j - 1),src.at<uchar>(i - 1,j),src.at<uchar>(i - 1,j + 1),
					src.at<uchar>(i,j - 1),src.at<uchar>(i,j + 1),
					src.at<uchar>(i + 1,j - 1), src.at<uchar>(i + 1,j),src.at<uchar>(i + 1,j + 1) };


				//�Ծ����м����صľ���ֵԶ������ð������
				int s = a[0];                       //������ֵ
				for (int m = 0; m < 8; m++)         //ð��������ѭ��,ѭ��N-1�Σ���ѭ������"�����"�Ƚ�ִ�еĴ���
				{
					for (int n = 0; n < 8 - m; n++)   //ð��������ѭ����ѭ��N-1-p��;��ѭ������"�ڲ���"ÿ�ζ��ٸ���֮��Ƚ�
					{
						if (abs(a[n] - s) > abs(a[n + 1] - s))    //���ǰ��Ĵ��ں���ģ��������������Բ�ֵ��С������
						{
							int change = a[n];
							a[n] = a[n + 1];
							a[n + 1] = change;
						}
					}
				}
				//ð�������������������a[9]�����Բ�ֵ������ˣ�
				double jingdu = (a[0] + a[1] + a[2] + a[3] + a[4]) / 5;  //ȡǰ�������ƽ��ֵ
				dst.at<uchar>(i, j) = round(jingdu);                   //��������ȡ����ֵ

			}

			//��Ե�����д���
			else
				dst.at<uchar>(i, j) = src.at<uchar>(i, j);

		}
	}

}

//�����˹�ֲ�����BoxMuller
double generateGaussian(double mu, double sigma)
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
	} while (u1 <= epsilon);     //��U1С�ڵ��ڼ�Сֵʱ����������ѭ��һ��ȡ��ֵ


								 //�����������Z0������ʹ�����ȷֲ�Ϊ�����������任ʹ���ӱ�׼��˹�ֲ�,���о�ֵΪ0������Ϊ1(��׼);
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);

	/*��ֵ̬ Z0 �Ƿ��ӱ�׼��̬�ֲ��ľ�ֵ=0����׼��=1��
	��ô�ñ�׼��̬�ֲ�����ֵ0����׼��1������һ���Զ���ķǱ�׼��̬�ֲ��أ�(��ֵmu����׼��sigma��
	��ʹ�����µ�ʽ��Z0��׼��̬�ֲ��任Ϊһ����ֵΪ mu����׼��Ϊ sigma ���Զ�����̬�ֲ�;    */

	return z0*sigma + mu;           //��׼�ֲ�����ֵ0����׼��1�� ����sigma�������ֲ��߶ȣ������ƶ���������mu�������ֲ����ģ������ƶ���
}

//Ϊͼ������˹����
Mat addGaussianNoise(Mat srcImag, double mu, double sigma, double l)
{
	Mat dstImage = srcImag.clone();
	for (int i = 0; i < dstImage.rows; i++)
	{
		for (int j = 0; j < dstImage.cols; j++)
		{
			//�����˹����ԭ������������ϸ�˹�ֲ����������Ȼ�󽫸�ֵ����ϵ��l�ٺ�ͼ��ԭ�е�����ֵ���
			int val = dstImage.at<uchar>(i, j) + generateGaussian(mu, sigma) * l;   //lԽ�󣬳�ϵ��Խ�࣬ƫ��ԭ��ֵԽ�࣬�仯Խ����
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
	cout << "�����뽷����������������" << endl;
	cin >> number;
	Mat saltimage = image.clone();
	salt(saltimage, number);

	double mu, sigma, l;
	cout << "������mu,sigma,l��ֵ" << endl;
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
	imshow("ԭͼ��", image);
	imshow("��������", saltimage);
	imshow("��˹����", gsimage);
	imshow("KNNƽ����������", image1);
	imshow("KNNƽ����˹����", image2);

	waitKey();  
	return 0;
}



