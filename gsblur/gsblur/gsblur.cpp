// gsblur.cpp : �������̨Ӧ�ó������ڵ㡣
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
	//���ٶ�̬���飬�洢����
	double **a = new double*[size];            //����һ��ָ�����飬�����׵�ַ������a��   ��
	for (int m = 0; m <  size; m++)             //Ϊָ�������ÿ��Ԫ�ط���һ������
		a[m] = new double[size];

	int k = (size - 1) / 2;
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			double sigma2 = pow(sigma, 2);
			double A = 1 / (2 * pi * sigma2);   //2pi��sigma����֮һ
			double C = pow((i - k), 2) + pow((j - k), 2);
			double B = exp(-(C / (2 * sigma2)));

			a[i][j] = A*B;
			//cout << a[i][j] << "    ";
		}
		    //cout << endl;
	}
	cout << endl << "��Ӧ��˹�ں˺���Ϊ��" << endl << endl;
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

	cout << endl << "��Ӧ������˹�ں˺���Ϊ��" << endl << endl;
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
	cout << endl << "��Ϊ��" <<sum<< endl;


	//��ʼ��ͼ����в���
	if (srcimage.empty())
	{
		cout << "����ͼƬ��" << endl;
		return;
	}


	for (int i = 0; i < srcimage.rows; i++)
	{
 		for (int j = 0; j < srcimage.cols; j++)
		{
			//��Ե�ж�
			if (  (i-k) >=0 && (i+k)<srcimage.rows &&  (j - k) >=0 && (j + k) < srcimage.cols)
			{
				     //cout << "i=" << i << "  " << "j=" << j << endl;
				//��(i-k��j-k)��(i+k,j+k)�Ƕ�Ӧ�ں˵�����Ԫ��   

				double juanji = 0;
				double jingdu = 0;
				for (int m = (-k);m <= k;m++)
				{
					for (int n = (-k);n <= k;n++)
					{
						//cout << "m=" << m << "  " << "n=" << n << endl;
						juanji = juanji + (srcimage.at<uchar>(i+m, j+n) * a[k + m][k + n]);
		  //////////�˴��Ƕ�̬�ĸ�˹�ں��������ͼ���Ӧ������ֵ������ˣ�ͬһ��m��ͬһ��n���ɽ����
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
	} while (u1 <= epsilon);     //��U1С�ڵ��ڼ�Сֵʱ����������ѭ��һ��ȡ��ֵ


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
	if (image.empty())
	{
		cout << "����ͼƬ����" << endl;
		system("pause");
		return -1;
	}

	double mu, sig, l;
	cout << "�������˹������mu,sig,l��ֵ" << endl;
	cin >> mu >> sig >> l;
	Mat gsimage = addGaussianNoise(image, mu, sig, l);

	
	int size;
	double sigma;
	Mat srcimage= gsimage.clone();
	Mat dstimage= image.clone();
	cout << "�������ں˴�С��sigma��ֵ��" << endl;
	cin >> size >> sigma;
	gskernel(size, sigma,srcimage,dstimage);


	imshow("ԭͼ��", image);
    imshow("�����˹�������ͼ��", gsimage);
	imshow("��˹�˲����ͼ��", dstimage);

	imwrite("F:\\song\\imwirte\\��˹����(0,0.8,16).jpg", gsimage);
	imwrite("F:\\song\\imwirte\\��˹�˲��˳���˹����(0,0.8,16;7,1.4).jpg", dstimage);
	waitKey();
	system("pause");
	return 0;
}

