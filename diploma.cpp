#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <complex>
#include <iostream>
#include<fstream>
# define M_PI           3.14159265358979323846
using namespace std;
// Объявление функций. //

// Упрощённое динмическое уравнение матрицы плотности. //
complex < double >** easyF(double t, complex < double >** Y, double w, double omega, double Delta, double Omega, int M, std::complex< double >** dYdt);

// Стандартное динмическое уравнение матрицы плотности. //
complex < double >** F(double t, complex < double >** Y, double w, double omega, double Delta, double Omega, int M, std::complex< double >** dYdt);

// Преобразование Фурье. //
complex <double>* Fourier(double* t, complex < double >* Y, double* f, int reso, int size, int per);

// Мнимая единица //
complex< double > imagOne = { 0.0, 1.0 };

// Функция нахождения искажений мнимой части спектра решения. //
double imagMistake(complex <double>* comp, complex <double>* easyComp, int reso);

// Функция нахождения искажений вещественной части спектра решения. //
double realMistake(complex <double>* comp, complex <double>* easyComp, int reso);

// Функция нахождения низкочастотной части искажений вещественной части спектра решения. //
double LowRealMistake(complex <double>* comp, complex <double>* easyComp, int subSize, double w, double Delta, double omega);

// Функция нахождения высокочастотной части искажений вещественной части спектра решения. //
double HighRealMistake(complex <double>* comp, complex <double>* easyComp, int subSize, double w, double Delta, double omega);

// Функция нахождения низкочастотной искажений мнимой части спектра решения. //
double LowImagMistake(complex <double>* comp, complex <double>* easyComp, int subSize, double w, double Delta, double omega);

// Функция нахождения высокочастотной части искажений мнимой части спектра решения. //
double HighImagMistake(complex <double>* comp, complex <double>* easyComp, int subSize, double w, double Delta, double omega);

int main()
{
	cout.setf(ios::fixed);  // вывод в фиксированном формате 
	cout.precision(20);
	//Начальные условия.
	double Delta = 0.3;
	double Omega = 0.2;
	int M = 4;
	double omega = 0.0 * Delta;								
	double wStart = 0.0 * Delta;

	// Ввод числа точек на период, количества периодов и масштаба спектров. //
	int per;
	int num;
	int reso, points, step;
	cout << " points per period? : " << endl;
	cin >> per;
	cout << " num of periods? : " << endl;
	cin >> num;												
	cout << " resolution of spectra (in Deltas)? : " << endl;
	cin >> reso;
	cout << " num of points? : " << endl;
	cin >> points;
	cout << " and step (in Deltas)? : " << endl;
	cin >> step;

	// Определение полных размеров массивов. //
	int size = per * num;
	int subSize = reso * 2 + 1;

	// Задание сетки времени. //
	double ts = 2.0 * M_PI / Delta;
	double tim;
	double* t = new double[size];
	t[0] = ts / (double)per;
	for (int i = 0; i < size - 1; i++)
		t[i + 1] = t[i] + ts / (double)per;

	// Задание сетки частот (для перобразования Фурье). //
	double* f = new double[subSize];
	double fs = 1 / ts;
	f[0] = -1.0 * fs * reso;
	for (int i = 1; i < subSize; i++) {
		f[i] = f[i - 1] + fs;
	}

	// Динамическое выделение памяти для массивов получаемых значений искажений. //
	double* realErr = new double[int(points/step)];
	double* imagErr = new double[int(points / step)];

	double* lowRealErr = new double[int(points / step)];
	double* highRealErr = new double[int(points / step)];
	double* lowImagErr = new double[int(points / step)];
	double* highImagErr = new double[int(points / step)];

	// Динамическое выделение памяти для массива получаемых значений w. //
	double* xAxis = new double[int(points / step)];

	// Шаг сетки времени. //
	double h;

	// Задание коэффициентов метода Рунге-Кутты 4-го порядка по правилу 3/8. //
	complex < double > k1[2][2], k2[2][2], k3[2][2], k4[2][2] = { {{0.0,0.0},{0.0,0.0}} ,{{0.0,0.0},{0.0,0.0}} };
	double c[4] = { 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 };
	double b[4] = { 1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0 };
	double a[4][4] = { {0.0, 0.0, 0.0, 0.0},{1.0 / 3.0, 0.0, 0.0, 0.0},{-1.0 / 3.0, 1.0, 0.0, 0.0},{1.0, -1.0, 1.0, 0.0} };

	// Задание количества используемых потоков (openmp). //
	omp_set_num_threads(12);
	tim = omp_get_wtime();
	double w;
	int detune;
	// Распараллеливание цикла for в котором реализован метод Рунге-Кутта, преобразование Фурье и вычисление ошибки. //
#pragma omp parallel for private (detune, w, k1, k2, k3, k4, omega) shared (lowRealErr, lowImagErr, highRealErr, highImagErr, realErr, imagErr, xAxis, size, subSize, per, num, reso, ts, t, a, b, c, points, step)
	for (detune = 1; detune < int(points / step) + 1; detune++) {
		w = wStart + Delta * (double)detune * (double)step;
		omega = w;
		
		// Динамическое выделение памяти и заполнение матрицы начального условия метода Рунге-Кутты. //
		double** rho0 = new double* [2];
		for (int i = 0; i < 2; i++)
			rho0[i] = new double[2];
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				if (i == j)
					rho0[i][j] = 1.0;
				else
					rho0[i][j] = 0.0;
			}
		}

		// Динамическое выделение памяти под трехмерный массив решений стандартного уравнения матрицы плотности для метода Рунге-Кутты и его заполнение. //
		complex < double >*** Y = new complex < double > **[2];
		for (int i = 0; i < 2; i++) {
			Y[i] = new complex < double > *[2];
			for (int j = 0; j < 2; j++) {
				Y[i][j] = new complex < double >[size];
			}
		}

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				Y[i][j][0] = rho0[i][j];
				for (int k = 1; k < size; k++) {
					Y[i][j][k] = { 0.0, 0.0 };
				}
			}
		}

		// Динамическое выделение памяти под трехмерный массив решений упрощённого уравнения матрицы плотности для метода Рунге-Кутты и его заполнение. //
		complex < double >*** easyY = new complex < double > **[2];
		for (int i = 0; i < 2; i++) {
			easyY[i] = new complex < double > *[2];
			for (int j = 0; j < 2; j++) {
				easyY[i][j] = new complex < double >[size];
			}
		}

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				easyY[i][j][0] = rho0[i][j];
				for (int k = 1; k < size; k++) {
					easyY[i][j][k] = { 0.0, 0.0 };
				}
			}
		}

		// Динамическое выделение памяти под двумерный массив производной матрицы плотнсти для метода Рунге-Кутты. //
		complex < double >** dYdt = new complex < double > *[2];
		for (int i = 0; i < 2; i++)
			dYdt[i] = new complex < double >[2];

		// Динамическое выделение памяти под двумерный массив для итеративной перезаписи коэффициентов для метода Рунге-Кутты и его заполнение. //
		complex < double >** plus = new complex < double >*[2];
		for (int i = 0; i < 2; i++)
			plus[i] = new complex <double>[2];
		for (int l = 0; l < 2; l++) {
			for (int m = 0; m < 2; m++) {
				plus[l][m] = { 0.0, 0.0 };
			}
		}

		// Динамическое выделение памяти для получаемых решений и их заполнение. //
		complex < double >* easyData = new complex < double >[size];
		complex < double >* data = new complex < double >[size];

		for (int i = 0; i < size; i++) {
			easyData[i] = data[i] = { 0.0, 0.0 };
		}

		// Динамическое выделение памяти для получаемых с помощью преобразования Фурье компонент спектра и их заполнение. //
		complex < double >* easyComp = new complex < double >[subSize];
		complex < double >* comp = new complex < double >[subSize];

		for (int i = 0; i < subSize; i++) {
			easyComp[i] = comp[i] = { 0.0, 0.0 };
		}


		// Реализация метода Рунге-Кутты. //
		for (int count = 1; count < size; count++) {
			h = t[count] - t[count - 1];
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					for (int l = 0; l < 2; l++) {
						for (int m = 0; m < 2; m++) {
							plus[l][m] = easyY[l][m][count - 1];
						}
					}
					k1[i][j] = easyF(t[count - 1], plus, w, omega, Delta, Omega, M, dYdt)[i][j];
					for (int l = 0; l < 2; l++) {
						for (int m = 0; m < 2; m++) {
							plus[l][m] = easyY[l][m][count - 1] + h * a[1][0] * k1[l][m];
						}
					}
					k2[i][j] = easyF(t[count - 1] + h * c[1], plus, w, omega, Delta, Omega, M, dYdt)[i][j];
					for (int l = 0; l < 2; l++) {
						for (int m = 0; m < 2; m++) {
							plus[l][m] = easyY[l][m][count - 1] + h * (a[2][0] * k1[l][m] + a[2][1] * k2[l][m]);
						}
					}
					k3[i][j] = easyF(t[count - 1] + h * c[2], plus, w, omega, Delta, Omega, M, dYdt)[i][j];
					for (int l = 0; l < 2; l++) {
						for (int m = 0; m < 2; m++) {
							plus[l][m] = easyY[l][m][count - 1] + h * (a[3][0] * k1[l][m] + a[3][1] * k2[l][m] + a[3][2] * k3[l][m]);
						}
					}
					k4[i][j] = easyF(t[count - 1] + h * c[3], plus, w, omega, Delta, Omega, M, dYdt)[i][j];
					easyY[i][j][count] = easyY[i][j][count - 1] + h * (b[0] * k1[i][j] + b[1] * k2[i][j] + b[2] * k3[i][j] + b[3] * k4[i][j]);

					for (int l = 0; l < 2; l++) {
						for (int m = 0; m < 2; m++) {
							plus[l][m] = Y[l][m][count - 1];
						}
					}
					k1[i][j] = F(t[count - 1], plus, w, omega, Delta, Omega, M, dYdt)[i][j];
					for (int l = 0; l < 2; l++) {
						for (int m = 0; m < 2; m++) {
							plus[l][m] = Y[l][m][count - 1] + h * a[1][0] * k1[l][m];
						}
					}
					k2[i][j] = F(t[count - 1] + h * c[1], plus, w, omega, Delta, Omega, M, dYdt)[i][j];
					for (int l = 0; l < 2; l++) {
						for (int m = 0; m < 2; m++) {
							plus[l][m] = Y[l][m][count - 1] + h * (a[2][0] * k1[l][m] + a[2][1] * k2[l][m]);
						}
					}
					k3[i][j] = F(t[count - 1] + h * c[2], plus, w, omega, Delta, Omega, M, dYdt)[i][j];
					for (int l = 0; l < 2; l++) {
						for (int m = 0; m < 2; m++) {
							plus[l][m] = Y[l][m][count - 1] + h * (a[3][0] * k1[l][m] + a[3][1] * k2[l][m] + a[3][2] * k3[l][m]);
						}
					}
					k4[i][j] = F(t[count - 1] + h * c[3], plus, w, omega, Delta, Omega, M, dYdt)[i][j];
					Y[i][j][count] = Y[i][j][count - 1] + h * (b[0] * k1[i][j] + b[1] * k2[i][j] + b[2] * k3[i][j] + b[3] * k4[i][j]);
				}
			}
		}

		// Перезапись полученных значений внедиагональных элементов в одномерный массив. //
		for (int k = 0; k < size; k++) {
			easyData[k] = easyY[0][1][k];
			data[k] = Y[0][1][k];
		}

		// Преобразование Фурье внедиагональных элементов. //
		easyComp = Fourier(t, easyData, f, subSize, size, per);
		comp = Fourier(t, data, f, subSize, size, per);

		// Вычисление искажений решений упрощённого уравнения относительно стандартного. //
		realErr[detune - 1] = realMistake(comp, easyComp, subSize);
		imagErr[detune - 1] = imagMistake(comp, easyComp, subSize);

		lowRealErr[detune - 1] = LowRealMistake(comp, easyComp, subSize, w, Delta, omega);
		highRealErr[detune - 1] = HighRealMistake(comp, easyComp, subSize, w, Delta, omega);
		lowImagErr[detune - 1] = LowImagMistake(comp, easyComp, subSize, w, Delta, omega);
		highImagErr[detune - 1] = HighImagMistake(comp, easyComp, subSize, w, Delta, omega);

		// Запись значения частоты излучения в массив абсцисс. //
		xAxis[detune - 1] = w / Delta;

		// Удаление из памяти. //
		for (int i = 0; i < 2; i++) {
			delete[] rho0[i];
		}
		delete[] rho0;

		for (int i = 0; i < 2; i++) {
			delete[] plus[i];
		}
		delete[] plus;

		for (int i = 0; i < 2; i++) {
			delete[] dYdt[i];
		}
		delete[] dYdt;

		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++) delete[] Y[i][j];

		}
		for (int i = 0; i < 2; i++) delete[] Y[i];
		delete[] Y;

		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++) delete[] easyY[i][j];

		}
		for (int i = 0; i < 2; i++) delete[] easyY[i];
		delete[] easyY;


		delete[] data;
		delete[] easyData;

		delete[] easyComp;
		delete[] comp;


	}

	// Вывод времени, затраченного на общее решение задачи. //
	tim = omp_get_wtime() - tim;
	cout << "time of execution: " << tim;

	// Вывод в файл. //
	ofstream fout("imag_standard-easy_output.txt");

	for (int i = 0; i < int(points / step); i++) {
		fout << realErr[i] << "        " << xAxis[i] << endl;
		cout << realErr[i] << "        " << xAxis[i] << endl;
	}
	fout << endl;
	for (int i = 0; i < int(points / step); i++) {
		fout << imagErr[i] << "        " << xAxis[i] << endl;
		cout << imagErr[i] << "        " << xAxis[i] << endl;
	}
	fout << endl;
	for (int i = 0; i < int(points / step); i++) {
		fout << lowRealErr[i] << "        " << highRealErr[i] << "        " << xAxis[i] << endl;
		cout << lowRealErr[i] << "        " << highRealErr[i] << "        " << xAxis[i] << endl;
	}
	fout << endl;
	for (int i = 0; i < int(points / step); i++) {
		fout << lowImagErr[i] << "        " << highImagErr[i] << "        " << xAxis[i] << endl;
		cout << lowImagErr[i] << "        " << highImagErr[i] << "        " << xAxis[i] << endl;
	}

	fout.close();

	// Удаление из памяти полей времени, частоты и найденных зависимостей. //
	delete[] t;

	delete[] f;

	delete[] xAxis;
	delete[] realErr;
	delete[] imagErr;

	return 0;


}

// Упрощённое динмическое уравнение матрицы плотности. //
complex < double >** easyF(double t, complex < double >** Y, double w, double omega, double Delta, double Omega, int M, std::complex< double >** dYdt) {
	double sum = 0.0;
	double L[2][2] = { {2.0, 0.0}, {0.0, 1.0} };
	for (int m = 0; m < M; m++)
		sum += cos(m * Delta * t);
	double V = (sum + 0.5) * Omega;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			dYdt[i][j] = 0.0;
	dYdt[0][0] = -1.0 * imagOne * Y[1][0] * V * exp(imagOne * (w - omega) * t) + imagOne * Y[0][1] * V * exp(-1.0 * imagOne * (w - omega) * t) - Y[0][0] + L[0][0];
	dYdt[0][1] = -1.0 * imagOne * Y[1][1] * V * exp(imagOne * (w - omega) * t) + imagOne * Y[0][0] * V * exp(imagOne * (w - omega) * t) - Y[0][1] + L[0][1];
	dYdt[1][0] = -1.0 * imagOne * Y[0][0] * V * exp(-1.0 * imagOne * (w - omega) * t) + imagOne * Y[1][1] * V * exp(-1.0 * imagOne * (w - omega) * t) - Y[1][0] + L[1][0];
	dYdt[1][1] = -1.0 * imagOne * Y[0][1] * V * exp(-1.0 * imagOne * (w - omega) * t) + imagOne * Y[1][0] * V * exp(imagOne * (w - omega) * t) - Y[1][1] + L[1][1];
	return dYdt;
}

// Стандартное динмическое уравнение матрицы плотности. //
complex < double >** F(double t, complex < double >** Y, double w, double omega, double Delta, double Omega, int M, std::complex< double >** dYdt)
{
	double sum = 0.0;
	double L[2][2] = { {2.0, 0.0}, {0.0, 1.0} };
	for (int m = 0; m < M; m++)
		sum += cos(m * Delta * t);
	double V = (sum + 0.5) * Omega;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			dYdt[i][j] = 0.0;
	dYdt[0][0] = -1.0 * imagOne * Y[1][0] * V * (exp(imagOne * (w - omega) * t) + exp(-1.0 * imagOne * (w + omega) * t)) + imagOne * Y[0][1] * V * (exp(-1.0 * imagOne * (w - omega) * t) + exp(imagOne * (w + omega) * t)) - Y[0][0] + L[0][0];
	dYdt[0][1] = -1.0 * imagOne * Y[1][1] * V * (exp(imagOne * (w - omega) * t) + exp(-1.0 * imagOne * (w + omega) * t)) + imagOne * Y[0][0] * V * (exp(imagOne * (w - omega) * t) + exp(-1.0 * imagOne * (w + omega) * t)) - Y[0][1] + L[0][1];
	dYdt[1][0] = -1.0 * imagOne * Y[0][0] * V * (exp(-1.0 * imagOne * (w - omega) * t) + exp(imagOne * (w + omega) * t)) + imagOne * Y[1][1] * V * (exp(-1.0 * imagOne * (w - omega) * t) + exp(imagOne * (w + omega) * t)) - Y[1][0] + L[1][0];
	dYdt[1][1] = -1.0 * imagOne * Y[0][1] * V * (exp(-1.0 * imagOne * (w - omega) * t) + exp(imagOne * (w + omega) * t)) + imagOne * Y[1][0] * V * (exp(imagOne * (w - omega) * t) + exp(-1.0 * imagOne * (w + omega) * t)) - Y[1][1] + L[1][1];
	return dYdt;
}

// Преобразование Фурье. //
complex <double>* Fourier(double* t, complex < double >* Y, double* f, int subSize, int size, int per)
{
	complex <double>* comp = new complex <double>[subSize];
	double h;
	int i;
	for (i = 0; i < subSize; i++) {
		comp[i] = { 0.0, 0.0 };
		h = abs(t[size - 1] - t[size - per]) / per;
		for (int k = size - per; k < size; k++) {
			comp[i] += h * Y[k] * exp(imagOne * f[i] * 2.0 * M_PI * t[k]);
		}
	}
	return comp;
	delete[] comp;
}

// Функция нахождения искажений вещественной части спектра решения. //
double realMistake(complex <double>* comp, complex <double>* easyComp, int subSize) {
	double upSum = 0.0, lowSum = 0.0, mistake = 0.0;
	for (int i = 0; i < subSize; i++) {
		upSum += (abs(real(comp[i])) - abs(real(easyComp[i]))) * (abs(real(comp[i])) - abs(real(easyComp[i])));
		lowSum += real(comp[i]) * real(comp[i]);
	}
	mistake = sqrt(upSum / lowSum);
	return mistake;
}

// Функция нахождения искажений мнимой части спектра решения. //
double imagMistake(complex <double>* comp, complex <double>* easyComp, int subSize) {
	double upSum = 0.0, lowSum = 0.0, mistake = 0.0;
	for (int i = 0; i < subSize; i++) {
		upSum += (abs(imag(comp[i])) - abs(imag(easyComp[i]))) * (abs(imag(comp[i])) - abs(imag(easyComp[i])));
		lowSum += imag(comp[i]) * imag(comp[i]);
	}
	mistake = sqrt(upSum / lowSum);
	return mistake;
}

// Функция нахождения низкочастотной части искажений вещественной части спектра решения. //
double LowRealMistake(complex <double>* comp, complex <double>* easyComp, int subSize, double w, double Delta, double omega) {
	double upSum = 0.0, lowSum = 0.0, mistake = 0.0;
	for (int i = 0; i < subSize; i++) {
		if (i > ((subSize - 1) / 2 + 1 - abs(w / Delta)) && i < (subSize - 1) / 2 + 1 + abs(w / Delta))
			upSum += (abs(real(comp[i])) - abs(real(easyComp[i]))) * (abs(real(comp[i])) - abs(real(easyComp[i])));
		lowSum += real(comp[i]) * real(comp[i]);
	}
	mistake = sqrt(upSum / lowSum);
	return mistake;
}

// Функция нахождения высокочастотной части искажений вещественной части спектра решения. //
double HighRealMistake(complex <double>* comp, complex <double>* easyComp, int subSize, double w, double Delta, double omega){
	double upSum = 0.0, lowSum = 0.0, mistake = 0.0;
	for (int i = 0; i < subSize; i++) {
		if (i < ((subSize - 1) / 2 + 1 - abs(w / Delta)) || i > (subSize - 1) / 2 + 1 + abs(w / Delta))
			upSum += (abs(real(comp[i])) - abs(real(easyComp[i]))) * (abs(real(comp[i])) - abs(real(easyComp[i])));
		lowSum += real(comp[i]) * real(comp[i]);
	}
	mistake = sqrt(upSum / lowSum);
	return mistake;
}

// Функция нахождения низкочастотной части искажений мнимой части спектра решения. //
double LowImagMistake(complex <double>* comp, complex <double>* easyComp, int subSize, double w, double Delta, double omega) {
	double upSum = 0.0, lowSum = 0.0, mistake = 0.0;
	for (int i = 0; i < subSize; i++) {
		if (i > ((subSize - 1) / 2 + 1 - abs(w / Delta)) && i < (subSize - 1) / 2 + 1 + abs(w / Delta))
			upSum += (abs(imag(comp[i])) - abs(imag(easyComp[i]))) * (abs(imag(comp[i])) - abs(imag(easyComp[i])));
		lowSum += imag(comp[i]) * imag(comp[i]);
	}
	mistake = sqrt(upSum / lowSum);
	return mistake;
}
