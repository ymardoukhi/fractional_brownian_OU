#include <iostream>
#include <fstream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <libfbm.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
using namespace std;

//Time-Averaged probability distribution function initiation
double **TAMSDLOG(double t, int len, double delta_t, int max_div,
		 int run_num, double **trajectory);

//Fractional Ornstein-Uhlenbeck Trajectories
double **fou_trajectory(int len, double delta_t, int run_num, double H,
		       double t, double lambda, double sigma);

//----------------------------- MAIN ------------------------------------//
int main()
{
	int run_num = 1; int max_div = 1000;
	int len = 1000;
	char ilabel;
	double H = 0.5; double t = 30.0;
	double lambda = 2.0; double sigma = 2.0;

	ofstream output_tamsd;
	output_tamsd.open("tamsd_output.txt");

	//************** Reading the Input File ***************//
	//FILE *input_file;
	//input_file = fopen("input.txt","r");
	//fscanf(input_file, "%s%d\n", &ilabel, &len);
	//fscanf(input_file, "%s%d\n", &ilabel, &run_num);
	//fscanf(input_file, "%s%d\n", &ilabel, &max_div);
	//fscanf(input_file, "%s%f\n", &ilabel, &t);
	//fscanf(input_file, "%s%f\n", &ilabel, &H);
	//fscanf(input_file, "%s%f\n", &ilabel, &lambda);
	//fscanf(input_file, "%s%f\n", &ilabel, &sigma);
	//*****************************************************//

	
	double delta_t = t/double(len);

	
	cout << "The length of the trajectory is: " << len << endl;
	cout << "The number of trajectories: " << run_num << endl; 
	cout << "The MAX_DIV is: " << max_div << endl;
	cout << "The real time t is: " << t << endl;
	cout << "The time increment is: " << delta_t << endl;
	cout << "The Hurst exponent is: " << H << endl;
	cout << "Lambda is: " << lambda << endl;
	cout << "Sigma is: " << sigma << endl;

	double **trajectory = fou_trajectory(len, delta_t, run_num,
					    H, t, lambda, sigma);
	double **tamsd = TAMSDLOG(t, len, delta_t, max_div, run_num,
				  trajectory);
	for (int i=0; i<run_num; i++)
	{
		for (int j=0; j<max_div; j++)
		{
			output_tamsd << tamsd[i][j] << "\t";
		}
		output_tamsd << endl;
	}
	output_tamsd.close();
	return 0;
}
//------------------------------- END MAIN ---------------------------//
//----------------------------TAMSD calculation-----------------------//
double **TAMSDLOG(double t, int len, double delta_t, int max_div, int run_num,
		  double **trajectory)
{
	float dt;
	int lagtime;
	double **tamsd = (double **) calloc (run_num,sizeof(double *));
	for (int i=0; i<run_num; i++)
	{
		tamsd[i] = (double *) calloc (max_div, sizeof(double));
	}
	for (int k=0; k<run_num; k++)
	{
		for(int i=0 ; i<max_div ; i++)
		{
			dt = 1.0*(i)*(log10(len)-log10(1))/max_div;
			lagtime = (int)(pow(10,dt));
			for(int j=0; j<len-lagtime; j++)

			{
				tamsd[k][i] += pow(trajectory[k][j+lagtime]-
						trajectory[k][j],2)*delta_t; 
			}
			tamsd[k][i] = tamsd[k][i]/(double(t-lagtime*delta_t)) ;
		}
	}
	return tamsd;
}
//------------------------End of TAMSD calculation---------------------//
//-------------- Fractional Ornstein-Uhlenbeck Trajectories -----------//
double **fou_trajectory(int len, double delta_t, int run_num, double H,
		       double t, double lambda, double sigma)
{
	ofstream output_file;
	output_file.open("trajectory.txt");
	ofstream initial_pos;
	initial_pos.open("initial_pos.txt");
	ofstream noise;
	noise.open("noise.txt");

	double var = sqrt(pow(sigma, 2)/(2*(pow(lambda, 2*H)))*tgamma(2*H+1));
	cout << "var is: " << var << endl;
	boost::mt19937 *rng = new boost::mt19937();
	rng->seed(time(NULL));
	boost::normal_distribution<> distribution(0.0, var);
	boost::variate_generator< boost::mt19937, boost::normal_distribution<> >
						  dist(*rng, distribution);
	
	libfbm::zvec dim(1);
	dim[0] = len;
	libfbm::FGNContext ctx(H, dim);
	libfbm::Field fgn(ctx, true);

	double **trajectory = (double **) calloc (run_num, sizeof(double));
	for (int i=0; i<run_num; i++)
	{
		trajectory[i] = (double *) calloc (len, sizeof(double *));
	}

	double *time_array = (double *) calloc (len, sizeof(double));

	for (int i=0; i<len; i++)
	{
		time_array[i] = double(i)*delta_t;
	}
	for (int i=0; i<run_num; i++)
	{
		if (i%10 == 0) cout << "Run number: " << i << endl;
		fgn.generate();
		double x_0 = dist();
		initial_pos << x_0 << endl;;
		trajectory[i][0] = x_0;
		output_file << trajectory[i][0] << "\t";
		for (int j=1; j<len; j++)
		{
			double integral_sum = 0;
			for (int k=0; k<j; k++)
			{
				integral_sum = integral_sum + exp(lambda*time_array[k])*fgn(k);
			}
			cout << integral_sum << endl;
			trajectory[i][j] = exp(-lambda*time_array[j])*(x_0 +
					   sigma*integral_sum);
			output_file << trajectory[i][j] << "\t";
			noise << fgn(j) << "\t" << time_array[j] << endl;
		}
		output_file << endl;
		//noise << endl;
	      
	}
	output_file.close();
	initial_pos.close();
	noise.close();
	return trajectory;
}
