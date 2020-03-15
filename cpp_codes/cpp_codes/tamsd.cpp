#include <iostream>
#include <fstream>
#include <math.h>
#include <sstream>
#include <omp.h>
using namespace std;

//Time-Averaged probability distribution function initiation
float **TAMSDLOG(int t, int len, float delta_t, int max_div,
		 int run_num, double **trajectory);
//Reading trajectory file produced by fOU Python Library
double **read_trajectory(int len, int run_num);

//----------------------------- MAIN ------------------------------------//
int main()
{
	omp_set_dynamic(0);
	omp_set_num_threads(6);
	int len, run_num, max_div, t;
	char ilabel;
	ofstream output_tamsd;
	output_tamsd.open("tamsd_output.txt");

	//************** Reading the Input File ***************//
	FILE *input_file;
	input_file = fopen("input.txt","r");
	fscanf(input_file, "%s\t%d\n", &ilabel, &len);
	fscanf(input_file, "%s\t%d\n", &ilabel, &run_num);
	fscanf(input_file, "%s\t%d\n", &ilabel, &max_div);
	fscanf(input_file, "%s\t%d\n", &ilabel, &t);
	//*****************************************************//

	float delta_t = float(t)/float(len - 1.0);
	
	cout << "The length of the trajectory is: " << len << endl;
	cout << "The number of trajectories: " << run_num << endl; 
	cout << "The MAX_DIV is: " << max_div << endl;
	cout << "The real time t is: " << t << endl;
	cout << "The time increment is: " << delta_t << endl;

	double **trajectory = read_trajectory(len, run_num);
	float **tamsd = TAMSDLOG(t, len, delta_t, max_div, run_num,
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
//--------------------------- END MAIN ----------------------------------//


//--------------------------Read Traj. function--------------------------//
double **read_trajectory(int len, int run_num)
{
	ifstream trajectory_file;
	trajectory_file.open("trajectories.txt");

	double **trajectory = new double*[run_num];										//Initializing a dynamic lattice matrix (Rows)

	for (int i=0 ; i < run_num ; i++)
	{
		trajectory[i] = new double[len];									//Initializing a dynamic lattice matrix (Columns)
	}
	if(trajectory_file.is_open())
	{
		for (int j=0; j<len; j++)
		{
			if (j%100 == 0)
			{
				cout << j << endl;
			}
			string row;
			if ( getline(trajectory_file, row) )
			{
				istringstream istr(row);
				double number;
				for (int i=0; i<run_num; i++)
				{
					istr >> trajectory[i][j];
				}
			}
		}
		trajectory_file.close();
	}
	else
	{
		cout << "Could not open the file" << endl;
		exit;
	}
	return trajectory;
}
//----------------------End of Read Traj function---------------------//
//----------------------------TAMSD calculation-----------------------//
float **TAMSDLOG(int t, int len, float delta_t, int max_div, int run_num,
		 double **trajectory)
{
	float dt;
	int lagtime;
	float **tamsd = (float **) calloc (run_num,sizeof(float *));
	for (int i=0; i<run_num; i++)
	{
		tamsd[i] = (float *) calloc (max_div, sizeof(float));
	}
	#pragma omp parallel for
	for (int k=0; k<run_num; k++)
	{
		for(int i=0 ; i<max_div ; i++)
		{
			dt = 1.0*(i)*(log10(len)-log10(1))/max_div;
			lagtime = (int)(pow(10,dt));
			for(int j=0; j<len-lagtime; j++)

			{
				tamsd[k][i] = tamsd[k][i] + pow(trajectory[k][j+lagtime]-trajectory[k][j],2)*delta_t; 
			}
			tamsd[k][i] = tamsd[k][i]/(float(t-lagtime*delta_t)) ;
		}
	}
	return tamsd;
}
//------------------------End of TAMSD calculation---------------------//
