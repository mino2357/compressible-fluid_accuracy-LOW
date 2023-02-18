#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <omp.h>

class Vec2D{
	public:
	double x;
	double y;
};

using Float = double;
using Integer = int;
using unsignedInteger = unsigned int;
using Scalar2D = std::vector<std::vector<Float>>;
using Vector2D = std::vector<std::vector<Vec2D>>;

constexpr Integer Nx      = 1601;
constexpr Integer Ny      = 1601;
constexpr Integer INTV    = 2500;
constexpr Float   Lx      = 1.0;
constexpr Float   Ly      = 1.0;
constexpr Float   dt      = 4.0e-6;
constexpr Float   dx      = Lx / (Nx - 1);
constexpr Float   dy      = Ly / (Ny - 1);
constexpr Float   mu      = 1.0 / 100000.0;
constexpr Float   inv_dx  = 1.0 / dx;
constexpr Float   inv_2dx = 1.0 / (2.0 * dx);
constexpr Float   inv_dy  = 1.0 / dy;
constexpr Float   inv_2dy = 1.0 / (2.0 * dy);

// H. Iijima, H. Hotta, S. Imada "Semi-conservative reduced speed of sound technique for low Mach number flows with large density variations"
// ref. https://arxiv.org/abs/1812.04135
constexpr Float xi = 2.0;
constexpr Float inv_xi_2 = 1.0 / (xi * xi);

inline Float stateEq(Float a){
	return 10.0 * a;
}

inline void initScalar2D(Scalar2D& a){
	a.resize(Nx);
	unsignedInteger j=0;
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<a.size(); ++i){
		a[i].resize(Ny);
		for(j=0; j<a[i].size(); ++j){
			a[i][j] = 1.0;
		}
	}
}

inline void initVector2D(Vector2D& v){
	v.resize(Nx);
	unsignedInteger j=0;
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<v.size(); ++i){
		v[i].resize(Ny);
		for(j=0; j<v[i].size(); ++j){
			v[i][j].x = 0.0;
			v[i][j].y = 0.0;
		}
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<v.size(); ++i){
		v[i][0].x = 0.0;
		v[i][0].y = 0.0;
		v[i][v[i].size()-1].x = 1.0;
		v[i][v[i].size()-1].y = 0.0;
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger j=0; j<v[0].size(); ++j){
		v[0][j].x = 0.0;
		v[0][j].y = 0.0;
		v[v.size()-1][j].x = 0.0;
		v[v.size()-1][j].y = 0.0;
	}
}

inline void diffusionVector(const Scalar2D& a, Vector2D& vec, Vector2D& tmp_vec){
	unsignedInteger j=0;
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(j=0; j<vec[i].size(); ++j){
			tmp_vec[i][j].x = vec[i][j].x;
			tmp_vec[i][j].y = vec[i][j].y;
		}
	}
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<vec.size()-1; i++){
		for(j=1; j<vec[i].size()-1; ++j){
			auto inv_x = 1.0 / (dx * dx * a[i][j]);
			auto inv_y = 1.0 / (dy * dy * a[i][j]);
			vec[i][j].x +=
						+ dt * mu * inv_x * (tmp_vec[i + 1][j].x - 2.0 * tmp_vec[i][j].x + tmp_vec[i - 1][j].x)
						+ dt * mu * inv_y * (tmp_vec[i][j + 1].x - 2.0 * tmp_vec[i][j].x + tmp_vec[i][j - 1].x);
			vec[i][j].y +=
						+ dt * mu * inv_x * (tmp_vec[i + 1][j].y - 2.0 * tmp_vec[i][j].y + tmp_vec[i - 1][j].y)
						+ dt * mu * inv_y * (tmp_vec[i][j + 1].y - 2.0 * tmp_vec[i][j].y + tmp_vec[i][j - 1].y);
		}
	}
}

inline void advection(Vector2D& vec, Vector2D& tmp_vec){
	unsignedInteger j=0;
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(j=0; j<vec[i].size(); ++j){
			tmp_vec[i][j].x = vec[i][j].x;
			tmp_vec[i][j].y = vec[i][j].y;
		}
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<vec.size()-1; i++){
		for(j=1; j<vec[i].size()-1; ++j){
			tmp_vec[i][j].x = vec[i][j].x
						- dt * (vec[i][j].x * (vec[i+1][j].x - vec[i-1][j].x) * inv_2dx
						- std::abs(vec[i][j].x) * 0.5 * (vec[i+1][j].x - 2.0 * vec[i][j].x + vec[i-1][j].x) * inv_dx)
						- dt * (vec[i][j].y * (vec[i][j+1].x - vec[i][j-1].x) * inv_2dy
						- std::abs(vec[i][j].y) * 0.5 * (vec[i][j+1].x - 2.0 * vec[i][j].x + vec[i][j-1].x) * inv_dy);
			tmp_vec[i][j].y = vec[i][j].y
						- dt * (vec[i][j].x * (vec[i+1][j].y - vec[i-1][j].y) * inv_2dx
						- std::abs(vec[i][j].x) * 0.5 * (vec[i+1][j].y - 2.0 * vec[i][j].y + vec[i-1][j].y) * inv_dx)
						- dt * (vec[i][j].y * (vec[i][j+1].y - vec[i][j-1].y) * inv_2dy
						- std::abs(vec[i][j].y) * 0.5 * (vec[i][j+1].y - 2.0 * vec[i][j].y + vec[i][j-1].y) * inv_dy);
		}
	}
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<vec.size() - 1; i++){
		for(j=1; j<vec[i].size() - 1; ++j){
			vec[i][j].x = tmp_vec[i][j].x;
			vec[i][j].y = tmp_vec[i][j].y;
		}
	}
}

inline void gradient(const Scalar2D& a, Vector2D& vec){
	unsignedInteger j=0;
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<a.size()-1; ++i){
		for(j=1; j<a[i].size()-1; ++ j){
			//auto inv_x = 1.0 / (2.0 * dx * a[i][j]);
			//auto inv_y = 1.0 / (2.0 * dy * a[i][j]);
			vec[i][j].x += - dt * (stateEq(a[i+1][j]) - stateEq(a[i-1][j])) / (2.0 * dx * a[i][j]);
			vec[i][j].y += - dt * (stateEq(a[i][j+1]) - stateEq(a[i][j-1])) / (2.0 * dy * a[i][j]);
		}
	}
}

inline void transportContinuity(Scalar2D& a, Vector2D& vec, Scalar2D& tmp_a, Vector2D& tmp_vec){
	unsignedInteger j=0;
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(j=0; j<vec[i].size(); ++j){
			tmp_vec[i][j].x = vec[i][j].x;
			tmp_vec[i][j].y = vec[i][j].y;
			tmp_a[i][j] = a[i][j];
		}
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<a.size()-1; ++i){
		for(j=1; j<a[i].size()-1; ++ j){
			a[i][j] +=
					- dt * tmp_a[i][j] * (tmp_vec[i+1][j].x - tmp_vec[i-1][j].x) * inv_2dx * inv_xi_2
					- dt * tmp_a[i][j] * (tmp_vec[i][j+1].y - tmp_vec[i][j-1].y) * inv_2dy * inv_xi_2
					- dt * (tmp_vec[i][j].x * (tmp_a[i+1][j] - tmp_a[i-1][j]) * inv_2dx
					- std::abs(tmp_vec[i][j].x) * (tmp_a[i+1][j] - 2.0 * tmp_a[i][j] + tmp_a[i-1][j]) * inv_2dx) * inv_xi_2
					- dt * (tmp_vec[i][j].y * (tmp_a[i][j+1] - tmp_a[i][j-1]) * inv_2dy
					- std::abs(tmp_vec[i][j].y) * (tmp_a[i][j+1] - 2.0 * tmp_a[i][j] + tmp_a[i][j-1]) * inv_2dy) * inv_xi_2;
		}
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<a.size(); ++i){
		a[i][0] = a[i][1];
		a[i][a[i].size()-1] = a[i][a[i].size()-2];
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger j=0; j<a[0].size(); ++j){
		a[0][j] = a[1][j];
		a[a.size()-1][j] = a[a.size()-2][j];
	}
}

int main(){
	Scalar2D a;
	Vector2D u;
	initScalar2D(a);
	initVector2D(u);
	auto tmp_a = a;
	auto tmp_vec = u;
	auto u_old = u;
	auto scale = 12.0 / Nx;
	std::string str1 = "set output 'fluid-";
	std::string str2 = ".png'\n";
	std::FILE *gp = popen("gnuplot -persist", "w" );
	fprintf(gp, "set terminal png\n");
	fprintf(gp, "set term pngcairo size 3840, 2160\n"); // 7680, 4320
	//fprintf(gp, "set nokey\n");

	unsignedInteger j=0;
	std::chrono::system_clock::time_point  start, end;
	for(auto itr=0; ;itr++){
		#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
		for(unsignedInteger i=0; i<u.size(); i++){
			for(j=0; j<u[i].size(); ++j){
				u_old[i][j].x = u[i][j].x;
				u_old[i][j].y = u[i][j].y;
			}
		}
		diffusionVector(a, u, tmp_vec);
		gradient(a, u);
		advection(u, tmp_vec);
		transportContinuity(a, u_old, tmp_a, tmp_vec);
		if(itr%INTV == 0){
			std::cout << itr/INTV;
			std::ostringstream oss;
			oss.setf(std::ios::right);
			oss.fill('0');
			oss.width(8);
			oss << itr / INTV;
			std::string num_str = oss.str();
			//
			std::ofstream outputfile1("cav" + num_str + ".dat");
			for(unsignedInteger i=0; i<a.size(); i+=10){
				for(j=0; j<a[i].size(); j+=10){
					outputfile1 << std::setprecision(12) << i * dx << " " << j * dy << " " << u[i][j].x << " " << u[i][j].y << "\n";
				}
			}
			outputfile1.close();
			//
			std::ofstream outputfile2("cav-uy" + num_str + ".dat");
			for(unsignedInteger i=0; i<a.size(); i++){
				outputfile2 << std::setprecision(14) << i * dx << " " << u[Nx/2][i].x << "\n";
			}
			outputfile2.close();
			//
			std::ofstream outputfile3("cav-vx" + num_str + ".dat");
			for(unsignedInteger i=0; i<a.size(); i++){
				outputfile3 << std::setprecision(14) << i * dx << " " << u[i][Ny/2].y << "\n";
			}
			outputfile3.close();
			//
			std::ofstream outputfile4("cav-rho" + num_str + ".dat");
			for(unsignedInteger i=0; i<a.size(); i+=2){
				for(j=0; j<a[i].size(); j+=2){
					outputfile4 << std::setprecision(15) << i * dx << " " << j * dy << " " << a[i][j] << "\n";
				}
				outputfile4 << std::setprecision(15) << "\n";
			}
			outputfile4.close();
			//
			auto png = "set output '" + num_str + ".png'\n";
			fprintf(gp, "%s", png.c_str());
			//
			fprintf(gp, "set multiplot\n");
			//
			fprintf(gp, "reset\n");
			fprintf(gp, "set lmargin screen 0.1234\n");
			fprintf(gp, "set rmargin screen 0.45\n");
			fprintf(gp, "set tmargin screen 0.975\n");
			fprintf(gp, "set bmargin screen 0.525\n");
			fprintf(gp, "set title\n");
			fprintf(gp, "set xr [0:1.0]\n");
			fprintf(gp, "set yr [0:1.0]\n");
			fprintf(gp, "set palette defined(0'#aaaaaa',0.2'#00008b',0.4'#2ca9e1',0.6'#008000',0.8'#ffff00',1.0'#eb6101',4.0'#8b0000')\n");
			fprintf(gp, "set size square\n");
			fprintf(gp, "set grid\n");
			auto plot = "plot 'cav" + num_str + ".dat' u 1:2:($3/(sqrt($3*$3+$4*$4)+1.0e-12)*" + std::to_string(scale) + "):($4/(sqrt($3*$3+$4*$4)+1.0e-12)*" + std::to_string(scale) + "):(sqrt($3*$3+$4*$4)) with vector lc palette notitle \n";
			fprintf(gp, "%s", plot.c_str());
			//
			fprintf(gp, "reset\n");
			//fprintf(gp, "set title\n");
			fprintf(gp, "set lmargin screen 0.05\n");
			fprintf(gp, "set rmargin screen 0.45\n");
			fprintf(gp, "set tmargin screen 0.475\n");
			fprintf(gp, "set bmargin screen 0.025\n");
			fprintf(gp, "set title\n");
			fprintf(gp, "set xr [0:1.0]\n");
			fprintf(gp, "set yr [0:1.0]\n");
			fprintf(gp, "set zr [0.998:1.002]\n");
			//fprintf(gp, "set noxtics\n");
			//fprintf(gp, "set noytics\n");
			//fprintf(gp, "set nokey\n");
			fprintf(gp, "set palette defined(0 '#000090',1 '#000fff',2 '#0090ff',3 '#0fffee',4 '#90ff70',5 '#ffee00',6 '#ff7000',7 '#ee0000',8 '#7f0000')\n");
			fprintf(gp, "set view map\n");
			fprintf(gp, "set size square\n");
			fprintf(gp, "set pm3d\n");
			fprintf(gp, "set pm3d map\n");
			plot = "splot 'cav-rho" + num_str + ".dat' with pm3d notitle \n";
			fprintf(gp, "%s", plot.c_str());
			//
			fprintf(gp, "reset\n");
			fprintf(gp, "set lmargin screen 0.55\n");
			fprintf(gp, "set rmargin screen 0.95\n");
			fprintf(gp, "set tmargin screen 0.95\n");
			fprintf(gp, "set bmargin screen 0.55\n");
			fprintf(gp, "set xr [0:1.0]\n");
			//fprintf(gp, "set yr [-0.6:1]\n");
			plot = "plot 'cav-uy" + num_str + ".dat' w l\n";
			fprintf(gp, "%s", plot.c_str());
			//
			fprintf(gp, "reset\n");
			fprintf(gp, "set lmargin screen 0.55\n");
			fprintf(gp, "set rmargin screen 0.95\n");
			fprintf(gp, "set tmargin screen 0.45\n");
			fprintf(gp, "set bmargin screen 0.05\n");
			fprintf(gp, "set xr [0:1]\n");
			//fprintf(gp, "set yr [-0.6:1]\n");
			plot = "plot 'cav-vx" + num_str + ".dat' w l\n";
			fprintf(gp, "%s", plot.c_str());
			//
			fprintf(gp, "unset multiplot\n");
			//fprintf(gp, "quit\n");
			fflush(gp);
			std::cout << " png generated." << std::endl;
		}
		if(itr%INTV == 0){
 			start = std::chrono::system_clock::now();
		}else if(itr%INTV == INTV-1){
			end = std::chrono::system_clock::now();
			double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
			std::cout << elapsed / (INTV - 1) << " [ms/step]" << std::endl;
		}
	}
	pclose(gp);
}
