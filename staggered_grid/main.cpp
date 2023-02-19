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

constexpr Integer Nx      = 601;
constexpr Integer Ny      = 601;
constexpr Integer INTV    = 25000;
constexpr Float   Lx      = 1.0;
constexpr Float   Ly      = 1.0;
constexpr Float   dt      = 1.0e-5;
constexpr Float   dx      = Lx / (Nx - 1);
constexpr Float   dy      = Ly / (Ny - 1);
constexpr Float   mu      = 1.0 / 100.0; //100000.0;
constexpr Float   inv_dx  = 1.0 / dx;
constexpr Float   inv_2dx = 1.0 / (2.0 * dx);
constexpr Float   inv_dy  = 1.0 / dy;
constexpr Float   inv_2dy = 1.0 / (2.0 * dy);
constexpr Float   n_x     = 1.0;
constexpr Float   e_x     = 0.0;
constexpr Float   w_x     = 0.0;
constexpr Float   s_x     = 0.0;
constexpr Float   n_y     = 0.0;
constexpr Float   e_y     = 0.0;
constexpr Float   w_y     = 0.0;
constexpr Float   s_y     = 0.0;

// H. Iijima, H. Hotta, S. Imada "Semi-conservative reduced speed of sound technique for low Mach number flows with large density variations"
// ref. https://arxiv.org/abs/1812.04135
constexpr Float xi = 2.0;
constexpr Float inv_xi_2 = 1.0 / (xi * xi);

inline Float stateEq(Float a){
	return 10.0 * a;
}

inline void initScalar2D(Scalar2D& a){
	a.resize(Nx+1);
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<a.size(); ++i){
		a[i].resize(Ny+1);
		for(unsignedInteger j=0; j<a[i].size(); ++j){
			a[i][j] = 1.0;
		}
	}
}

inline void initVector2D(Vector2D& v){
	v.resize(Nx);
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<v.size(); ++i){
		v[i].resize(Ny);
		for(unsignedInteger j=0; j<v[i].size(); ++j){
			v[i][j].x = 0.0;
			v[i][j].y = 0.0;
		}
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<v.size()-1; ++i){
		v[i][0].x = s_x;
		v[i][0].y = s_y;
		v[i][v[i].size()-1].x = n_x;
		v[i][v[i].size()-1].y = n_y;
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger j=1; j<v[0].size()-1; ++j){
		v[0][j].x = w_x;
		v[0][j].y = w_y;
		v[v.size()-1][j].x = e_x;
		v[v.size()-1][j].y = e_y;
	}
}

inline void diffusionVector(Scalar2D& a, Vector2D& vec, Vector2D& tmp_vec){
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(unsignedInteger j=0; j<vec[i].size(); ++j){
			tmp_vec[i][j].x = vec[i][j].x;
			tmp_vec[i][j].y = vec[i][j].y;
		}
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<vec.size()-1; i++){
		for(unsignedInteger j=1; j<vec[i].size()-1; ++j){
			auto a_ave = (a[i][j] + a[i+1][j] + a[i][j+1] + a[i+1][j+1]) / 4.0;
			vec[i][j].x +=
						+ dt * mu * (tmp_vec[i+1][j].x - 2.0 * tmp_vec[i][j].x + tmp_vec[i-1][j].x) / (dx * dx * a_ave)
						+ dt * mu * (tmp_vec[i][j+1].x - 2.0 * tmp_vec[i][j].x + tmp_vec[i][j-1].x) / (dy * dy * a_ave);
			vec[i][j].y +=
						+ dt * mu * (tmp_vec[i+1][j].y - 2.0 * tmp_vec[i][j].y + tmp_vec[i-1][j].y) / (dx * dx * a_ave)
						+ dt * mu * (tmp_vec[i][j+1].y - 2.0 * tmp_vec[i][j].y + tmp_vec[i][j-1].y) / (dy * dy * a_ave);
		}
	}
	// BC.
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<vec.size()-1; i++){
		vec[i][0].x = s_x;
		vec[i][0].y = s_y;
		vec[i][vec[i].size()-1].x = n_x;
		vec[i][vec[i].size()-1].y = n_y;
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger j=1; j<vec.size()-1; ++j){
		vec[0][j].x = w_x;
		vec[0][j].y = w_y;
		vec[vec.size()-1][j].x = e_x;
		vec[vec.size()-1][j].y = e_y;
	}
}

inline void advection(Vector2D& vec, Vector2D& tmp_vec){
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(unsignedInteger j=0; j<vec[i].size(); ++j){
			tmp_vec[i][j].x = vec[i][j].x;
			tmp_vec[i][j].y = vec[i][j].y;
		}
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<vec.size()-1; i++){
		for(unsignedInteger j=1; j<vec[i].size()-1; ++j){
			tmp_vec[i][j].x = vec[i][j].x
						- dt * (vec[i][j].x * (vec[i+1][j].x - vec[i-1][j].x) * inv_2dx
						- std::abs(vec[i][j].x) / 2.0 * (vec[i+1][j].x - 2.0 * vec[i][j].x + vec[i-1][j].x) * inv_dx)
						- dt * (vec[i][j].y * (vec[i][j+1].x - vec[i][j-1].x) * inv_2dy
						- std::abs(vec[i][j].y) / 2.0 * (vec[i][j+1].x - 2.0 * vec[i][j].x + vec[i][j-1].x) * inv_dy);
			tmp_vec[i][j].y = vec[i][j].y
						- dt * (vec[i][j].x * (vec[i+1][j].y - vec[i-1][j].y) * inv_2dx
						- std::abs(vec[i][j].x) / 2.0 * (vec[i+1][j].y - 2.0 * vec[i][j].y + vec[i-1][j].y) * inv_dx)
						- dt * (vec[i][j].y * (vec[i][j+1].y - vec[i][j-1].y) * inv_2dy
						- std::abs(vec[i][j].y) / 2.0 * (vec[i][j+1].y - 2.0 * vec[i][j].y + vec[i][j-1].y) * inv_dy);
		}
	}
	// BC.
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<vec.size(); i++){
		vec[i][0].x = s_x;
		vec[i][0].y = s_y;
		vec[i][vec[i].size()-1].x = n_x;
		vec[i][vec[i].size()-1].y = n_y;
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger j=0; j<vec.size(); ++j){
		vec[0][j].x = w_x;
		vec[0][j].y = w_y;
		vec[vec.size()-1][j].x = e_x;
		vec[vec.size()-1][j].y = e_y;
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(unsignedInteger j=0; j<vec[i].size(); ++j){
			vec[i][j].x = tmp_vec[i][j].x;
			vec[i][j].y = tmp_vec[i][j].y;
		}
	}
}

inline void gradient(const Scalar2D& a, Vector2D& vec){
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<vec.size()-1; ++i){
		for(unsignedInteger j=1; j<vec[i].size()-1; ++ j){
			auto left_bottom  = a[i][j];
			auto right_bottom = a[i+1][j];
			auto left_top     = a[i][j+1];
			auto right_top    = a[i+1][j+1];
			auto a_ave = (left_bottom + right_bottom + left_top + right_top) / 4.0;
			vec[i][j].x += - dt * 0.5 * ((stateEq(right_bottom) - stateEq(left_bottom)) * inv_dx + (stateEq(right_top) - stateEq(left_top)) * inv_dx) / a_ave;
			vec[i][j].y += - dt * 0.5 * ((stateEq(left_top) - stateEq(left_bottom)) * inv_dy + (stateEq(right_top) - stateEq(right_bottom)) * inv_2dy) / a_ave;
		}
	}
}

inline void transportContinuity(Scalar2D& a, Vector2D& vec, Scalar2D& tmp_a, Vector2D& tmp_vec){
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(unsignedInteger j=0; j<vec[i].size(); ++j){
			tmp_vec[i][j].x = vec[i][j].x;
			tmp_vec[i][j].y = vec[i][j].y;
		}
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<a.size(); i++){
		for(unsignedInteger j=0; j<a[i].size(); ++j){
			tmp_a[i][j] = a[i][j];
		}
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<a.size()-1; ++i){
		for(unsignedInteger j=1; j<a[i].size()-1; ++j){
			// x
			auto vec_c_x = 0.25 * (vec[i-1][j-1].x + vec[i][j-1].x + vec[i-1][j].x + vec[i][j].x);
			//auto vec_n_x = 0.50 * (vec[i-1][j].x   + vec[i][j].x);
			//auto vec_e_x = 0.50 * (vec[i][j-1].x   + vec[i][j].x);
			//auto vec_w_x = 0.50 * (vec[i-1][j-1].x + vec[i][j-1].x);
			//auto vec_s_x = 0.50 * (vec[i-1][j-1].x + vec[i-1][j].x);
			// y
			auto vec_c_y = 0.25 * (vec[i-1][j-1].y + vec[i][j-1].y + vec[i-1][j].y + vec[i][j].y);
			//auto vec_n_y = 0.50 * (vec[i-1][j].y   + vec[i][j].y);
			//auto vec_e_y = 0.50 * (vec[i][j-1].y   + vec[i][j].y);
			//auto vec_w_y = 0.50 * (vec[i-1][j-1].y + vec[i][j-1].y);
			//auto vec_s_y = 0.50 * (vec[i-1][j-1].y + vec[i-1][j].y);
			a[i][j] +=
					- dt * tmp_a[i][j] * ((tmp_vec[i][j].x - tmp_vec[i-1][j].x) * inv_dx + (tmp_vec[i][j-1].x - tmp_vec[i-1][j-1].x) * inv_dx) * (0.5 * inv_xi_2)
					- dt * tmp_a[i][j] * ((tmp_vec[i][j].y - tmp_vec[i][j-1].y) * inv_dy + (tmp_vec[i-1][j].y - tmp_vec[i-1][j-1].y) * inv_dy) * (0.5 * inv_xi_2)
					- dt * (vec_c_x * (tmp_a[i+1][j] - tmp_a[i-1][j]) * inv_2dx
					- std::abs(vec_c_x) * (tmp_a[i+1][j] - 2.0 * tmp_a[i][j] + tmp_a[i-1][j]) * inv_2dx) * inv_xi_2
					- dt * (vec_c_y * (tmp_a[i][j+1] - tmp_a[i][j-1]) * inv_2dy
					- std::abs(vec_c_y) * (tmp_a[i][j+1] - 2.0 * tmp_a[i][j] + tmp_a[i][j-1]) * inv_2dy) * inv_xi_2;
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

	std::chrono::system_clock::time_point  start, end;
	for(auto itr=0; ; itr++){
		#pragma omp parallel for num_threads(omp_get_max_threads()/2)
		for(unsignedInteger i=0; i<u.size(); i++){
			for(unsignedInteger j=0; j<u[i].size(); ++j){
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
			for(unsignedInteger i=0; i<u.size(); i+=10){
				for(unsignedInteger j=0; j<u[i].size(); j+=10){
					outputfile1 << std::setprecision(12) << i * dx << " " << j * dy << " " << u[i][j].x << " " << u[i][j].y << "\n";
				}
			}
			outputfile1.close();
			//
			std::ofstream outputfile2("cav-uy" + num_str + ".dat");
			for(unsignedInteger i=0; i<u.size(); i++){
				outputfile2 << std::setprecision(14) << i * dx << " " << u[Nx/2][i].x << "\n";
			}
			outputfile2.close();
			//
			std::ofstream outputfile3("cav-vx" + num_str + ".dat");
			for(unsignedInteger i=0; i<u.size(); i++){
				outputfile3 << std::setprecision(14) << i * dx << " " << u[i][Ny/2].y << "\n";
			}
			outputfile3.close();
			//
			std::ofstream outputfile4("cav-rho" + num_str + ".dat");
			for(unsignedInteger i=0; i<a.size(); i+=2){
				for(unsignedInteger j=0; j<a[i].size(); j+=2){
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
			fprintf(gp, "set lmargin screen 0.05\n");
			fprintf(gp, "set rmargin screen 0.45\n");
			fprintf(gp, "set tmargin screen 0.475\n");
			fprintf(gp, "set bmargin screen 0.025\n");
			fprintf(gp, "set title\n");
			fprintf(gp, "set xr [0:1.0]\n");
			fprintf(gp, "set yr [0:1.0]\n");
			//fprintf(gp, "set zr [0.998:1.002]\n");
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
