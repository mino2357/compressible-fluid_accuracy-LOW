#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <chrono>
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

constexpr Integer Nx = 1003;
constexpr Integer Ny = 1003;
constexpr Integer INTV = 12500;
constexpr Float   Lx = 1.0;
constexpr Float   Ly = 1.0;
constexpr Float   dt = 4.0e-6;
constexpr Float   dx = Lx / (Nx - 1);
constexpr Float   dy = Ly / (Ny - 1);
constexpr Float   mu = 1.0 / 100000.0;
constexpr Float  n_x = 1.0;
constexpr Float  e_x = 0.0;
constexpr Float  w_x = 0.0;
constexpr Float  s_x = 0.0;
constexpr Float  n_y = 0.0;
constexpr Float  e_y = 0.0;
constexpr Float  w_y = 0.0;
constexpr Float  s_y = 0.0;

// H. Iijima, H. Hotta, S. Imada "Semi-conservative reduced speed of sound technique for low Mach number flows with large density variations"
// ref. https://arxiv.org/abs/1812.04135
constexpr Float xi = 2.0;

inline Float stateEq(Float a){
	return 10.0 * a;
}

inline void initScalar2D(Scalar2D& a){
	a.resize(Nx+1);
	unsignedInteger j=0;
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<a.size(); ++i){
		a[i].resize(Ny+1);
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
	for(unsignedInteger i=1; i<v.size()-1; ++i){
		v[i][0].x = 0.0;
		v[i][0].y = 0.0;
		v[i][v[i].size()-1].x = 1.0;
		v[i][v[i].size()-1].y = 0.0;
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger j=1; j<v[0].size()-1; ++j){
		v[0][j].x = 0.0;
		v[0][j].y = 0.0;
		v[v.size()-1][j].x = 0.0;
		v[v.size()-1][j].y = 0.0;
	}
}

inline void diffusionVector(Scalar2D& a, Vector2D& vec, Vector2D& tmp_vec){
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
			auto a_ave = (a[i][j] + a[i+1][j] + a[i][j+1] + a[i+1][j+1]) / 4.0;
			vec[i][j].x +=
						+ dt * mu * (tmp_vec[i+1][j].x - 2.0 * tmp_vec[i][j].x + tmp_vec[i-1][j].x) / (dx * dx) / a_ave
						+ dt * mu * (tmp_vec[i][j+1].x - 2.0 * tmp_vec[i][j].x + tmp_vec[i][j-1].x) / (dy * dy) / a_ave;
			vec[i][j].y +=
						+ dt * mu * (tmp_vec[i+1][j].y - 2.0 * tmp_vec[i][j].y + tmp_vec[i-1][j].y) / (dx * dx) / a_ave
						+ dt * mu * (tmp_vec[i][j+1].y - 2.0 * tmp_vec[i][j].y + tmp_vec[i][j-1].y) / (dy * dy) / a_ave;
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
	for(j=1; j<vec.size()-1; ++j){
		vec[0][j].x = w_x;
		vec[0][j].y = w_y;
		vec[vec.size()-1][j].x = e_x;
		vec[vec.size()-1][j].y = e_y;
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
						- dt * (vec[i][j].x * (vec[i+1][j].x - vec[i-1][j].x)/(2.0 * dx)
						- std::abs(vec[i][j].x) / 2.0 * (vec[i+1][j].x - 2.0 * vec[i][j].x + vec[i-1][j].x) / dx)
						- dt * (vec[i][j].y * (vec[i][j+1].x - vec[i][j-1].x)/(2.0 * dy)
						- std::abs(vec[i][j].y) / 2.0 * (vec[i][j+1].x - 2.0 * vec[i][j].x + vec[i][j-1].x) / dy);
			tmp_vec[i][j].y = vec[i][j].y
						- dt * (vec[i][j].x * (vec[i+1][j].y - vec[i-1][j].y)/(2.0 * dx)
						- std::abs(vec[i][j].x) / 2.0 * (vec[i+1][j].y - 2.0 * vec[i][j].y + vec[i-1][j].y) / dx)
						- dt * (vec[i][j].y * (vec[i][j+1].y - vec[i][j-1].y)/(2.0 * dy)
						- std::abs(vec[i][j].y) / 2.0 * (vec[i][j+1].y - 2.0 * vec[i][j].y + vec[i][j-1].y) / dy);
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
	for(j=0; j<vec.size(); ++j){
		vec[0][j].x = w_x;
		vec[0][j].y = w_y;
		vec[vec.size()-1][j].x = e_x;
		vec[vec.size()-1][j].y = e_y;
	}
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(j=0; j<vec[i].size(); ++j){
			vec[i][j].x = tmp_vec[i][j].x;
			vec[i][j].y = tmp_vec[i][j].y;
		}
	}
}

inline void gradient(const Scalar2D& a, Vector2D& vec){
	unsignedInteger j=0;
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<vec.size()-1; ++i){
		for(j=1; j<vec[i].size()-1; ++ j){
			auto left_bottom  = a[i][j];
			auto right_bottom = a[i+1][j];
			auto left_top     = a[i][j+1];
			auto right_top    = a[i+1][j+1];
			auto a_ave = (left_bottom + right_bottom + left_top + right_top) / 4.0;
			vec[i][j].x += - dt * 0.5 * ((stateEq(right_bottom) - stateEq(left_bottom)) / dx + (stateEq(right_top) - stateEq(left_top)) / dx) / a_ave;
			vec[i][j].y += - dt * 0.5 * ((stateEq(left_top) - stateEq(left_bottom)) / dy + (stateEq(right_top) - stateEq(right_bottom)) / dy) / a_ave;
		}
	}
}

inline Float pm(Float a, Float b, Float c, Float d) {
	Float x = (std::abs(a) > std::abs(b)) ? a : b;
	Float y = (std::abs(c) > std::abs(d)) ? c : d;
	Float z = (std::abs(x) > std::abs(y)) ? x : y;
	return z;
}

inline void transportContinuity(Scalar2D& a, Vector2D& vec, Scalar2D& tmp_a, Vector2D& tmp_vec){
	unsignedInteger j=0;
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(j=0; j<vec[i].size(); ++j){
			tmp_vec[i][j].x = vec[i][j].x;
			tmp_vec[i][j].y = vec[i][j].y;
		}
	}
	#pragma omp parallel for private(j) num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=0; i<a.size(); i++){
		for(j=0; j<a[i].size(); ++j){
			tmp_a[i][j] = a[i][j];
		}
	}
	#pragma omp parallel for num_threads(omp_get_max_threads()/2)
	for(unsignedInteger i=1; i<a.size()-1; ++i){
		for(j=1; j<a[i].size()-1; ++j){
			// 合っているかわからない
			auto vec_max_x = pm(vec[i-1][j-1].x, vec[i][j-1].x, vec[i-1][j].x, vec[i][j].x);
			auto vec_max_y = pm(vec[i-1][j-1].y, vec[i][j-1].y, vec[i-1][j].y, vec[i][j].y);
			a[i][j] +=
					- dt * tmp_a[i][j] * ((tmp_vec[i][j].x - tmp_vec[i-1][j].x) / dx + (tmp_vec[i][j-1].x - tmp_vec[i-1][j-1].x) / dx) / (2.0 * xi * xi)
					- dt * tmp_a[i][j] * ((tmp_vec[i][j].y - tmp_vec[i][j-1].y) / dy + (tmp_vec[i-1][j].y - tmp_vec[i-1][j-1].y) / dy) / (2.0 * xi * xi)
					- dt * (vec_max_x * (tmp_a[i+1][j] - tmp_a[i-1][j]) / (2.0 * dx)
					- std::abs(vec_max_x) * (tmp_a[i+1][j] - 2.0 * tmp_a[i][j] + tmp_a[i-1][j]) / (2.0 * dx)) / (xi * xi)
					- dt * (vec_max_y * (tmp_a[i][j+1] - tmp_a[i][j-1]) / (2.0 * dy)
					- std::abs(vec_max_y) * (tmp_a[i][j+1] - 2.0 * tmp_a[i][j] + tmp_a[i][j-1]) / (2.0 * dy)) / (xi * xi);
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
	auto scale = 4.0 / Nx;
	std::string str1 = "set output 'fluid-";
	std::string str2 = ".png'\n";
	std::FILE *gp = popen("gnuplot -persist", "w" );
	fprintf(gp, "set title'Navier–Stokes equations. Re=100000.' \n");
	fprintf(gp, "set contour\n");
	fprintf(gp, "set xr [0:%f]\n", Lx);
	fprintf(gp, "set yr [0:%f]\n", Ly);
	fprintf(gp, "set terminal png\n");
	//fprintf(gp, "set palette defined(0'#aaaaaa',0.8'#00008b',1.8'#2ca9e1',3'#008000',4.2'#ffff00',5'#eb6101',5.5'#8b0000')\n");
	fprintf(gp, "set palette defined(0'#aaaaaa',0.2'#00008b',0.4'#2ca9e1',0.6'#008000',0.8'#ffff00',1.0'#eb6101', 4.0'#8b0000')\n");
	//fprintf(gp, "set term pngcairo size 7680, 4320\n");
	fprintf(gp, "set term pngcairo size 3840, 2160\n");
	fprintf(gp, "set size square\n");
	fprintf(gp, "set grid\n");

	unsignedInteger j=0;
	std::chrono::system_clock::time_point  start, end;
	for(auto itr=0; ; itr++){
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
			fprintf(gp, "%s", (str1 + num_str + str2).c_str());
			fprintf(gp, "plot '-' with vectors lw 1 lc palette notitle\n");
			//#pragma omp parallel for num_threads(omp_get_max_threads()/2)
			for(unsignedInteger i=0; i<u.size(); i=i+3){
				for(j=0; j<u[i].size(); j=j+3){
					auto eps = 1.0e-12;
					auto norm = std::sqrt(u[i][j].x * u[i][j].x + u[i][j].y * u[i][j].y);
					fprintf(gp, "%f %f %f %f %f\n", i * dx, j * dy, scale * u[i][j].x / (norm + eps), scale * u[i][j].y / (norm + eps), norm);
					//fprintf(gp, "%f %f %f %f %f\n", i * dx, j * dy, 0.25*u[i][j].x, 0.25*u[i][j].y, a[i][j]);
				}
			}
			fprintf(gp, "e\n");
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
