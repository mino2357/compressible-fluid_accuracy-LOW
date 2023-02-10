#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>

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

constexpr Integer Nx = 1025;
constexpr Integer Ny = 1025;
constexpr Integer INTV = 5000; // outpu interval
constexpr Float   Lx = 1.0;
constexpr Float   Ly = 1.0;
constexpr Float   dt = 2.0e-5;
constexpr Float   dx = Lx / (Nx - 1);
constexpr Float   dy = Ly / (Ny - 1);
constexpr Float   mu = 1.0 / 10000.0; // viscosity
constexpr Float   PI = 3.1415926535897932384626433832795028841971;

// H. Iijima, H. Hotta, S. Imada "Semi-conservative reduced speed of sound technique for low Mach number flows with large density variations"
// ref. https://arxiv.org/abs/1812.04135
constexpr Float xi = 2.0;

inline Float stateEq(Float a){
	return 10.0 * a;
}

inline void initScalar2D(Scalar2D& a){
	a.resize(Nx);
	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=0; i<a.size(); ++i){
		a[i].resize(Ny);
		for(unsignedInteger j=0; j<a[i].size(); ++j){
			a[i][j] = 1.0;
		}
	}
}

inline void initVector2D(Vector2D& v){
	v.resize(Nx);
	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=0; i<v.size(); ++i){
		v[i].resize(Ny);
		for(unsignedInteger j=0; j<v[i].size(); ++j){
			v[i][j].x = 0.0;
			v[i][j].y = 0.0;
		}
	}
	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=0; i<v.size(); ++i){
		v[i][0].x = 0.0;
		v[i][0].y = 0.0;
		v[i][v[i].size()-1].x = 1.0;
		v[i][v[i].size()-1].y = 0.0;
	}
	#pragma omp parallel for num_threads(28)
	for(unsignedInteger j=0; j<v[0].size(); ++j){
		v[0][j].x = 0.0;
		v[0][j].y = 0.0;
		v[v.size()-1][j].x = 0.0;
		v[v.size()-1][j].y = 0.0;
	}
}

inline void diffusionVector(Scalar2D& a, Vector2D& vec, Vector2D& tmp_vec){
	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(unsignedInteger j=0; j<vec[i].size(); ++j){
			tmp_vec[i][j].x = vec[i][j].x;
			tmp_vec[i][j].y = vec[i][j].y;
		}
	}

	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=1; i<vec.size()-1; i++){
		for(unsignedInteger j=1; j<vec[i].size()-1; ++j){
			vec[i][j].x +=
						+ dt * mu * (tmp_vec[i + 1][j].x - 2.0 * tmp_vec[i][j].x + tmp_vec[i - 1][j].x) / (dx * dx) / a[i][j]
						+ dt * mu * (tmp_vec[i][j + 1].x - 2.0 * tmp_vec[i][j].x + tmp_vec[i][j - 1].x) / (dy * dy) / a[i][j];
			vec[i][j].y +=
						+ dt * mu * (tmp_vec[i + 1][j].y - 2.0 * tmp_vec[i][j].y + tmp_vec[i - 1][j].y) / (dx * dx) / a[i][j]
						+ dt * mu * (tmp_vec[i][j + 1].y - 2.0 * tmp_vec[i][j].y + tmp_vec[i][j - 1].y) / (dy * dy) / a[i][j];
		}
	}
}

inline void advection(Vector2D& vec, Vector2D& tmp_vec){
	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(unsignedInteger j=0; j<vec[i].size(); ++j){
			tmp_vec[i][j].x = vec[i][j].x;
			tmp_vec[i][j].y = vec[i][j].y;
		}
	}

	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=1; i<vec.size()-1; i++){
		for(unsignedInteger j=1; j<vec[i].size()-1; ++j){
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

	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=1; i<vec.size() - 1; i++){
		for(unsignedInteger j=1; j<vec[i].size() - 1; ++j){
			vec[i][j].x = tmp_vec[i][j].x;
			vec[i][j].y = tmp_vec[i][j].y;
		}
	}
}

inline void gradient(Scalar2D& a, Vector2D& vec){
	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=1; i<a.size()-1; ++i){
		for(unsignedInteger j=1; j<a[i].size()-1; ++ j){
			vec[i][j].x += - dt * (stateEq(a[i+1][j]) - stateEq(a[i-1][j])) / dx / a[i][j];
			vec[i][j].y += - dt * (stateEq(a[i][j+1]) - stateEq(a[i][j-1])) / dy / a[i][j];
		}
	}
}

inline void transportContinuity(Scalar2D& a, Vector2D& vec, Scalar2D& tmp_a, Vector2D& tmp_vec){
	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=0; i<vec.size(); i++){
		for(unsignedInteger j=0; j<vec[i].size(); ++j){
			tmp_vec[i][j].x = vec[i][j].x;
			tmp_vec[i][j].y = vec[i][j].y;
			tmp_a[i][j] = a[i][j];
		}
	}

	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=1; i<a.size()-1; ++i){
		for(unsignedInteger j=1; j<a[i].size()-1; ++ j){
			a[i][j] +=
					- dt * tmp_a[i][j] * (tmp_vec[i+1][j].x - tmp_vec[i-1][j].x) / (2.0 * dx) / (xi * xi)
					- dt * tmp_a[i][j] * (tmp_vec[i][j+1].y - tmp_vec[i][j-1].y) / (2.0 * dy) / (xi * xi)
					- dt * (tmp_vec[i][j].x * (tmp_a[i+1][j] - tmp_a[i-1][j]) / (2.0 * dx)
					- std::abs(tmp_vec[i][j].x) * (tmp_a[i+1][j] - 2.0 * tmp_a[i][j] + tmp_a[i-1][j]) / (2.0 * dx)) / (xi * xi)
					- dt * (tmp_vec[i][j].y * (tmp_a[i][j+1] - tmp_a[i][j-1]) / (2.0 * dy)
					- std::abs(tmp_vec[i][j].y) * (tmp_a[i][j+1] - 2.0 * tmp_a[i][j] + tmp_a[i][j-1]) / (2.0 * dy)) / (xi * xi);
		}
	}

	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=0; i<a.size(); ++i){
		a[i][0] = a[i][1];
		a[i][a[i].size()-1] = a[i][a[i].size()-2];
	}

	#pragma omp parallel for num_threads(28)
	for(unsignedInteger j=0; j<a[0].size(); ++j){
		a[0][j] = a[1][j];
		a[a.size()-1][j] = a[a.size()-2][j];
	}
}

inline void obstacle(Vector2D& vec){
	unsignedInteger begin1 = 0.25 * vec.size();
	unsignedInteger end1 = 0.75 * vec.size();
	unsignedInteger begin2 = 0.25 * vec[0].size();
	unsignedInteger end2 = 0.75 * vec[0].size();
	
	#pragma omp parallel for num_threads(28)
	for(unsignedInteger i=0; i<vec.size(); ++i){
		for(unsignedInteger j=0; j<vec[i].size(); ++j){
			if((begin1 < i && i < end1) && (begin2 < j && j < end2)){
				vec[i][j].x = 0.0;
				vec[i][j].y = 0.0;
			}
		}
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
	auto scale = 1.0e-3;
	std::string str1 = "set output 'fluid-";
	std::string str2 = ".png'\n";

	std::FILE *gp = popen("gnuplot -persist", "w" );
	fprintf(gp, "set title'Navier–Stokes equations. Re=10000.' \n");
	fprintf(gp, "set contour\n");
	fprintf(gp, "set xr [0:%f]\n", Lx);
	fprintf(gp, "set yr [0:%f]\n", Ly);
	fprintf(gp, "set terminal png\n");
	//fprintf(gp, "set palette defined(0'#aaaaaa',0.8'#00008b',1.8'#2ca9e1',3'#008000',4.2'#ffff00',5'#eb6101',5.5'#8b0000')\n");
	fprintf(gp, "set palette defined(0'#aaaaaa',0.4'#00008b',0.8'#2ca9e1',1.2'#008000',1.6'#ffff00',2.0'#eb6101',5.5'#8b0000')\n");
	fprintf(gp, "set term pngcairo size 7680, 4320\n");
	fprintf(gp, "set size square\n");
	fprintf(gp, "set grid\n");

	for(auto i=0; ; i++){
		// When using OpenMP, writing u_old = u is slow.
		#pragma omp parallel for num_threads(28)
		for(unsignedInteger i=0; i<u.size(); i++){
			for(unsignedInteger j=0; j<u[i].size(); ++j){
				u_old[i][j].x = u[i][j].x;
				u_old[i][j].y = u[i][j].y;
			}
		}
		diffusionVector(a, u, tmp_vec);
		obstacle(u);
		gradient(a, u);
		obstacle(u);
		advection(u, tmp_vec);
		obstacle(u);
		transportContinuity(a, u_old, tmp_a, tmp_vec);
		obstacle(u);
		if(i%INTV == 0){
			std::ostringstream oss;
			oss.setf(std::ios::right);
			oss.fill('0');
			oss.width(8);
			oss << i / INTV;
			std::string num_str = oss.str();
			std::cout << num_str << std::endl;

			fprintf(gp, (str1 + num_str + str2).c_str());
			fprintf(gp, "plot '-' with vectors lw 1 lc palette notitle\n");
			#pragma omp parallel for num_threads(28)
			for(unsignedInteger i=0; i<a.size(); ++i){
				for(unsignedInteger j=0; j<a[i].size(); ++ j){
					auto eps = 1.0e-12;
					auto norm = std::sqrt(u[i][j].x * u[i][j].x + u[i][j].y * u[i][j].y);
					fprintf(gp, "%f %f %f %f %f\n", i * dx, j * dy, scale * u[i][j].x / (norm + eps), scale * u[i][j].y / (norm + eps), norm);
					//fprintf(gp, "%f %f %f %f %f\n", i * dx, j * dy, 0.25*u[i][j].x, 0.25*u[i][j].y, std::sqrt(u[i][j].x*u[i][j].x+u[i][j].y*u[i][j].y));
				}
			}
			fprintf(gp, "e\n");
			fflush(gp);
		}
	}
	pclose(gp);
}