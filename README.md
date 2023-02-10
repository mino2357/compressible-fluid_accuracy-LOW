# compressible-fluid

Navier-Stokes equations. Simple fully explicit method.

**Navier-Stokes equations**

$$\frac{\partial \rho \bm{u}}{\partial t} + \nabla \cdot (\rho \bm{u} \otimes \bm{u}) = - \nabla p + \mu \Delta \bm{u}$$

**Equation of continuity**

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \bm{u}) = 0$$

**Equation of state**

Assume a barotropic fluid.

$$\rho = \rho(p)$$

## Discretization method

- FDM
- structured grid
- Advection term
  - 1st order upwind scheme
- diffusion term
  - 2nd order central scheme
- collocated grid
- fully explicit

## TODO

- [ ] collocated grid vs. staggered grid.
- [ ] Stability Analysis.
- [ ] LES, RANS.

## Dependency and how to run

1. Install C++ compiler, [gnuplot](http://www.gnuplot.info/), [OpenMP](https://www.openmp.org/).

2. Set the number of threads in your environment, when you use OpenMP.

e.g. (28C56T)
```C++
#pragma omp parallel for num_threads(28)
```

3. `make run`

## Visualization

[![box in cavity](http://img.youtube.com/vi/llIm1qXyo4s/0.jpg)](https://www.youtube.com/watch?v=llIm1qXyo4s)