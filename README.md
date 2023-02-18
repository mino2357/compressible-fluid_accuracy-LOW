# compressible-fluid

Navier-Stokes equations. Simple fully explicit method.

**Navier-Stokes equations**

$$\frac{\partial \rho \boldsymbol{u}}{\partial t} + \nabla \cdot (\rho \boldsymbol{u} \otimes \boldsymbol{u}) = - \nabla p + \mu \Delta \boldsymbol{u}$$

**Equation of continuity**

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \boldsymbol{u}) = 0$$

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

## How to run

1. Install C++ compiler, [gnuplot](http://www.gnuplot.info/), [OpenMP](https://www.openmp.org/).

2. `make run`

## Visualization

[![box in cavity](http://img.youtube.com/vi/llIm1qXyo4s/0.jpg)](https://www.youtube.com/watch?v=llIm1qXyo4s)

## Reference (incompressible)

Ghia, U. K. N. G., Kirti N. Ghia, and C. T. Shin. "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method." Journal of computational physics 48.3 (1982): 387-411.

```
1.0000 1.00000
0.9766 0.47221
0.9688 0.47783
0.9609 0.48070
0.9531 0.47804
0.8516 0.34635
0.7344 0.20673
0.6172 0.08344
0.5000 0.03111
0.4531 -0.07540
0.2813 -0.23186
0.1719 -0.32709
0.1016 -0.38000
0.0703 -0.41657
0.0625 -0.42537
0.0547 -0.42735
0.0000 0.00000
```