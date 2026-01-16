# paper-cusum-detection

This repository contains Julia code for the paper on **CUSUM-based outage detection in power systems**. The code allows replicating the results, including critical region extraction, demand trajectory simulations, and detection performance metrics.

## Requirements

- Julia 1.10+
- Packages used:
  - `PowerModels`
  - `JuMP`
  - `Ipopt`
  - `LinearAlgebra`
  - `CDDLib`
  - `Polyhedra`
  - `LazySets`
  - `Colors`
  - `Distributions`
  - `Random`
  - `ProximalOperators`
  - `DataFrames`
  - `CSV`
  - `Plots`
  - `LaTeXStrings`

You can install the packages via the Julia REPL:

```julia
using Pkg
Pkg.add([
    "PowerModels", "JuMP", "Ipopt", "LinearAlgebra",
    "CDDLib", "Polyhedra", "LazySets",
    "Colors", "Distributions", "Random",
    "ProximalOperators", "DataFrames", "CSV",
    "Plots", "LaTeXStrings"
])
