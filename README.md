# paper-cusum-detection

This repository contains Julia code for the paper on **Outage Identification from Electricity Market Data:
Quickest Change Detection Approach**. The code allows replicating the results, including critical region extraction, CuSum statistic construction, and detection performance metrics.

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
