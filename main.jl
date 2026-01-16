using PowerModels, JuMP, Ipopt, LinearAlgebra
using CDDLib, Polyhedra, LazySets
using Colors
using Distributions
using Random
using ProximalOperators
using DataFrames, CSV
using Plots, Plots.PlotMeasures, LaTeXStrings
red = RGB(186/255, 40/255, 58/255)
# --- Paper-ready font sizes ---
const FS_TICK   = 22
const FS_LABEL  = 22
const FS_LEGEND = 22
const FS_TITLE  = 22
include("aux_fun.jl")

function solve_OPF(net; ξ=[0;0])
    # DC-OPF definition
    model = Model(optimizer_with_attributes(Ipopt.Optimizer))
    JuMP.set_silent(model)
    # model variables
    @variable(model, p[1:net[:G]])
    @variable(model, l[1:net[:D]])
    # model objective
    @objective(model, Min, p'*diagm(net[:c2])*p + net[:c1]'p + sum(net[:c0]) + l'*diagm(net[:s2])*l + net[:s1]'*l)
    # OPF equations
    @constraint(model, λ, ones(net[:N])'*(net[:M_p]*p .- net[:M_l]*(net[:l]  .- l) .- net[:M_ξ]*ξ) .== 0)
    @constraint(model, μ̅, net[:f̅] .>=  net[:T]*(net[:M_p]*p .- net[:M_l]*(net[:l] .- l) .- net[:M_ξ]*ξ))
    @constraint(model, μ̲, net[:f̅] .>= -net[:T]*(net[:M_p]*p .- net[:M_l]*(net[:l] .- l) .- net[:M_ξ]*ξ))
    @constraint(model, φ̅, -p .>= -net[:p̅])
    @constraint(model, φ̲, p .>= net[:p̲])
    @constraint(model, κ̅,  l .>= 0)
    @constraint(model, κ̲, -l .>= -net[:l] - net[:M_l]'*net[:M_ξ]*ξ)
    @info("done build Det-OPF")
    # solve model
    optimize!(model)
    @info("DC-OPF terminates with status $(termination_status(model))")
    sol = Dict(:status => termination_status(model),
                :model => model,
                :obj => JuMP.objective_value(model),
                :p => JuMP.value.(p),
                :l => JuMP.value.(l),
                :CPUtime => solve_time(model),
                :λ => JuMP.dual.(λ), :μ̅ => JuMP.dual.(μ̅), :μ̲ => JuMP.dual.(μ̲), :φ̅ => JuMP.dual.(φ̅), :φ̲ => JuMP.dual.(φ̲), :κ̅ => JuMP.dual.(κ̅), :κ̅ => JuMP.dual.(κ̅),
                :π => JuMP.dual.(λ) .* ones(net[:N]) .- net[:T]'*(JuMP.dual.(μ̅) .- JuMP.dual.(μ̲)),
                :f => net[:T]*(net[:M_p]*JuMP.value.(p) .- net[:M_l]*(net[:l] .- JuMP.value.(l))))
    return sol
end

function QP_standard_data(net)
    Q = blockdiag(diagm(net[:c2]) .* 2, diagm(net[:s2]) .* 2)
    q = [net[:c1];net[:s1]]

    A = [-ones(net[:N])'*net[:M_p] -ones(net[:N])'*net[:M_l]; net[:T]*net[:M_p] net[:T]*net[:M_l]; -net[:T]*net[:M_p] -net[:T]*net[:M_l]; blockdiag(I(net[:G]),I(net[:D]));-blockdiag(I(net[:G]),I(net[:D]))]
    B = [-ones(net[:N])'*net[:M_ξ]; net[:T]*net[:M_ξ]; -net[:T]*net[:M_ξ]; zeros(net[:G], 2); net[:M_l]'*net[:M_ξ]; zeros(net[:G], 2); zeros(net[:D], 2)]
    b = [-ones(net[:N])'*net[:M_l]*net[:l]; net[:T]*net[:M_l]*net[:l] .+ net[:f̅];-net[:T]*net[:M_l]*net[:l] .+ net[:f̅];net[:p̅];net[:l];-net[:p̲];zeros(net[:D])]
    return Dict(:Q => Q, :q => q, :A => A, :B => B, :b => b, :Ã => [], :B̃ => [], :b̃ => [], :A̅ => [], :B̅ => [], :b̅ => [], :Λ̃ => [], :K => [], :num_act => 0, :num_ina => 0, :net => net)
end

function solve_QP(data;ξ=[0,0])
    # DC-OPF definition
    model = Model(optimizer_with_attributes(Ipopt.Optimizer))
    JuMP.set_silent(model)
    n = length(data[:q])
    @variable(model, x[1:n])
    @objective(model, Min, 1/2*x'*data[:Q]*x + data[:q]'x)
    @constraint(model, μ, -data[:A]*x .>= - data[:B]*ξ - data[:b])
    # solve model
    optimize!(model)
    # @info("QP terminates with status $(termination_status(model))")


    # inspect active/inactive constraints 
    active_con = findall(x->x> 1e-7,JuMP.dual.(μ)); data[:num_act] = length(active_con);
    inacti_con = findall(x->x<=1e-7,JuMP.dual.(μ)); data[:num_ina] = length(inacti_con);

    qp_data = deepcopy(data)
    qp_data[:Ã] = data[:A][active_con,:]; qp_data[:A̅] = data[:A][inacti_con,:]
    qp_data[:B̃] = data[:B][active_con,:]; qp_data[:B̅] = data[:B][inacti_con,:]
    qp_data[:b̃] = data[:b][active_con,:]; qp_data[:b̅] = data[:b][inacti_con,:]

    qp_data[:D] = -inv(qp_data[:Ã]*inv(qp_data[:Q])*qp_data[:Ã]')*qp_data[:B̃]
    qp_data[:d] = vec(-inv(qp_data[:Ã]*inv(qp_data[:Q])*qp_data[:Ã]')*(qp_data[:b̃] .+ qp_data[:Ã]*inv(qp_data[:Q])*qp_data[:q]))

    qp_data[:P] = -inv(qp_data[:Q])*qp_data[:Ã]'*qp_data[:D]
    qp_data[:p] = vec(-inv(qp_data[:Q])*qp_data[:q] - inv(qp_data[:Q])*qp_data[:Ã]'*qp_data[:d])


    Λ = [ones(data[:net][:N]) -data[:net][:T]' data[:net][:T]']
    Λ = [Λ zeros(data[:net][:N],length(μ) - size(Λ,2))]
    Λ̃ = Λ[:,active_con]
    qp_data[:Λ̃] = Λ̃

        
    sol = Dict(:status => termination_status(model),
                :model => model,
                :obj => JuMP.objective_value(model),
                :x => JuMP.value.(x),
                :μ => JuMP.dual.(μ),
                :μ̃ => JuMP.dual.(μ)[active_con])

    return sol, qp_data
end

function mypoly(qp_data)
    A = [qp_data[:A̅]*qp_data[:P] - qp_data[:B̅];-qp_data[:D]]
    b = [qp_data[:b̅] - qp_data[:A̅]*qp_data[:p];qp_data[:d]]

    # add limits on random demand 
    A_add = [I(2);
    -I(2)]
    b_add = [set[:load_variation].*ones(2);set[:load_variation].*ones(2)]
    A = vcat(A,A_add)
    b = vcat(b,b_add)


    # round very small entries 
    A = round.(A,digits=10)
    b = round.(b,digits=10)

    # remove zero rows from (A, b)
    keep = [!iszero(norm(A[i, :])) for i in 1:size(A,1)]
    A = A[keep, :]
    b = b[keep]


    return Dict(:A=>A, :b=>vec(b))
end

function in_polyhedron(A::AbstractMatrix, b::AbstractVector, x::AbstractVector; atol=1e-18)
    return all(A * x .<= b .+ atol)
end

function extract_CR(net_data)
    N = 3000
    counter = 0
    CR = Dict()
    for i in 1:N
        Random.seed!(i)
        ξ = rand(Uniform(-set[:load_variation],set[:load_variation]),2)
        Random.seed!()
        if i == 1 
            counter += 1
            data = QP_standard_data(net_data)
            sol_qp, data = solve_QP(data;ξ=ξ)
            pol = mypoly(data)

            CR[i] = Dict(:A=>pol[:A], :b=>pol[:b], :D => data[:D], :d => data[:d], :Λ̃ => data[:Λ̃])
        else
            flag = 0
            for j in collect(keys(CR))
                if in_polyhedron(CR[j][:A],CR[j][:b],ξ) == true
                    flag = 1
                end
            end
            if flag == 0
                counter += 1
                data = QP_standard_data(net_data)
                sol_qp, data = solve_QP(data;ξ=ξ)
                pol = mypoly(data)

                CR[counter] = Dict(:A=>pol[:A], :b=>pol[:b], :D => data[:D], :d => data[:d], :Λ̃ => data[:Λ̃])
            end
        end
    end
    return CR
end

# experiment settings 
set = Dict(:load_variation => 200, :σ_Δξ => 8, :outage_line => 3)

# Load network data from PowerModels
cd(dirname(@__FILE__))
PowerModels.silence()
caseID="data/pglib_opf_case5_pjm.m"
# nominal optimization data 
net_nom = load_network_data(caseID; reduced_trans_cap = true)
# solve DC-OPF in a standard QP form 
data_nom = QP_standard_data(net_nom)
sol_qp_nom, data_nom = solve_QP(data_nom)
# alternative optimization data: dictionary with DC-OPF data for each line outage
net_alt = Dict()
data_qp_alt = Dict()
sol_qp_alt = Dict()
for i in 1:net_nom[:E]
    net = load_network_data(caseID; line_outaged = "$i", reduced_trans_cap = true)
    # solve DC-OPF in a standard QP form 
    data = QP_standard_data(net)
    sol_qp, data = solve_QP(data)
    # save to dictionaries 
    net_alt[i] = net
    data_qp_alt[i] = data
    sol_qp_alt[i] = sol_qp_alt
end

#######
using Plots, CDDLib, LazySets, Distributions, Random, Statistics, LaTeXStrings
# -----------------------------
# Extract critical regions
# -----------------------------
CR_nom = extract_CR(net_nom)
CR_alt = Dict()
for i in 1:net_nom[:E]
    CRs = extract_CR(net_alt[i])
    CR_alt[i] = CRs
end
CR_alt = sort(CR_alt)

# -----------------------------
# Function to plot critical regions
# -----------------------------
function plot_CR(CR_list; title_text="", palette_colors=nothing)
    plt = plot(
        frame = :box,
        aspect_ratio=1,
        grid=false,
        xlabel=L"{\xi}_1"*" [MW]",
        ylabel=L"{\xi}_2"*" [MW]",
        title=title_text,
        legend=false,
        # Set font sizes for paper
        # Increase fonts for paper
        titlefont = font(18),
        guidefont = font(18),
        tickfont  = font(18),

    # Increase spacing so labels don’t crowd
        xlabelpad = 10,
        ylabelpad = 12,
        # --- Spacing between ticks and axes
        left_margin   = 5mm,
        bottom_margin = 5mm,
        right_margin  = 5mm,
        top_margin    = 5mm,

        #xticks = :auto,
        #yticks = :auto
    )
    
    if palette_colors === nothing
        palette_colors = cgrad(:blues, length(CR_list), categorical=true)
    end
    
    for i in 1:length(CR_list)
        hr = hrep(CR_list[i][:A], CR_list[i][:b])
        poly = polyhedron(hr, CDDLib.Library())
        plot!(poly, color=palette_colors[i], alpha=0.25, label="CR $i")
    end
    
    # Set consistent limits
    lim = set[:load_variation]
    plot!(xlims=(-lim, lim), ylims=(-lim, lim))
    return plt
end

# -----------------------------
# Plot nominal and outage CRs
# -----------------------------
palette_nom = cgrad(:blues, length(CR_nom), categorical=true)
plt_nom = plot_CR(CR_nom, title_text="Nominal Operation", palette_colors=palette_nom)

palette_alt = cgrad(:blues, length(CR_alt[set[:outage_line]]), categorical=true)
plt_alt = plot_CR(CR_alt[set[:outage_line]], title_text="Post-outage Operation", palette_colors=palette_alt)

# -----------------------------
# Overlay sample demand trajectories
# -----------------------------
N = 1000
Random.seed!(127)
#Random.seed!()
Σ_Δξ = [set[:σ_Δξ]^2 0; 0 set[:σ_Δξ]^2]
Δξ = rand(MultivariateNormal(zeros(2), Σ_Δξ), N)

function add_demand_trajectories!(plt; color=:blue)
    demand = zeros(2, N)
    for i in 2:N
        demand[:, i] = demand[:, i-1] .+ Δξ[:, i]
    end

    # Smooth trajectories
    d1_smooth = movavg(demand[1, :], 3)
    d2_smooth = movavg(demand[2, :], 3)

    plot!(plt, d1_smooth, d2_smooth, lw=2, color=color, label=L"\mathbf{\xi}")
end

add_demand_trajectories!(plt_nom, color=:blue)
add_demand_trajectories!(plt_alt, color=:blue)
######
# -----------------------------
# Combine plots for paper figure
# -----------------------------
final_CR_plot = plot(         
    plt_nom,
    plt_alt,         
    layout = (1, 2),
    size = (1000, 500),
    leftmargin   = 5mm,
    rightmargin  = 5mm,
    topmargin    = 5mm,
    bottommargin = 5mm
)

# Save high-resolution PDF suitable for papers
savefig(final_CR_plot, "critical_regions.pdf")

display(final_CR_plot)

#######
### CuSum statistic
function effective_random_indices(ξ::AbstractVector)
    lv = set[:load_variation]
    return findall(k -> abs(ξ[k]) < lv - ξ_BOUND_TOL, eachindex(ξ))
end

function Σ_λ_nominal(cr, ξ_t)
    idx = effective_random_indices(ξ_t)

    if isempty(idx)
        nλ  = size(CR_nom[cr][:Λ̃], 1)
        Σ_λ = 1e-8 * I(nλ)
    else
        D_eff = CR_nom[cr][:D][:, idx]
        Σ_eff = Σ_Δξ[idx, idx]
        ΛD    = CR_nom[cr][:Λ̃] * D_eff
        Σ_λ   = ΛD * Σ_eff * ΛD'
    end

    # PSD projection + εI
    Σ_λ, _ = prox(IndPSD(), Σ_λ)
    #Σ_λ = Σ_λ .+ JITTER * I(size(Σ_λ,1))
    Σ_λ = round.(Σ_λ .+ diagm(ones(size(Σ_λ,1))) .* (eps() .+ 1),digits=6)

    return Σ_λ
end

function Σ_λ_alternative(cr, ξ_t, line)
    idx = effective_random_indices(ξ_t)

    if isempty(idx)
        nλ  = size(CR_alt[line][cr][:Λ̃], 1)
        Σ_λ = 1e-8 * I(nλ)
    else
        D_eff = CR_alt[line][cr][:D][:, idx]
        Σ_eff = Σ_Δξ[idx, idx]
        ΛD    = CR_alt[line][cr][:Λ̃] * D_eff
        Σ_λ   = ΛD * Σ_eff * ΛD'
    end

    # PSD projection + εI
    Σ_λ, _ = prox(IndPSD(), Σ_λ)
    #Σ_λ = Σ_λ .+ JITTER * I(size(Σ_λ,1))
    Σ_λ = round.(Σ_λ .+ diagm(ones(size(Σ_λ,1))) .* (eps() .+ 1),digits=6)

    return Σ_λ
end

function logpdf_x_nom(cr, x, ξ_t)
    Σ_λ = Σ_λ_nominal(cr, ξ_t)
    d   = MultivariateNormal(zeros(size(Σ_λ,1)), Σ_λ)
    return logpdf(d, x)
end

function logpdf_x_alt(cr, x, line, ξ_t)
    Σ_λ = Σ_λ_alternative(cr, ξ_t, line)
    d   = MultivariateNormal(zeros(size(Σ_λ,1)), Σ_λ)
    return logpdf(d, x)
end

l(x, cr_alt, cr_nom, line, ξ_t) =
    logpdf_x_alt(cr_alt, x, line, ξ_t) - logpdf_x_nom(cr_nom, x, ξ_t)

###########
# compute the price change
ξ = zeros(2, N)
for i in 2:N
    ξ[:, i] = ξ[:, i-1] .+ Δξ[:, i]
end
ξ .= clamp.(ξ, -set[:load_variation], set[:load_variation])

change_time = 500
Δλ = zeros(net_nom[:N], N-1)

for i in 2:N
    if i < change_time
        cr_ind_nom = 0
        for j in keys(CR_nom)
            if in_polyhedron(CR_nom[j][:A], CR_nom[j][:b], ξ[:, i])
                cr_ind_nom = j
                break
            end
        end
        λt   = CR_nom[cr_ind_nom][:Λ̃] * (CR_nom[cr_ind_nom][:D] * ξ[:, i]   + CR_nom[cr_ind_nom][:d])
        λt_1 = CR_nom[cr_ind_nom][:Λ̃] * (CR_nom[cr_ind_nom][:D] * ξ[:, i-1] + CR_nom[cr_ind_nom][:d])
        Δλ[:, i-1] = λt .- λt_1
    else
        cr_ind_alt = 0
        for j in keys(CR_alt[set[:outage_line]])
            if in_polyhedron(CR_alt[set[:outage_line]][j][:A], CR_alt[set[:outage_line]][j][:b], ξ[:, i])
                cr_ind_alt = j
                break
            end
        end
        λt   = CR_alt[set[:outage_line]][cr_ind_alt][:Λ̃] * (CR_alt[set[:outage_line]][cr_ind_alt][:D] * ξ[:, i]   + CR_alt[set[:outage_line]][cr_ind_alt][:d])
        λt_1 = CR_alt[set[:outage_line]][cr_ind_alt][:Λ̃] * (CR_alt[set[:outage_line]][cr_ind_alt][:D] * ξ[:, i-1] + CR_alt[set[:outage_line]][cr_ind_alt][:d])
        Δλ[:, i-1] = λt .- λt_1
    end
end

plot(Δλ')

# compute statistics 
w = zeros(length(CR_alt), N)   

for i in 2:N
    Δprice = Δλ[:, i-1]
    cr_ind_nom = 0
    for j in keys(CR_nom)
        if in_polyhedron(CR_nom[j][:A], CR_nom[j][:b], ξ[:, i])
            cr_ind_nom = j
            break
        end
    end

    if cr_ind_nom == 0
        w[:, i] .= w[:, i-1]
        continue
    end

    for a in 1:length(CR_alt)
        cr_ind_alt = 0
        for j in keys(CR_alt[a])
            if in_polyhedron(CR_alt[a][j][:A], CR_alt[a][j][:b], ξ[:, i])
                cr_ind_alt = j
                break
            end
        end

        if cr_ind_alt == 0
            w[a, i] = w[a, i-1]
            continue
        end

        try
            r = l(Δprice, cr_ind_alt, cr_ind_nom, a, ξ[:, i])
            w[a, i] = max(0, w[a, i-1] + r)
        catch
            w[a, i] = w[a, i-1]
        end
    end
end

plot(w')
#########
plo_demand = plot()
demand = zeros(2,N)
palette_load = cgrad(:blues, 2, categorical = true)

for i in 1:N
    if i == 1
        demand[:,i] = net_nom[:l][2:3]
    else
        demand[:,i] = demand[:,i-1] .+ Δξ[:,i-1]
    end
end

for k in 1:2
    plot!(plo_demand,
          1:N, demand[k, :],
          lw = 2,
          label = "load $(k + 1)",
          c = palette_load[k])
end

plot!(plo_demand,
      frame = :box,
      ylabel = "Demand [MW]",
      xlabel = "sample",
      xlims = (1, N),
      guidefont = font(FS_LABEL),
      tickfont  = font(FS_TICK),
      legendfont = font(12))

vline!(plo_demand,
       [change_time],
       line=:dash,
       c=red,
       lw=2,
       label=false)
#############
plo_price = plot()
palette_price = cgrad(:blues, 5)

price = zeros(net_nom[:N],N)
for i in 1:N
    if i == 1
        sol_opf_nom = solve_OPF(net_nom)
        price[:,i] = sol_opf_nom[:π]
    else
        price[:,i] = price[:,i-1] .+ Δλ[:,i-1]
    end
end

for b in 1:net_nom[:N]
    plot!(plo_price,
          1:N, price[b,:],
          lw=2,
          label="bus $b",
          c=palette_price[b])
end

plot!(plo_price,
      frame=:box,
      ylabel="Price [\$/MWh]",
      xlabel="sample",
      xlims=(1, N),
      guidefont = font(FS_LABEL),
      tickfont  = font(FS_TICK),
      legendfont = font(12))

vline!(plo_price,
       [change_time],
       line=:dash,
       c=red,
       lw=2,
       label=false)

##############
plo_cusum = plot(frame=:box,
                 ylabel="CuSuM",
                 xlabel="sample",
                 xlims=(1, N),
                 legend=:topleft,
                 guidefont = font(FS_LABEL),
                 tickfont  = font(FS_TICK),
                 ylims=(0,100),
                 legendfont = font(12))

palette = cgrad(:blues, length(CR_alt), categorical=true)

for i in 1:length(CR_alt)
    if i == set[:outage_line]
        plot!(plo_cusum, w[i,:],
              lw=2,
              c=:green,
              label="line $(net_nom[:n_s][i])→$(net_nom[:n_r][i]) (true)")
    else
        plot!(plo_cusum, w[i,:],
              lw=2,
              c=palette[i],
              label="line $(net_nom[:n_s][i])→$(net_nom[:n_r][i])")
    end
end

vline!(plo_cusum,
       [change_time],
       line=:dash,
       c=red,
       lw=2,
       label=false)
hline!(plo_cusum,
       [50],
       line=:solid,
       c=red,
       lw=2,
       label="threshold")
############
final_plot = plot(
    plo_demand,
    plo_price,
    plo_cusum,
    size = (2000, 500),
    margin = 15mm,
    bottom_margin = 20mm,
    layout = @layout([a b c])

)

display(final_plot)
savefig(final_plot, "demand_price_cusum.pdf")

 ##########################
 ## Module 2
 ##########################

## Detection threshold
#α = 0.1                   # target false alarm probability
num_outages = length(CR_alt)
# η = log(num_outages / α)
η =50
# -----------------------------
# Parameters
# -----------------------------
M = 1000                # number of Monte Carlo runs
T = N                      # time horizon
# -----------------------------
# Storage
# -----------------------------
# =============================
# Initialization
# =============================
A = length(CR_alt)          # number of candidate outages
w_mean = zeros(A, T)

τ_detect = zeros(Int, M)    # global detection time τ
a_hat    = zeros(Int, M)    # identified outage

a_true = set[:outage_line]

# =============================
# Monte Carlo loop
# =============================
for m in 1:M
    println("MC run $m / $M")

    # -----------------------------
    # Generate demand noise
    # -----------------------------
    Random.seed!(127)
    Δξ = rand(MultivariateNormal(zeros(2), Σ_Δξ), T)

    # -----------------------------
    # Demand trajectory
    # -----------------------------
    ξ = zeros(2, T)
    for t in 2:T
        ξ[:, t] = ξ[:, t-1] .+ Δξ[:, t]
    end
    ξ .= clamp.(ξ, -set[:load_variation], set[:load_variation])

    # -----------------------------
    # Price increments
    # -----------------------------
    Δλ = zeros(net_nom[:N], T-1)

    for t in 2:T
        if t < change_time
            cr_ind_nom = 0
            for j in keys(CR_nom)
                if in_polyhedron(CR_nom[j][:A], CR_nom[j][:b], ξ[:, t])
                    cr_ind_nom = j
                    break
                end
            end
            cr_ind_nom == 0 && continue

            λt   = CR_nom[cr_ind_nom][:Λ̃] *
                   (CR_nom[cr_ind_nom][:D] * ξ[:, t]   + CR_nom[cr_ind_nom][:d])
            λt_1 = CR_nom[cr_ind_nom][:Λ̃] *
                   (CR_nom[cr_ind_nom][:D] * ξ[:, t-1] + CR_nom[cr_ind_nom][:d])

            Δλ[:, t-1] = λt .- λt_1

        else
            cr_ind_alt = 0
            for j in keys(CR_alt[a_true])
                if in_polyhedron(
                    CR_alt[a_true][j][:A],
                    CR_alt[a_true][j][:b],
                    ξ[:, t]
                )
                    cr_ind_alt = j
                    break
                end
            end
            cr_ind_alt == 0 && continue

            λt   = CR_alt[a_true][cr_ind_alt][:Λ̃] *
                   (CR_alt[a_true][cr_ind_alt][:D] * ξ[:, t]   +
                    CR_alt[a_true][cr_ind_alt][:d])
            λt_1 = CR_alt[a_true][cr_ind_alt][:Λ̃] *
                   (CR_alt[a_true][cr_ind_alt][:D] * ξ[:, t-1] +
                    CR_alt[a_true][cr_ind_alt][:d])

            Δλ[:, t-1] = λt .- λt_1
        end
    end

    # -----------------------------
    # CUSUM recursion (Algorithm 1)
    # -----------------------------
    w_tmp = zeros(A, T)

    for t in 2:T
        Δprice = Δλ[:, t-1]

        # Nominal CR
        cr_ind_nom = 0
        for j in keys(CR_nom)
            if in_polyhedron(CR_nom[j][:A], CR_nom[j][:b], ξ[:, t])
                cr_ind_nom = j
                break
            end
        end

        if cr_ind_nom == 0
            w_tmp[:, t] .= w_tmp[:, t-1]
            continue
        end

        # Update all candidate outages
        for a in 1:A
            cr_ind_alt = 0
            for j in keys(CR_alt[a])
                if in_polyhedron(CR_alt[a][j][:A],
                                 CR_alt[a][j][:b],
                                 ξ[:, t])
                    cr_ind_alt = j
                    break
                end
            end

            if cr_ind_alt == 0
                w_tmp[a, t] = w_tmp[a, t-1]
                continue
            end

            try
                r = l(Δprice, cr_ind_alt, cr_ind_nom, a, ξ[:, t])
                w_tmp[a, t] = max(0, w_tmp[a, t-1] + r)
            catch
                w_tmp[a, t] = w_tmp[a, t-1]
            end
        end

        # -----------------------------
        # stopping rule
        # -----------------------------
        if τ_detect[m] == 0
            w_max, a_max = findmax(w_tmp[:, t])
            if w_max ≥ η
                τ_detect[m] = t
                a_hat[m]    = a_max
            end
        end
    end

end

# =============================
# Performance Metrics
# =============================
valid = ((τ_detect .> change_time) .+ (a_hat .== a_true)) .== 2


# Detection delay
delay = τ_detect[valid.==1] .- change_time
E_delay   = mean(delay)
Std_delay = std(delay)

# Identification accuracy
P_correct = sum(a_hat[τ_detect .> change_time] .== a_true)/length(a_hat[τ_detect .> change_time]) * 100


##################
## Module 3: ARL
##################
# -----------------------------
# ARL parameters
# -----------------------------
T_arl = 5000
τ_arl = zeros(Int, M)   

for m in 1:M
    println("ARL MC run $m / $M")

    # -----------------------------
    # Generate nominal demand noise
    # -----------------------------
    Δξ = rand(MultivariateNormal(zeros(2), Σ_Δξ), T_arl)

    ξ = zeros(2, T_arl)
    for t in 2:T_arl
        ξ[:, t] = ξ[:, t-1] .+ Δξ[:, t]
    end
    ξ .= clamp.(ξ, -set[:load_variation], set[:load_variation])

    # -----------------------------
    # Nominal price increments
    # -----------------------------
    Δλ = zeros(net_nom[:N], T_arl-1)

    for t in 2:T_arl
        for j in keys(CR_nom)
            if in_polyhedron(CR_nom[j][:A], CR_nom[j][:b], ξ[:, t])
                λt   = CR_nom[j][:Λ̃] *
                       (CR_nom[j][:D] * ξ[:, t]   + CR_nom[j][:d])
                λt_1 = CR_nom[j][:Λ̃] *
                       (CR_nom[j][:D] * ξ[:, t-1] + CR_nom[j][:d])
                Δλ[:, t-1] = λt - λt_1
                break
            end
        end
    end

    # -----------------------------
    # CUSUM recursion (nominal only)
    # -----------------------------
    w_tmp = zeros(A, T_arl)

    for t in 2:T_arl
        Δprice = Δλ[:, t-1]

        # Nominal critical region
        cr_nom = 0
        for j in keys(CR_nom)
            if in_polyhedron(CR_nom[j][:A], CR_nom[j][:b], ξ[:, t])
                cr_nom = j
                break
            end
        end
        cr_nom == 0 && continue

        # Update CUSUMs
        for a in 1:A
            cr_alt = 0
            for j in keys(CR_alt[a])
                if in_polyhedron(CR_alt[a][j][:A],
                                 CR_alt[a][j][:b],
                                 ξ[:, t])
                    cr_alt = j
                    break
                end
            end
            cr_alt == 0 && continue

            r = l(Δprice, cr_alt, cr_nom, a, ξ[:, t])
            w_tmp[a, t] = max(0, w_tmp[a, t-1] + r)
        end

        # -----------------------------
        # First false alarm 
        # -----------------------------
        if maximum(w_tmp[:, t]) ≥ η
            τ_arl[m] = t
            break
        end
    end
end

# -----------------------------
# Average Run Length
# -----------------------------
valid = τ_arl .> 0
ARL = mean(τ_arl[valid])
############

