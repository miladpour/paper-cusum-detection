# auxiliary functions
ns(l) = Int(net[:n_s][l])
nr(l) = Int(net[:n_r][l])
Φ(x) = quantile(Normal(0,1),1-x)
function remove_col_and_row(B,refbus)
    @assert size(B,1) == size(B,2)
    n = size(B,1)
    return B[1:n .!= refbus, 1:n .!= refbus]
end

function build_B̆(B̂inv,refbus)
    Nb = size(B̂inv,1)+1
    B̆ = zeros(Nb,Nb)
    for i in 1:Nb, j in 1:Nb
        if i < refbus && j < refbus
            B̆[i,j] = B̂inv[i,j]
        end
        if i > refbus && j > refbus
            B̆[i,j] = B̂inv[i-1,j-1]
        end
        if i > refbus && j < refbus
            B̆[i,j] = B̂inv[i-1,j]
        end
        if i < refbus && j > refbus
            B̆[i,j] = B̂inv[i,j-1]
        end
    end
    return B̆
end

function blockdiag(mats::AbstractMatrix...)
    # total size
    rows = sum(size(M,1) for M in mats)
    cols = sum(size(M,2) for M in mats)
    # create zero matrix of total size
    M = zeros(eltype(mats[1]), rows, cols)
    
    # fill in the blocks
    r = 1
    c = 1
    for A in mats
        M[r:r+size(A,1)-1, c:c+size(A,2)-1] = A
        r += size(A,1)
        c += size(A,2)
    end
    return M
end

function is_invertible(A)
    try
        factorize(A)   # LU for dense, other methods for sparse
        return true
    catch
        return false
    end
end

function load_network_data(caseID;line_outaged = false, reduced_trans_cap = false, remove_load = false, load_scale = 1)
    data_net = PowerModels.parse_file(caseID)
    # Network size
    G = length(data_net["gen"])
    N = length(data_net["bus"])
    # remove load if necessery 
    if remove_load != false
        delete!(data_net["load"], remove_load)
        loads = collect(values(data_net["load"]))   # get all branch data
        new_loads = Dict(string(i) => loads[i] for i in 1:length(loads))
        for i in 1:length(loads)
            loads[i]["index"] = i
        end
        # replace
        data_net["load"] = new_loads
    end
    D = length(data_net["load"])
    # remove outaged line if any 
    if line_outaged != false
        delete!(data_net["branch"], line_outaged)
        # rebuild branch dict with continuous keys
        branches = collect(values(data_net["branch"]))   # get all branch data
        new_branch = Dict(string(i) => branches[i] for i in 1:length(branches))
        for i in 1:length(branches)
            branches[i]["index"] = i
        end
        # replace
        data_net["branch"] = new_branch
    end
    E = length(data_net["branch"])

    # order bus indexing
    bus_keys=collect(keys(data_net["bus"]))
    bus_keys = bus_keys[sortperm(parse.(Int64, bus_keys))]
    bus_key_dict = Dict()
    for i in 1:N
        push!(bus_key_dict, i => bus_keys[i])
    end
    node(key) = [k for (k,v) in bus_key_dict if v == key][1]

    # Load generation data
    gen_key=collect(keys(data_net["gen"]))
    gen_key = gen_key[sortperm(parse.(Int64, gen_key))]
    p̅ = zeros(G); p̲ = zeros(G); c0 = zeros(G); c1 = zeros(G); c2 = zeros(G); M_p = zeros(N,G);
    for g in gen_key
        p̅[parse(Int64,g)] = data_net["gen"][g]["pmax"]*data_net["baseMVA"]
        p̲[parse(Int64,g)] = data_net["gen"][g]["pmin"]*data_net["baseMVA"]
        if sum(data_net["gen"][g]["cost"]) != 0
            if length(data_net["gen"][g]["cost"]) == 2
                c1[parse(Int64,g)] = data_net["gen"][g]["cost"][1] / data_net["baseMVA"]
                c0[parse(Int64,g)] = data_net["gen"][g]["cost"][2]
            end
            if length(data_net["gen"][g]["cost"]) == 3
                c2[parse(Int64,g)] = data_net["gen"][g]["cost"][1] / data_net["baseMVA"]^2
                c1[parse(Int64,g)] = data_net["gen"][g]["cost"][2] / data_net["baseMVA"]
                c0[parse(Int64,g)] = data_net["gen"][g]["cost"][3] 
            end
        end
        M_p[node(string(data_net["gen"][g]["gen_bus"])),parse(Int64,g)] = 1
    end
    replace!(c2, 0 => 0.1)

    # Load demand data
    load_key=collect(keys(data_net["load"]))
    load_key = load_key[sortperm(parse.(Int64, load_key))]

    d = zeros(D); M_d = zeros(N,D); s1 = zeros(D); s2 = zeros(D); M_ξ = zeros(N,2)
    for h in load_key
        d[parse(Int64,h)] = data_net["load"][h]["pd"]*data_net["baseMVA"]*load_scale
        M_d[node(string(data_net["load"][h]["load_bus"])),parse(Int64,h)] = 1
        s1[parse(Int64,h)] = 100
        s2[parse(Int64,h)] = 0.1
        count = 0
        for h in ["2","3"]
            count += 1
            M_ξ[node(string(data_net["load"][h]["load_bus"])),count] = 1
        end
    end

    # Load transmission data
    line_key=collect(keys(data_net["branch"]))
    line_key = line_key[sortperm(parse.(Int64, line_key))]

    β = zeros(E); f̅ = zeros(E); n_s = trunc.(Int64,zeros(E)); n_r = trunc.(Int64,zeros(E))
    for l in line_key
        β[data_net["branch"][l]["index"]] = -imag(1/(data_net["branch"][l]["br_r"] + data_net["branch"][l]["br_x"]im))
        n_s[data_net["branch"][l]["index"]] = node(string(data_net["branch"][l]["f_bus"]))
        n_r[data_net["branch"][l]["index"]] = node(string(data_net["branch"][l]["t_bus"]))
        f̅[data_net["branch"][l]["index"]] = data_net["branch"][l]["rate_a"]*data_net["baseMVA"]
    end

    ### Reduce transmission capacity of two braches to provoke congestion
    ### for 5-Bus PJM system: 1->2, 4->5.
    # find inex of the line based on sending and recieving ends 
    if reduced_trans_cap != false
        n_s_r = [n_s n_r]
        try 
            f̅[(findall(i -> (n_s_r[i,1] == 1) && (n_s_r[i,2] == 2), 1:size(n_s_r,1)))[1]] = 200
        catch
        end
        try
            f̅[(findall(i -> (n_s_r[i,1] == 4) && (n_s_r[i,2] == 5), 1:size(n_s_r,1)))[1]] = 150
        catch 
        end
    end

    # Find reference node
    ref = 1
    for n in 1:N
        if sum(M_p[n,:]) == 0 &&  sum(M_d[n,:]) == 0 == 0
            ref = n
        end
    end

    # Compute PTDF matrix
    B_line = zeros(E,N); B̃_bus = zeros(N,N); B = zeros(N,N)
    for n in 1:N
        for l in 1:E
            if n_s[l] == n
                B[n,n] += β[l]
                B_line[l,n] = β[l]
            end
            if n_r[l] == n
                B[n,n] += β[l]
                B_line[l,n] = -β[l]
            end
        end
    end
    for l in 1:E
        B[Int(n_s[l]),Int(n_r[l])] = - β[l]
        B[Int(n_r[l]),Int(n_s[l])] = - β[l]
    end
    B̃_bus = remove_col_and_row(B,ref)
    B̃_bus = inv(B̃_bus)
    B̃_bus = build_B̆(B̃_bus,ref)
    PTDF = B_line*B̃_bus

    # safe network data
    net = Dict(
    # transmission data
    :f̅ => f̅, :n_s => n_s, :n_r => n_r, :T => PTDF,
    # load data
    :l => d, :M_l => M_d, :s1 => s1, :s2 => s2, :M_ξ => M_ξ,
    # generation data
    :p̅ => p̅, :p̲ => p̲, :M_p => M_p,
    :c1 => c1, :c2 => c2, :c0=> c0,
    # graph data
    :N => N, :E => E, :G => G, :D => D, :ref => ref,
    )
    return net
end


function movavg(v, window)
    n = length(v)
    c = cumsum(v)
    out = similar(v)
    for i in 1:n
        l = max(1, i - window + 1)
        out[i] = (c[i] - (l > 1 ? c[l-1] : zero(eltype(v)))) / (i - l + 1)
    end
    return out
end
