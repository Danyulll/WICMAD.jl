module PostProcessing

using StatsBase

export dahl_partition, dahl_from_res, map_partition, map_from_res

function dahl_partition(Z::AbstractMatrix{<:Integer})
    S, N = size(Z)
    PSM = zeros(Float64, N, N)
    for s in 1:S
        zs = Z[s, :]
        A = zs .== permutedims(zs)
        PSM .+= Float64.(A)
    end
    PSM ./= S
    score = zeros(Float64, S)
    for s in 1:S
        zs = Z[s, :]
        A = zs .== permutedims(zs)
        score[s] = sum((Float64.(A) .- PSM) .^ 2)
    end
    s_hat = argmin(score)
    z_hat = canonical_labels(vec(Z[s_hat, :]))
    K_hat = length(unique(z_hat))
    (; z_hat, K_hat, s_hat, PSM, score)
end

function canonical_labels(z::AbstractVector{<:Integer})
    uniq = unique(z)
    mapping = Dict(u => i for (i, u) in enumerate(uniq))
    [mapping[val] for val in z]
end

function dahl_from_res(res::NamedTuple)
    haskey(res, :Z) || error("res[:Z] not found")
    dahl_partition(res.Z)
end

function map_partition(Z::AbstractMatrix{<:Integer})
    S, N = size(Z)
    keys = Vector{String}(undef, S)
    for s in 1:S
        keys[s] = join(canonical_labels(vec(Z[s, :])), "-")
    end
    tab = counts(keys)
    key_hat, freq = findmax(tab)
    s_hat = findfirst(==(key_hat), keys)
    z_hat = canonical_labels(vec(Z[s_hat, :]))
    K_hat = length(unique(z_hat))
    (; z_hat, K_hat, s_hat, key_hat, freq)
end

function map_from_res(res::NamedTuple)
    haskey(res, :Z) || error("res[:Z] not found")
    map_partition(res.Z)
end

end # module
