# FOMV: Field Operator for Measured Viability
# Código de simulación para el modelo HARD-nonlinear (Julia version)
# Autor: Osvaldo Morales
# Licencia: AGPL-3.0

using Random, Statistics, LinearAlgebra, Plots, ProgressMeter

# =============================================================================
# Parámetros del modelo (Tabla 1 del paper)
# =============================================================================
Base.@kwdef struct ModelParams
    α1::Float64 = 0.1
    α2::Float64 = 0.2
    δ::Float64  = 0.05
    β1::Float64 = 0.3
    γ1::Float64 = 0.2
    γ2::Float64 = 0.1
    γ3::Float64 = 0.1
    φ1::Float64 = 0.3
    φ2::Float64 = 0.2
    ψ1::Float64 = 0.2
    ψ2::Float64 = 0.2
    κ1::Float64 = 0.2
    κ2::Float64 = 0.1
    Ec::Float64 = 0.1
    Er::Float64 = 0.5
    Lc::Float64 = 1.5
    Lr::Float64 = 0.8
end

Base.@kwdef struct SimParams
    σ::Float64 = 0.05
    Tmax::Int = 200
    R::Int = 10000
    nB::Int = 20
    nM::Int = 20
    B_range::Tuple{Float64,Float64} = (0.0, 1.2)
    M_range::Tuple{Float64,Float64} = (0.0, 0.8)
    bootstrap_reps::Int = 1000
    α::Float64 = 0.05
    fast_samples::Int = 1000
end

# =============================================================================
# Funciones auxiliares
# =============================================================================
sigmoid(x::Float64) = 1.0 / (1.0 + exp(-x))

function generate_noise(σ::Float64)
    while true
        u = 2.0 .* rand(6) .- 1.0
        prob = prod(0.75 * (1.0 .- u.^2))
        if rand() < prob
            return σ * u
        end
    end
end

# =============================================================================
# Dinámica HARD-nonlinear
# =============================================================================
function dynamics(x::Vector{Float64}, θ::ModelParams, η::Vector{Float64})
    B, M, E, G, T, C = x
    B_star = B + θ.α1*(1 - E) - θ.α2*G
    M_star = (1 - θ.δ)*M + θ.β1 * sigmoid(B - T)
    E_star = E + θ.γ1*G - θ.γ2*B - θ.γ3*M
    G_star = G + θ.φ1*E - θ.φ2*(B + M)*(1 - T)
    T_star = T - θ.ψ1*M*(1 - G) + θ.ψ2*G
    C_star = C + θ.κ1*T - θ.κ2*B
    x_new = [B_star, M_star, E_star, G_star, T_star, C_star] .+ η
    return clamp.(x_new, 0.0, 1.0)
end

# =============================================================================
# Funciones de absorción
# =============================================================================
is_collapsed(x::Vector{Float64}, θ::ModelParams) = (x[3] ≤ θ.Ec) || (x[1] + x[2] ≥ θ.Lc)
is_recovered(x::Vector{Float64}, θ::ModelParams) = (x[3] ≥ θ.Er) && (x[1] + x[2] ≤ θ.Lr)

# =============================================================================
# Simulación de una trayectoria
# =============================================================================
function simulate_trajectory(x0::Vector{Float64}, θ::ModelParams, σ::Float64, Tmax::Int)
    x = copy(x0)
    for t in 1:Tmax
        if is_collapsed(x, θ)
            return x, t, 'C'
        end
        if is_recovered(x, θ)
            return x, t, 'R'
        end
        η = generate_noise(σ)
        x = dynamics(x, θ, η)
    end
    return x, Tmax, 'N'
end

# =============================================================================
# Estimación de committor y MFPT para un punto dado
# =============================================================================
function estimate_q_mfpt(x0::Vector{Float64}, θ::ModelParams, σ::Float64, Tmax::Int, R::Int)
    hits_R = 0
    times_C = Float64[]
    for _ in 1:R
        _, t, abs_type = simulate_trajectory(x0, θ, σ, Tmax)
        if abs_type == 'R'
            hits_R += 1
        elseif abs_type == 'C'
            push!(times_C, t)
        end
    end
    q_hat = hits_R / R
    mfpt_hat = isempty(times_C) ? NaN : mean(times_C)
    return q_hat, mfpt_hat
end

# =============================================================================
# Distribución estacionaria de variables rápidas condicionada a (B,M)
# =============================================================================
function stationary_fast_given_slow(B::Float64, M::Float64, θ::ModelParams, σ::Float64;
                                    n_samples::Int=1000, burnin::Int=500)
    samples = Matrix{Float64}(undef, n_samples, 4)
    fast = rand(4)
    for i in 1:(n_samples + burnin)
        x = [B, M, fast[1], fast[2], fast[3], fast[4]]
        η = generate_noise(σ)
        x_new = dynamics(x, θ, η)
        fast = x_new[3:6]
        if i > burnin
            samples[i - burnin, :] = fast
        end
    end
    return samples
end

# =============================================================================
# Estimación en grid (B,M) con proyección
# =============================================================================
function estimate_on_grid(B_grid::Vector{Float64}, M_grid::Vector{Float64},
                          θ::ModelParams, σ::Float64, Tmax::Int, R::Int;
                          fast_samples::Int=1000)
    nB = length(B_grid)
    nM = length(M_grid)
    Q = fill(NaN, nB, nM)
    MFPT = fill(NaN, nB, nM)

    prog = Progress(nB * nM, desc="Grid estimation")
    for i in 1:nB, j in 1:nM
        B = B_grid[i]
        M = M_grid[j]
        fast_samples_arr = stationary_fast_given_slow(B, M, θ, σ, n_samples=fast_samples)
        q_vals = Float64[]
        mfpt_vals = Float64[]
        for k in 1:fast_samples
            x0 = [B, M, fast_samples_arr[k,1], fast_samples_arr[k,2],
                  fast_samples_arr[k,3], fast_samples_arr[k,4]]
            q, mfpt = estimate_q_mfpt(x0, θ, σ, Tmax, R)
            push!(q_vals, q)
            push!(mfpt_vals, mfpt)
        end
        Q[i,j] = mean(skipmissing(q_vals))
        MFPT[i,j] = mean(skipmissing(mfpt_vals))
        next!(prog)
    end
    return Q, MFPT
end

# =============================================================================
# Bootstrap para bandas de confianza
# =============================================================================
function bootstrap_bands(B_grid::Vector{Float64}, M_grid::Vector{Float64},
                         θ::ModelParams, σ::Float64, Tmax::Int, R::Int;
                         fast_samples::Int=1000, bootstrap_reps::Int=1000, α::Float64=0.05)
    _, MFPT_hat = estimate_on_grid(B_grid, M_grid, θ, σ, Tmax, R, fast_samples=fast_samples)
    MFPT_boot = []
    prog = Progress(bootstrap_reps, desc="Bootstrap")
    for _ in 1:bootstrap_reps
        _, MFPT_b = estimate_on_grid(B_grid, M_grid, θ, σ, Tmax, R, fast_samples=fast_samples)
        push!(MFPT_boot, MFPT_b)
        next!(prog)
    end
    MFPT_boot = cat(MFPT_boot..., dims=3)
    lower = mapslices(x -> quantile(x, α/2), MFPT_boot, dims=3)[:,:,1]
    upper = mapslices(x -> quantile(x, 1 - α/2), MFPT_boot, dims=3)[:,:,1]
    return MFPT_hat, lower, upper
end

# =============================================================================
# Validación de proyección
# =============================================================================
function validate_projection(B_test::Vector{Float64}, M_test::Vector{Float64},
                             θ::ModelParams, σ::Float64, Tmax::Int, R::Int;
                             fast_samples::Int=1000)
    errors = Float64[]
    for (B,M) in zip(B_test, M_test)
        fast_samples_arr = stationary_fast_given_slow(B, M, θ, σ, n_samples=fast_samples)
        mfpt_2d_vals = Float64[]
        for k in 1:fast_samples
            x0 = [B, M, fast_samples_arr[k,1], fast_samples_arr[k,2],
                  fast_samples_arr[k,3], fast_samples_arr[k,4]]
            _, mfpt = estimate_q_mfpt(x0, θ, σ, Tmax, R)
            push!(mfpt_2d_vals, mfpt)
        end
        mfpt_2d = mean(skipmissing(mfpt_2d_vals))

        mfpt_6d_vals = Float64[]
        for _ in 1:fast_samples
            fast = rand(4)
            x0 = [B, M, fast[1], fast[2], fast[3], fast[4]]
            _, mfpt = estimate_q_mfpt(x0, θ, σ, Tmax, R)
            push!(mfpt_6d_vals, mfpt)
        end
        mfpt_6d = mean(skipmissing(mfpt_6d_vals))

        if !isnan(mfpt_6d) && mfpt_6d > 10
            err = abs(mfpt_2d - mfpt_6d) / mfpt_6d
            push!(errors, err)
            @printf("B=%.2f, M=%.2f: 2D=%.1f, 6D=%.1f, error=%.2f%%\n",
                    B, M, mfpt_2d, mfpt_6d, err*100)
        else
            @printf("B=%.2f, M=%.2f: MFPT pequeño o NaN, omitido\n", B, M)
        end
    end
    if !isempty(errors)
        @printf("\nError medio: %.2f%%, máximo: %.2f%%\n",
                mean(errors)*100, maximum(errors)*100)
    end
    return errors
end

# =============================================================================
# Visualización
# =============================================================================
function plot_mfpt(B_grid::Vector{Float64}, M_grid::Vector{Float64},
                   MFPT::Matrix{Float64}; bands_lower=nothing, title="MFPT on slow manifold")
    heatmap(B_grid, M_grid, MFPT',
            xlabel="Backlog B", ylabel="Memory M",
            title=title, color=:viridis)
    if bands_lower !== nothing
        Z = MFPT .- bands_lower
        contour!(B_grid, M_grid, Z', levels=[0], color=:red, linestyle=:dash, linewidth=2)
    end
    current()
end

# =============================================================================
# Main
# =============================================================================
function main()
    println("="^60)
    println("FOMV: Simulación del modelo HARD-nonlinear (Julia)")
    println("="^60)

    θ = ModelParams()
    sim = SimParams()

    B_grid = range(sim.B_range[1], sim.B_range[2], length=sim.nB)
    M_grid = range(sim.M_range[1], sim.M_range[2], length=sim.nM)

    println("\nEstimando MFPT en grid (esto puede tomar tiempo)...")
    Q, MFPT = estimate_on_grid(B_grid, M_grid, θ, sim.σ, sim.Tmax, sim.R,
                                fast_samples=sim.fast_samples)

    println("\nCalculando bandas bootstrap (esto puede tomar mucho tiempo)...")
    MFPT_hat, MFPT_lower, MFPT_upper = bootstrap_bands(
        B_grid, M_grid, θ, sim.σ, sim.Tmax, sim.R,
        fast_samples=sim.fast_samples, bootstrap_reps=sim.bootstrap_reps, α=sim.α)

    plot_mfpt(B_grid, M_grid, MFPT_hat, bands_lower=MFPT_lower,
              title="MFPT con banda de protección (95%)")

    println("\nValidando proyección con algunos puntos 6D...")
    Random.seed!(42)
    B_test = rand(5) * 0.8 .+ 0.2
    M_test = rand(5) * 0.5 .+ 0.1
    validate_projection(B_test, M_test, θ, sim.σ, sim.Tmax, sim.R,
                        fast_samples=sim.fast_samples)

    println("\nSimulación completada.")
    return MFPT_hat, MFPT_lower
end

# Ejecutar main (descomentar para correr)
# MFPT_hat, MFPT_lower = main()
