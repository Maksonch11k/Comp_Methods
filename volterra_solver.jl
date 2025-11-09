using LinearAlgebra
using QuadGK

include("utils.jl")

export solve_volterra_II

function solve_volterra_II(K, f, a, b, n)
    k = 3
    basis, knots, num_coeffs = create_spline_basis_and_knots(a, b, n, k)
    
    μ = calculate_three_point_coeffs(f, a, b, n, k)

    M = zeros(num_coeffs, num_coeffs)
    for j in 1:num_coeffs
        K_omega_i_funcs = [t -> quadgk(x -> K(t, x) * basis[i](x), a, t)[1] for i in 1:num_coeffs]
        coeffs_for_row_j = [calculate_three_point_coeffs(fn, a, b, n, k)[j] for fn in K_omega_i_funcs]
        M[j, :] = coeffs_for_row_j
    end
    
    c = (I - M) \ μ
    u_h(t) = spline_eval(t, c, knots, k)
    
    return u_h
end