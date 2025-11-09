using LinearAlgebra
using QuadGK

include("utils.jl")

export solve_fredholm_II

function solve_fredholm_II(K, f, a, b, n)
    k = 3 
    basis, knots, num_coeffs = create_spline_basis_and_knots(a, b, n, k)
    
    μ = calculate_three_point_coeffs(f, a, b, n, k)

    M = zeros(num_coeffs, num_coeffs)
    for j in 1:num_coeffs
        K_omega_i_funcs = [t -> quadgk(x -> K(t, x) * basis[i](x), a, b)[1] for i in 1:num_coeffs]
        coeffs_for_row_j = [calculate_three_point_coeffs(fn, a, b, n, k)[j] for fn in K_omega_i_funcs]
        M[j, :] = coeffs_for_row_j
    end

    c = (I - M) \ μ
    u_h(t) = spline_eval(t, c, knots, k)
    u_tilde_h(t) = f(t) + quadgk(x -> K(t, x) * u_h(x), a, b)[1]

    return u_tilde_h
end


export solve_fredholm_I_direct, solve_fredholm_I_regularized


function solve_fredholm_I_direct(K, g, a, b, n)
    k = 3 
  
    basis, knots, num_coeffs = create_spline_basis_and_knots(a, b, n, k)
    
    
    colloc_pts = [(knots[i+1]+knots[i+2])/2.0 for i in 1:num_coeffs] 
    g_vec = g.(colloc_pts)
    
    M = zeros(num_coeffs, num_coeffs)
    for i in 1:num_coeffs
        for j in 1:num_coeffs
            integrand(t) = K(colloc_pts[i], t) * basis[j](t)
            M[i, j] = quadgk(integrand, a, b)[1]
        end
    end

    
    c = pinv(M) * g_vec
    
  
    u_h(t) = spline_eval(t, c, knots, k)
    return u_h
end


function solve_fredholm_I_regularized(K, g, a, b, n, alpha)
    k = 3 

    basis, knots, num_coeffs = create_spline_basis_and_knots(a, b, n, k)

    colloc_pts = [(knots[i+1]+knots[i+2])/2.0 for i in 1:num_coeffs] 
    g_vec = g.(colloc_pts)
    
    M = zeros(num_coeffs, num_coeffs)
    for i in 1:num_coeffs
        for j in 1:num_coeffs
            integrand(t) = K(colloc_pts[i], t) * basis[j](t)
            M[i, j] = quadgk(integrand, a, b)[1]
        end
    end
    
    
    A = M' * M + alpha * I
    rhs = M' * g_vec
    c = A \ rhs
    
    
    u_h(t) = spline_eval(t, c, knots, k)
    return u_h
end