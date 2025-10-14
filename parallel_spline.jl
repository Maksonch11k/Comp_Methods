using Base.Threads

include("spline_approximation.jl")

function three_point_functional_approx_parallel(f::Function, a::Float64, b::Float64, n::Int)
    k = 3
    knots = create_uniform_knots(a, b, n, k)
    h = (b-a)/n
    
    grid_points = [a + i*h for i in -1:(n+1)]
    
    num_coeffs = n + k - 1
    coeffs = zeros(num_coeffs)
    
    w = [1/8, 6/8, 1/8]
    
    @threads for i in 1:num_coeffs
        center_idx = i
        f_vals = [f(grid_points[center_idx]), f(grid_points[center_idx+1]), f(grid_points[center_idx+2])]
        coeffs[i] = w[1]*f_vals[1] + w[2]*f_vals[2] + w[3]*f_vals[3]
    end

    return t -> spline_eval(t, coeffs, knots, k)
end