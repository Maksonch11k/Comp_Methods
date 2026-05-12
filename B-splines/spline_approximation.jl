function cox_de_boor(i::Int, k::Int, t::Float64, knots::Vector{Float64})
    if k == 1
        return (knots[i] <= t < knots[i+1]) ? 1.0 : 0.0
    end

    term1 = 0.0
    if knots[i+k-1] - knots[i] > 0
        term1 = (t - knots[i]) / (knots[i+k-1] - knots[i]) * cox_de_boor(i, k - 1, t, knots)
    end

    term2 = 0.0
    if knots[i+k] - knots[i+1] > 0
        term2 = (knots[i+k] - t) / (knots[i+k] - knots[i+1]) * cox_de_boor(i + 1, k - 1, t, knots)
    end

    return term1 + term2
end

function create_uniform_knots(a::Float64, b::Float64, n::Int, k::Int)
    h = (b - a) / n
    return [(a + (i - k) * h) for i in 1:(n + 2k - 1)]
end

function spline_eval(t::Float64, coeffs::Vector{Float64}, knots::Vector{Float64}, k::Int)
    i = findlast(x -> x <= t, knots)
    if isnothing(i) || i < k || i > length(knots) - k
        i = k
    end

    val = 0.0
    for j in (i - k + 1):i
        if j > 0 && j <= length(coeffs)
            val += coeffs[j] * cox_de_boor(j, k, t, knots)
        end
    end
    return val
end

function variation_diminishing_approx(f::Function, a::Float64, b::Float64, n::Int)
    k = 3
    knots = create_uniform_knots(a, b, n, k)
    
    greville_abscissae = [(knots[i+1] + knots[i+2]) / 2.0 for i in 1:(length(knots) - k)]
    
    coeffs = f.(greville_abscissae)
    
    return t -> spline_eval(t, coeffs, knots, k)
end

function three_point_functional_approx(f::Function, a::Float64, b::Float64, n::Int)
    k = 3
    knots = create_uniform_knots(a, b, n, k)
    h = (b-a)/n
    
    grid_points = [a + i*h for i in -1:(n+1)]
    
    num_coeffs = n + k - 1
    coeffs = zeros(num_coeffs)
    
    w = [1/8, 6/8, 1/8]
    
    for i in 1:num_coeffs
        center_idx = i
        f_vals = [f(grid_points[center_idx]), f(grid_points[center_idx+1]), f(grid_points[center_idx+2])]
        coeffs[i] = w[1]*f_vals[1] + w[2]*f_vals[2] + w[3]*f_vals[3]
    end

    return t -> spline_eval(t, coeffs, knots, k)
end