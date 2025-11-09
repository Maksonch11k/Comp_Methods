export create_spline_basis_and_knots, calculate_three_point_coeffs, spline_eval

function cox_de_boor(i::Int, k::Int, t::Float64, knots::Vector{Float64})
    if k == 1
        return (knots[i] <= t < knots[i+1]) ? 1.0 : 0.0
    end

    term1 = 0.0
    if knots[i+k-1] - knots[i] > 1e-9
        term1 = (t - knots[i]) / (knots[i+k-1] - knots[i]) * cox_de_boor(i, k - 1, t, knots)
    end

    term2 = 0.0
    if knots[i+k] - knots[i+1] > 1e-9
        term2 = (knots[i+k] - t) / (knots[i+k] - knots[i+1]) * cox_de_boor(i + 1, k - 1, t, knots)
    end

    return term1 + term2
end

function create_uniform_knots(a::Float64, b::Float64, num_coeffs::Int, k::Int)
    total_knots = num_coeffs + k
    knots = zeros(total_knots)
    
    for i in 1:k
        knots[i] = a
    end
    
    n_internal_knots = total_knots - 2*k
    h = (b - a) / (n_internal_knots + 1)
    for i in 1:n_internal_knots
        knots[k+i] = a + i*h
    end

    for i in (total_knots - k + 1):total_knots
        knots[i] = b
    end
    
    return knots
end

function spline_eval(t::Float64, coeffs::Vector{Float64}, knots::Vector{Float64}, k::Int)
    i = findfirst(x -> x > t, knots)
    if isnothing(i)
        i = length(knots)
    end
    i = max(i - 1, k)
    
    val = 0.0
    for j in (i - k + 1):i
        if j > 0 && j <= length(coeffs)
            val += coeffs[j] * cox_de_boor(j, k, t, knots)
        end
    end
    return val
end

function create_spline_basis_and_knots(a::Float64, b::Float64, n_intervals::Int, k::Int)
    num_coeffs = n_intervals + k - 1
    knots = create_uniform_knots(a, b, num_coeffs, k)
    basis = [t -> cox_de_boor(i, k, t, knots) for i in 1:num_coeffs]
    return basis, knots, num_coeffs
end

function calculate_three_point_coeffs(f::Function, a::Float64, b::Float64, n_intervals::Int, k::Int)
    num_coeffs = n_intervals + k - 1
    knots = create_uniform_knots(a, b, num_coeffs, k)
    h = (b - a) / n_intervals
    
    coeffs = zeros(num_coeffs)
    w = [-1/2, 2, -1/2]
    
    coeffs[1] = f(a)
    coeffs[end] = f(b)

    for i in 2:(num_coeffs-1)
        node = (knots[i+1] + knots[i+2]) / 2.0
        coeffs[i] = w[1]*f(node-h) + w[2]*f(node) + w[3]*f(node+h)
    end

    return coeffs
end