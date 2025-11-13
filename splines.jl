using Base.Threads
using Printf


function B(i, p, x, t_array)
    j = i+1
    if p == 0
        return t_array[j] <= x < t_array[j+1] ? 1.0 : 0.0
    else
        result = 0.0
        denom1 = t_array[j+p] - t_array[j]
        if denom1 != 0
            result += ((x - t_array[j])/denom1) * B(i, p-1, x, t_array)
        end
        
        denom2 = t_array[j+p+1] - t_array[j+1]
        if denom2 != 0
            result += ((t_array[j+p+1] - x)/denom2) * B(i+1, p-1, x, t_array)
        end

        return result
    end
end


function do_array_for_approx(a, b, h)
    n = round(Int, (b-a)/h)
    return collect(range(a - 3h, b + 3h, length=n+7))
end


function three_point_functional_coeffs(f, knots)
    p = 3
    num_coeffs = length(knots) - p - 1
    coeffs = zeros(num_coeffs)

    for i in 1:num_coeffs
        p1 = knots[i+1]
        p3 = knots[i+3]
        p2 = (p1 + p3) / 2
        
        if i == 1
            coeffs[i] = f(p1)
        elseif i == num_coeffs
            coeffs[i] = f(p3)
        else
            try
                coeffs[i] = -0.5 * f(p1) + 2.0 * f(p2) - 0.5 * f(p3)
            catch e
                coeffs[i] = f(knots[i+2])
            end
        end
    end
    return coeffs
end