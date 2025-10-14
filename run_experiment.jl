using Plots
using PrettyTables
using Printf

include("spline_approximation.jl")

function calculate_error(f::Function, s::Function, a::Float64, b::Float64, n_fine::Int)
    fine_grid = range(a, stop=b, length=n_fine)
    max_err = 0.0
    for x in fine_grid
        err = abs(f(x) - s(x))
        if err > max_err
            max_err = err
        end
    end
    return max_err
end

a, b = 0.0, 1.0
test_functions = [
    (x -> sin(2*pi*x), "sin(2πx)"),
    (x -> sign(x - 0.5), "sign(x - 0.5)"),
    (x -> abs(x - 0.5), "|x - 0.5|")
]
n_values = [10, 20, 40, 80]

results = []
plot_array = []

for (f, fname) in test_functions
    n_vis = 10
    s1_vis = variation_diminishing_approx(f, a, b, n_vis)
    s2_vis = three_point_functional_approx(f, a, b, n_vis)
    
    p = plot(f, a, b, label="Исходная f(x)", lw=2, title=fname, legend=:bottom)
    plot!(s1_vis, a, b, label="Уменьш. вариации", lw=2, linestyle=:dash)
    plot!(s2_vis, a, b, label="Трехточечный функц.", lw=2, linestyle=:dot)
    push!(plot_array, p)
    
    for n in n_values
        s1 = variation_diminishing_approx(f, a, b, n)
        err1 = calculate_error(f, s1, a, b, 10 * n)
        
        s2 = three_point_functional_approx(f, a, b, n)
        err2 = calculate_error(f, s2, a, b, 10 * n)
        
        push!(results, (fname, n, err1, err2))
    end
end

plot(plot_array..., layout=(3,1), size=(800, 1000))
savefig("spline_approximations_comparison.png")

header = ["Функция", "N", "Ошибка (Уменьш. вариации)", "Ошибка (Трехточ. функц.)"]
data = Matrix{Any}(undef, length(results), 4)
for (i, res) in enumerate(results)
    data[i, 1] = res[1]
    data[i, 2] = res[2]
    data[i, 3] = @sprintf("%.2e", res[3])
    data[i, 4] = @sprintf("%.2e", res[4])
end

println("Таблица максимальных абсолютных ошибок (max|f(x) - s(x)|):")
pretty_table(data, header=header, alignment=:c)