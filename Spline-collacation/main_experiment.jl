using Plots
using QuadGK
using Printf
include("fredholm_solvers.jl")
include("volterra_solver.jl")

println("--- Часть 1: Уравнение Фредгольма II рода ---")
println("\n--- Пример 1.1: Уравнение с тригонометрическим ядром ---")
K_1_1(t, x) = sin(t) * cos(x)
f_1_1(t) = sin(t)
a_1_1, b_1_1 = 0.0, π/2
u_exact_1_1(t) = 2 * sin(t)
n_1_1 = 30
u_approx_1_1 = solve_fredholm_II(K_1_1, f_1_1, a_1_1, b_1_1, n_1_1)
t_plot_1_1 = range(a_1_1, stop=b_1_1, length=200)
p1_1 = plot(t_plot_1_1, u_exact_1_1, label="Точное решение", lw=2)
plot!(p1_1, t_plot_1_1, u_approx_1_1, label="Приближенное (n=$n_1_1)", ls=:dash, lw=2)
title!("Часть 1, Пример 1")
error_1_1 = maximum(abs.(u_exact_1_1.(t_plot_1_1) .- u_approx_1_1.(t_plot_1_1)))
println("Максимальная абсолютная ошибка: $error_1_1")
savefig(p1_1, "fredholm_II_ex1.png")

println("\n--- Пример 1.2: Уравнение с полиномиальным ядром ---")
K_1_2(t, x) = t*x
f_1_2(t) = t * 5/6
a_1_2, b_1_2 = 0.0, 1.0
u_exact_1_2(t) = t
n_1_2 = 30
u_approx_1_2 = solve_fredholm_II(K_1_2, f_1_2, a_1_2, b_1_2, n_1_2)
t_plot_1_2 = range(a_1_2, stop=b_1_2, length=200)
p1_2 = plot(t_plot_1_2, u_exact_1_2, label="Точное решение", lw=2)
plot!(p1_2, t_plot_1_2, u_approx_1_2, label="Приближенное (n=$n_1_2)", ls=:dash, lw=2)
title!("Часть 1, Пример 2")
error_1_2 = maximum(abs.(u_exact_1_2.(t_plot_1_2) .- u_approx_1_2.(t_plot_1_2)))
println("Максимальная абсолютная ошибка: $error_1_2")
savefig(p1_2, "fredholm_II_ex2.png")


println("\n\n--- Часть 2: Уравнение Фредгольма I рода ---")


K_2_1(s, t) = t <= s ? 2 : 0
u_exact_2_1(s) = 1
g_2_1(s) = 2s
a_2_1, b_2_1 = 0.0, 1.0
n_2_1 = 10
t_plot_2_1 = range(a_2_1, stop=b_2_1, length=100)

println("\n--- Пример 2.1: Гауссово ядро ---")
alpha = 0.1
errors_1 = []
exact_vals_2_1 = u_exact_2_1.(t_plot_2_1)
u_approx_func = solve_fredholm_I_regularized(K_2_1, g_2_1, a_2_1, b_2_1, n_2_1, alpha)
current_error = maximum(abs.(exact_vals_2_1 .- u_approx_func.(t_plot_2_1)))
u_approx_func_direct = solve_fredholm_I_direct(K_2_1, g_2_1, a_2_1, b_2_1, n_2_1)
current_error_direct = maximum(abs.(exact_vals_2_1 .- u_approx_func_direct.(t_plot_2_1)))
println("Максимальная ошибка при методе регуляризации = $current_error")
println("Максимальная ошибка при прямом методе ", @sprintf("%.4f", current_error_direct))


best_u_approx_1 = solve_fredholm_I_regularized(K_2_1, g_2_1, a_2_1, b_2_1, n_2_1, alpha)
p2_1_reg = plot(t_plot_2_1, u_exact_2_1, label="Точное решение", lw=2)
plot!(p2_1_reg, t_plot_2_1, best_u_approx_1, label="Решение (alpha=1e-8)", ls=:dash)
savefig(p2_1_reg, "fredholm_I_regularized_1.png")

println("\n--- Пример 2.2: Ядро cos(t*x) ---")
K_2_2(t, x) = cos(t * x)
u_exact_2_2(t) = 1.0 + t
g_2_2(t) = quadgk(x -> K_2_2(t, x) * u_exact_2_2(x), 0.0, 1.0)[1]
a_2_2, b_2_2 = 0.0, 1.0
n_2_2 = 10
alpha = 1e-8
errors_2 = []
t_plot_2_2 = range(a_2_2, stop=b_2_2, length=200)
exact_vals_2_2 = u_exact_2_2.(t_plot_2_2)
u_approx_func =  solve_fredholm_I_regularized(K_2_2, g_2_2, a_2_2, b_2_2, n_2_2, alpha)
current_error = maximum(abs.(exact_vals_2_2 .- u_approx_func.(t_plot_2_2)))
u_approx_func_direct_2 = solve_fredholm_I_direct(K_2_2, g_2_2, a_2_2, b_2_2, n_2_2)
current_error_direct_2 = maximum(abs.(exact_vals_2_2 .- u_approx_func_direct_2.(t_plot_2_2)))
println("Максимальная ошибка при методе регуляризации = $current_error")
println("Максимальная ошибка при прямом методе ", @sprintf("%.4f", current_error_direct_2))
best_u_approx_2 = solve_fredholm_I_regularized(K_2_2, g_2_2, a_2_2, b_2_2, n_2_2, alpha)
p2_2 = plot(t_plot_2_2, u_exact_2_2, label="Точное решение", lw=2)
plot!(p2_2, t_plot_2_2, best_u_approx_2, label="Решение (alpha=$alpha)", ls=:dash, lw=2)
title!("Часть 2, Пример 2")
savefig(p2_2, "fredholm_I_regularized_2.png")



println("\n\n--- Часть 3: Уравнение Вольтерра II рода ---")


println("\n--- Пример 3.1: Простой полиномиальный пример ---")
K_3_1(t, x) = t*x
f_3_1(t) = t - t^4/3 
a_3_1, b_3_1 = 0.0, 2.0 
u_exact_3_1(t) = t
n_3_1 = 40

u_approx_3_1 = solve_volterra_II(K_3_1, f_3_1, a_3_1, b_3_1, n_3_1)
t_plot_3_1 = range(a_3_1, stop=b_3_1, length=200)
p3_1 = plot(t_plot_3_1, u_exact_3_1, label="Точное решение", lw=2)
plot!(p3_1, t_plot_3_1, u_approx_3_1, label="Приближенное (n=$n_3_1)", ls=:dash, lw=2)
title!("Часть 3, Пример 1 (Устойчивый)")
error_3_1 = maximum(abs.(u_exact_3_1.(t_plot_3_1) .- u_approx_3_1.(t_plot_3_1)))
println("Максимальная абсолютная ошибка: $error_3_1")
savefig(p3_1, "volterra_II_ex1.png")



println("\n--- Пример 3.2: Пример с sin(t) на коротком интервале ---")
K_3_2(t, x) = t - x

f_3_2(t) = 2*sin(t) - t

a_3_2, b_3_2 = 0.0, π/2 
u_exact_3_2(t) = sin(t)
n_3_2 = 40

u_approx_3_2 = solve_volterra_II(K_3_2, f_3_2, a_3_2, b_3_2, n_3_2)
t_plot_3_2 = range(a_3_2, stop=b_3_2, length=200)
p3_2 = plot(t_plot_3_2, u_exact_3_2, label="Точное решение", lw=2)
plot!(p3_2, t_plot_3_2, u_approx_3_2, label="Приближенное (n=$n_3_2)", ls=:dash, lw=2)
title!("Часть 3, Пример 2 (Неустойчивый, короткий интервал)")
error_3_2 = maximum(abs.(u_exact_3_2.(t_plot_3_2) .- u_approx_3_2.(t_plot_3_2)))
println("Максимальная абсолютная ошибка: $error_3_2")
savefig(p3_2, "volterra_II_ex2.png")

println("\nВсе части выполнены.")