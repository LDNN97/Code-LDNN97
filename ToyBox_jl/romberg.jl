module Romberg

export integral_function, calculate, romberg

abstract type integral_function end

struct f1 <: integral_function
    lb::Float64
    ub::Float64
end

function calculate(f::f1, x::Float64)
    exp(-x)
end

function romberg(f::integral_function, ε::Float64)
    k = 1
    len = f.ub - f.lb
    T_k = [(f.ub - f.lb) / 2. * (calculate(f, f.lb) + calculate(f, f.ub))]
    println(T_k)
    while true
        x = f.lb:(len/2.):f.ub
        T_k1 = Base.eval(:(zeros($(k)+1)))
        # T_k1 = zeros(1)
        T_k1[1] = T_k[1] / 2.
        for i=2:2:length(x)
            T_k1[1] += len / 2. * calculate(f, x[i])
        end
        for i=2:(k+1)
            T_k1[i] = 4^k / (4^k - 1) * T_k1[i-1] - 1 / (4^k - 1) * T_k[i-1]
            # Tk1 = 4^k / (4^k - 1) * T_k1[i-1] - 1 / (4^k - 1) * T_k[i-1]
            # append!(T_k1, Tk1)
        end
        # println(T_k1)
        if abs(T_k1[k + 1] - T_k[k]) < ε
            return T_k1[k + 1]
        end
        T_k = T_k1
        k += 1
        len /= 2.
    end
end

a = f1(0, 1)

using BenchmarkTools
@time romberg(a, 0.00001)

end  # module
