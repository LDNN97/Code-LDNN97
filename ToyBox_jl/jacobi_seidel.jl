M = Array{Float64}([8 -3 2; 4 11 -1; 6 3 12])
b = Array{Float64}([20; 33; 36])

function jacobi(M, b)
    x = rand(size(M, 1))
    for k in 1:10
        new_x = zeros(size(M, 1))
        for i in 1:size(M, 1)
            num = 0
            for j in 1:size(M, 2)
                if i != j
                    num += M[i, j] * x[j]
                end
            end
            new_x[i] = (b[i] - num) / M[i, i]
        end
        x = new_x
    end
    return x
end

function seidel(M, b)
    x = rand(size(M, 1))
    for k in 1:10
        for i in 1:size(M, 1)
            num = 0
            for j in 1:size(M, 2)
                if i != j
                    num += M[i, j] * x[j]
                end
            end
            x[i] = (b[i] - num) / M[i, i]
        end
    end
    return x
end

println(jacobi(M, b))
println(seidel(M, b))
