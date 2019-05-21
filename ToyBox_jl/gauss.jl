M = Array{Float64}([1 2 3; 2 5 2; 3 1 5])
b = Array{Float64}([14; 18; 20])

function gauss(M, b)
    M = cat(M, b, dims=2)
    # println(M)
    for i in 1:size(M, 1)
        max_i = i - 1 + argmax(M[i:size(M, 1), i])
        # println(max_i)
        if M[max_i, i] == 0
            error("det == 0")
        end
        M[i, :], M[max_i, :] = M[max_i, :], M[i, :]
        for j in i+1:size(M, 1)
            t = M[j, i] / M[i, i]
            for k in i:size(M, 2)
                M[j, k] -= M[i, k] * t
            end
        end
        # println(M)
    end
    x = zeros(size(M, 1))
    for i in size(M, 1):-1:1
        for j in i:size(M, 2)-1
            M[i, end] -= x[j] * M[i, j]
        end
        x[i] = M[i, end] / M[i, i]
        # println(i, " ", x[i])
    end
    return x
end

function LU(M, b)
    M = cat(M, b, dims=2)
    for i in 1:size(M, 1)
        max_i = i - 1 + argmax(M[i:size(M, 1), i])
        # println(max_i)
        if M[max_i, i] == 0
            error("det == 0")
        end
        M[i, :], M[max_i, :] = M[max_i, :], M[i, :]
        # println(M)
        for j in 1:i-1
            M[i, i] -= M[i, j] * M[j, i]
            # println(i, " ", j, " ", M[i, i])
        end

        for j in i+1:size(M, 2)-1
            for k in 1:i-1
                M[i, j] -= M[i, k] * M[k, j]
            end
        end

        for j in i+1:size(M, 1)
            for k in 1:i-1
                M[j, i] -= M[j, k] * M[k, i]
            end
            M[j, i] /= M[i, i]
            # println(j, " ", i, " ", M[j, i])
        end
    end

    y = M[:, end]
    # println(y)
    for i in 1:size(M, 1)
        for j in 1:i-1
            y[i] -= M[i, j] * y[j]
        end
    end
    # println(y)
    x = y
    for i in size(M, 1):-1:1
        for j in i+1:size(M, 2) - 1
            x[i] -= M[i, j] * x[j]
        end
        x[i] /= M[i, i]
    end
    return x
end

print(gauss(M, b))

print(LU(M, b))
