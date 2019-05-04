module Three

using Plots

"""
    [tutorial](https://www.cnblogs.com/xpvincent/archive/2013/01/25/2877411.html)
"""
function tdma(m::Array{Float64}, d::Array{Float64})
    m[1, 2] = m[1, 2] / m[1,1]
    d[1] = d[1] / m[1, 1]

    n = length(d)
    for i=2:n-1
        m[i, i + 1] = m[i, i + 1] / (m[i, i] - m[i - 1, i] * m[i, i - 1])
        d[i] = (d[i] - d[i - 1] * m[i, i - 1]) / (m[i, i] - m[i - 1, i] * m[i, i - 1])
    end

    x = Array{Float64, 1}(undef, n)
    x[n] = (d[n] - d[n - 1] * m[n, n - 1]) / (m[n, n] - m[n - 1, n] * m[n, n - 1])
    for i=n-1:-1:1
        x[i] = d[i] - m[i, i+1] * x[i + 1]
    end

    return x
end

"""
    [tutorial](https://blog.csdn.net/u012856866/article/details/23952585)
"""
function train_cond1(t_x::Array{Float64}, t_y::Array{Float64},
                     b_d::Float64, e_d::Float64)
    n = length(t_x)

    m = zeros(n, n)
    d = zeros(n)

    h = zeros(n - 1)
    a = zeros(n - 1)
    b = zeros(n - 1)
    c = zeros(n - 1)

    for i=1:n-1
        h[i] = t_x[i + 1] - t_x[i]
    end

    for i=2:n-1
        a[i] = h[i] / (h[i - 1] + h[i])
        b[i] = h[i - 1] / (h[i - 1] + h[i])
        t1 = h[i - 1] * (t_y[i + 1] - t_y[i]) / h[i]
        t2 = h[i] * (t_y[i] - t_y[i - 1]) / h[i - 1]
        c[i] = 3 * (t1 + t2) / (h[i - 1] + h[i])
    end

    m[1,1] = 1
    m[2,2] = 2
    m[2,3] = b[2]
    for i=3:n-2
        m[i, i-1] = a[i]
        m[i, i] = 2
        m[i, i+1] = b[i]
    end
    m[n-1,n-2] = a[n-1]
    m[n-1, n-1] = 2
    m[n, n] = 1

    d[1] = b_d
    d[2] = c[2] - a[2] * b_d
    for i=3:n-2
        d[i] = c[i]
    end
    d[n-1] = c[n-1] - b[n-1] * e_d
    d[n] = e_d

    mk = tdma(m, d)
end

function predict(p_x::Float64)
    ind = searchsortedfirst(vec(t_x), p_x)

    n = length(t_x)
    h = zeros(n - 1)
    for i=1:n-1
        h[i] = t_x[i + 1] - t_x[i]
    end

    p1 = mk[ind - 1] * (p_x - t_x[ind])^2 * (p_x - t_x[ind - 1]) / h[ind - 1]^2
    p2 = mk[ind] * (p_x - t_x[ind - 1])^2 * (p_x - t_x[ind]) / h[ind - 1]^2
    p3 = t_y[ind - 1] *
    (p_x - t_x[ind])^2 * (2 * (p_x - t_x[ind - 1]) + h[ind - 1]) / h[ind - 1]^3
    p4 = t_y[ind] *
    (p_x - t_x[ind - 1])^2 * (-2 * (p_x - t_x[ind]) + h[ind - 1]) / h[ind - 1]^3
    p_y = p1 + p2 + p3 + p4
end


t_x = zeros(11)
t_y = zeros(11)
for i=1:11
    n = -5 + i - 1
    t_x[i] = n
    t_y[i] = 1 / (1 + n^2)
end
# t_x = [0.52 8 17.95 28.65 50.65 104.6 156.6 260.7 364.4 468 507 520]
# t_y = [5.28794 13.84 20.2 24.9 31.1 36.5 36.6 31 20.9 7.8 1.5 0.2]
b_d = 0.014793
e_d = -0.014793

mk = train_cond1(t_x, t_y, b_d, e_d)

x = -4.9:0.1:4.9
y1 = map(x -> 1 / (1 + x^2), x)
y2 = predict.(x)


a = plot(x, [y1, y2])
Plots.pdf(a, "result")

end  # modue Three
