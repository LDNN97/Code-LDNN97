##=====

N = 10
A = zeros(10, 10)

i = eachindex(A)

for i=1:1:10, j=1:1:10
    abs(i-j) <= 1 && (A[i, j] += 1)
    i == j && (A[i, j] -= 3)
end

A

##======

function myf(x)
    k = one(x)
    for i = 1:x
        k *= i
    end
    k
end

typeof(myf(1))
typeof(myf(big(30)))

##======

using Plots;
gr()

alphas = [0.0, 0.5, 0.98]
T = 200
series = []
label = []

for alpha in alphas
    x = zeros(T + 1)
    x[1] = 0.0
    for t in 1:T
        x[t + 1] = alpha * x[t] + randn()
    end
    push!(series, x)
    push!(label, "alpha = $alpha")
end

plot(series, label=label, lw=3)

##======
r = 2.9:0.001:4; numAttract = 100
steady = ones(length(r), 1)*.25

for i = 1:400
    @. steady = r * steady * (1 - steady)
end

x = zeros(length(steady), numAttract)
x[:, 1] = steady
for i = 2:numAttract
    @. x[:, i] = r * x[:, i-1] * (1 - x[:, i - 1])
end

plot(collect(r), x, seriestype=:scatter, markersize=0.002, legend=false, color="black")

##======

struct MyRange
    start
    step
    stop
end

function _MyRange(a::MyRange, i::Int64)
    tmp = a.start + a.step * (i - 1)
    if tmp > a.stop
        error("Index is out of bound")
    else
        return tmp
    end
end

a = MyRange(1, 2, 20)
_MyRange(a, 5)

Base.getindex(a::MyRange, i::Int) = _MyRange(a, i)

a[5]

##======

struct MyLinSpace
    start
    stop
    n
end

function Base.getindex(a::MyLinSpace, i::Int)
    dx = (a.stop - a.start) / a.n
    a.start + dx * (i - 1)
end

l = MyLinSpace(1, 2, 50)

l[6]

(a::MyRange)(x) = a.start + a.step * (x - 1)

a = MyRange(1, 2, 20)

a(1.1)

##=====

abstract type AbstractPerson end
abstract type AbstractStudent <: AbstractPerson end

struct Person <: AbstractPerson
    name
end

struct Student <: AbstractStudent
    name
    grade
end

struct GraduateStudent <: AbstractStudent
    name
    grade
end

person_info(p::AbstractPerson) = println(p.name)
person_info(s::AbstractStudent) = (println(s.name); println(s.grade))

a = Person("haha")
b = Student("haha", 12)

person_info(a)

person_info(b)

##=====

macro myevalpoly(x, a...)
    isempty(x) && error("empty")
    ex = :($(a[length(a)]))
    for i in 1:length(a) - 1
        ex = :($ex * $(x)+ $(a[length(a) - i]))
    end
    println(ex)
    ex
end

@myevalpoly(2, 1, 3, 5, 7)
