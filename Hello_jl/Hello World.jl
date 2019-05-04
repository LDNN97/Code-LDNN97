abstract type TypeA end

struct TypeB <: TypeA end
struct TypeC <: TypeA end

wooo(a1::TypeA, a2::TypeA) = println("A/A")
wooo(a::TypeA, b::TypeB) = println("A/B")

callme(a1::TypeA, a2::TypeA) = wooo(a1, a2)

b = TypeB(); c = TypeB();

callme(c, b)

mutable struct my
    a :: Int32
end

a = my(1)

dump(a)

fieldnames(my)

typeof(a)

Any <: DataType

DataType <: Any

isa(Any, DataType)

isa(DataType, Any)

dump(DataType)

typeof(Any)

Any <: DataType

typeof(DataType)

mutable struct Point{T}
    x::T
end

a = Point{Int32}(1)

a.x = .3

typeof(Any)

subtypes(Number)

subtypes(Real)

dump(Integer)

dump(Complex)

dump(Rational)

abstract type Pointy{T} end

mutable struct Point1D{T} <: Pointy{T}
    x::T
end

mutable struct Point2D{T} <: Pointy{T}
    x::T
    y::T
end

mutable struct Point3D{T} <: Pointy{T}
    x::T
    y::T
    z::T
end

function module_t(p::Pointy)
    m = 0
    for field in fieldnames(typeof(p))
        m += getfield(p, field)^2
    end
    sqrt(m)
end

a = Point1D(1)
module_t(Point1D{Int64}(1))

b = Point3D(1, 2, 3)
module_t(b)

subtypes(Type{})

supertype(Type{})

subtypes(DataType)

typeof(Any)

typeof(DataType)

DataType <: Any

Any <: DataType

Type <: Any

dump(Point3D)
mutable struct aa{Int32 <: T1 <: supertype(Int32), Float32 <: T2 <: Real}
    x :: T1
    y :: T2
end

dump(aa)

function addtwo(x, y)
    x + y
end

addtwo(1, 2)

function bar(; x=1, y=x+3, z=x+y)
    println(x)
    println(y)
    println(z)
end

bar(x=2)

using SparseArrays

II = [1, 4, 3, 5]
J = [4, 7, 18, 9]
V = [1, 2, -5, 3]

S = sparse(II, J, V)

A = [3, 2, 1]

sort(A)

A

A = "\u2200x\u2203y"

B = A[1:4]

Fs = Vector{Any}(undef, 2); i = 1;

while i <= 2
    Fs[i] = i
    global i += 1
end

Fs[1]
Fs[2]

while i <= 2
    let i = i
        Fs[i] = i
    end
    global i += 1
end

Fs[1]
Fs[2]

using Distributed

const jobs = Channel{Int}(32)
const results = Channel{Tuple}(32)

function do_work()
    for job_id in jobs
        exec_time = rand()
        sleep(exec_time)
        put!(results, (job_id, exec_time))
    end
end

function make_jobs(n)
    for i in 1:n
        put!(jobs, i)
    end
end

n = 12

@async make_jobs(n)

for i in 1:4
    @async do_work()
end

print(n)

a = 100
@elapsed while a > 0
    global a = a - 1
end

while n > 0
    job_id, exec_time = take!(results)
    println("$job_id finished in $(round(exec_time, digits=2)) seconds")
    global n = n - 1
end

struct MyNumber
    x::Float64
end

for op = (:sin, :cos, :tan, :log, :exp)
    eval(quote
        Base.$op(a::MyNumber) = MyNumber($op(a.x))
    end)
end

x = MyNumber(1)

sin(x)
