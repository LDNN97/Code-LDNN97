module metapractice

abstract type Bit end
struct Zero <: Bit end
struct One <: Bit end

Base.:+(::One, ::One) = Zero()

a = Zero()
b = One()

Base.promote_rule(::Type{Zero}, ::Type{One}) = One
Base.convert(::Type{One}, ::Zero) = One()
Base.:+(x::Bit, y::Bit) = +(promote(x, y)...)

@code_lowered +(Zero(), Zero())

@code_typed +(Zero(), Zero())

dump(promote(Zero(), One()))

promote_rule()

promote(Zero(), Zero())

## =====
f(x::Vararg{Integer}) = x

@code_lowered f(1, 2, 3)

@code_typed f(1, 2, 3)

f(1, 2, 3, 4, 5, 6)

Vararg{Integer, 2} <: Vararg{Integer}

methods(f)

map2(f, t) = _map2(f, t...)

_map2(f) = ()

_map2(f, x, y...) = (f(x), _map2(f, y...)...)

@code_typed map2(-, (1, 2, 3))

add(f, t) = _add(f, t...)

_add(f) = 0

_add(f, x, y...) = f(x) + _add(f, y...)

@code_typed add(+, (1, 2, 3))

add(+, (1, 2, 3))

function add2(x)
    if x == ()
        return 0
    end
    return x[1] + add2(x[2:end])
end

function add3(x)
    sum = 0
    for xx in x
        sum += xx
    end
    sum
end

add2((1, 2, 3))

using BenchmarkTools

@benchmark add2((1, 2, 3))

@benchmark add(+, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

@benchmark sum((1, 2, 3))

@benchmark add3((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

## =====
pure_function(x) = 1 + x

@code_typed pure_function(2)

Base.@pure pure_function2(x) = 1 + x

@code_typed pure_function2(2)

## =====
x = :(1 + 2);

e = quote quote $x end end

e2 = :(:($x))

eval(e2)

e3 = :(1 + 2)

e4 = :(:($$x))

e5 = :($x)

## =====
macro sayhello()
    println("Hello")
    return :( println("Hello, world!") )
end

@sayhello
@macroexpand @sayhello

macro sayhello(name)
    return :(println("Hello, world", " ", $name))
end

@macroexpand @sayhello "LDNN"

## =====
macro assert(ex, msgs...)
    msg_body = isempty(msgs) ? ex : msgs[1]
    msg = string(msg_body)
    return :($ex ? nothing : throw(AssertionError($msg)))
end

@macroexpand @assert a == b

@macroexpand @assert a==b "a should equal b!"

## =====
struct MyNumber
    x
end

for op = (:sin, :cos)
    @eval Base.$op(a::MyNumber) = MyNumber($op(a.x))
end

a = MyNumber(1)

sin(a)

## =====
@generated function sum3(t::NTuple{N, Any}) where {N}
    # This body is the "function generator"
    elements = [ :( t[$i] ) for i = 1:N ]
    expr = :( +($(elements...)) )
    return expr # The returned expression is the "generated function body"
end

@code_typed sum3((1, 2.0, pi))

@generated function add_tuples(t1::NTuple{N, Any}, t2::NTuple{N, Any}) where {N}
    # Create expression
    element1 = [:(t1[$i]) for i = 1:N]
    element2 = [:(t2[$i]) for i = 1:N]
    ex = :($(element1[1]) + $(element2[1]))
    for i = 2:N
        ex = quote ($ex)..., $(element1[i]) + $(element2[i]) end
    end
    ex
end

@code_typed add_tuples((1, 2, 3), (1.1, 2.2, 3.3))

a = add_tuples((1, 2, 3, 5), (1.1, 2.2, 3.3, 5.5))

## =====
function test()
a = 1
while true
x = :(ones($(a)))
print(x)
b = eval(x)
println(a, b)
a += 1
if a == 5
    break
end
end
end

dump(:(ones($(d))))
a = 1
b = :(ones($(:c)))
c= 1
eval(b)
b = zeros(2)
println(b)
test()

@code_lowered test()

end  # modue metapractice
