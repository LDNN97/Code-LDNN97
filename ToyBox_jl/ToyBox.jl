module ToyBox

st = "LMNBCTRAGP"

for i in 1:length(st) - 2
    print(st[i:i+2], " ")
    tot = (st[i] - 'L')^2 + (st[i + 1] - 'D')^2 + (st[i + 2] - 'N')^2
    println(tot)
end

mutable struct my
    a::Float32
    b::Float32
    my(x) = begin
        y = x
        y2 = x * 2
        new(y, y2)
    end
end

a = my(2)
a.a
a.b
end # module
