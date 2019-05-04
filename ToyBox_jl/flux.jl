using Flux

f(x) = 3x^2 + 2x + 1

df(x) = Tracker.gradient(f, x; nest=true)[1]

df(2)

W = param(2)
b = param(3)
f(x) = W * x + b

grads = Tracker.gradient(() -> f(4), params(W, b))

grads[W]
grads[b]

Tracker.gradient(()->f(4), params(W, b))

W = rand(2, 5)
b = rand(2)

predict(x) = W * x .+ b

function loss(x, y)
   ŷ = predict(x)
   sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2)

loss(x, y)

W = param(W)
b = param(b)

grad = Tracker.gradient(() -> loss(x, y), params(W, b))

grad[W]
grad[b]

Δ = grad[W]

Tracker.update!(W, -0.1Δ)

loss(x, y)

struct Affine
  W
  b
end

Affine(in::Integer, out::Integer) =
  Affine(param(randn(out, in)), param(randn(out)))

# Overload call, so the object can be used as a function
(m::Affine)(x) = m.W * x .+ m.b

a = Affine(10, 5)

a(rand(10)) # => 5-element vector

Flux.@treelike Affine

model2 = Chain(
  Dense(10, 5, σ),
  Dense(5, 2),
  softmax)

model2(rand(10))

Flux.@treelike Dense(5, 2)

a = [1 2; 3 4]

b = [1 2]

b * a

c = [2, 3]

b * c

d = [1, 2]

e = b * c

dump(e)

dump(b)

c = Vector([1, 2])

dump(c)
