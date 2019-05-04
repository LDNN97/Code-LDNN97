using DecisionTree
using RDatasets: dataset

iris = dataset("datasets", "iris")
features = convert(Array, iris[:, 1:4])
labels = convert(Array, iris[:, 5])

model = DecisionTreeClassifier(max_depth=2)

fit!(model, features, labels)

print_tree(model.root, 5)

predict(model, [5.9, 3.0, 5.1, 1.9])
