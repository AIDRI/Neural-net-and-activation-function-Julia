# Coded by Adrien I, 12/27/2020

using Statistics
using Random

mutable struct Self
    features::Array{Any, 2} #change if you have differents features
    targets::Array{Any, 1} #change if you have differents targets, but in general, this size is good
    features_for_forward::Array{Any,1}
    weights::Array{Any,1}
    bias::Array{Any,1}
    eta::Float64
    result::Array
end

function activation(z)
    1 ./ (1 .+ exp.(-z))
end

function derivative_activation(z)
    sigmoid(z) .* (1 .- sigmoid(z))
end

function for_prop(self::Self)
    self.features_for_forward = [self.features]
    for i in 1:length(self.weights)
        z = self.features_for_forward[i] * self.weights[i] .+ self.bias[i] #pre_activation of our neuron
        push!(self.features_for_forward, sigmoid(z)) #activation of the neuron
    end
    self.result = self.features_for_forward[end]
    return self
end

function back_prop(self::Self)
    nabla = (self.targets - self.result) .* sigmoid_derivative(self.result)

    self.weights[end] += self.eta * transpose(self.features_for_forward[end-1]) * nabla
    self.bias[end] .+= self.eta * mean(nabla)

    for i in 1:length(self.weights)-1 #update our gradient
        J = length(self.weights) - i
        
        cost = nabla * transpose(self.weights[J+1])
        nabla = cost .* sigmoid_derivative(self.features_for_forward[J+1])
        self.weights[J] += self.eta * transpose(self.features_for_forward[J]) * nabla
        self.bias[J] .+= self.eta * mean(nabla)
    end

    return self
end

function train(self::Self, epochs=1000)
    for i in 1:epochs
        for_prop(self) #forward propagation
        back_prop(self) #back prop
    end
end

function predict(self::Self)
    forward(self) #simple forward througout our NN to predict
end

function init_var(input_layer::Int64, hidden_layer, output_layer::Int64, eta)
    sizes = [input_layer hidden_layer output_layer]

    weights = []
    bias = []
    for i in 1:length(sizes)
        push!(weights, randn((sizes[i-1], sizes[i]))) #create weights
        push!(bias, zeros((1, sizes[i]))) #create bias
    end
   return self([], [], [], weights, bias, eta, [])
end

#=  

Example of code to generate features and targets, you just have to add it into your code
Base on an example sick/healthy (basic example to test a regression or a NN)

function gen_features(self::Self)
    input = 2
    hidden = [10, 20, 30, 20, 15, 10]
    output = 1
    eta = 0.005

    size_per_class = 200

    sick = randn((size_per_class, 2)) .+ 2         #sick > 0
    healthy = randn((size_per_class, 2)) .- 2      #healthy > 0

    features = vcat(sick, healthy)
    targets = vcat(zeros(size_per_class), ones(size_per_class))

    self.features = features
    self.targets = targets

    init_var(input, hidden, output, eta)

    return self
end


=#