#=
Be careful, these functions are not all coded to be used in a neural network, especially the functions that return 0 or 1. 
They return a value and not an array. 
However, you can easily modify them and adapt them to your project, as they remain very general.

Coded by Adrien I, 12/27/2020
=#

function sigmoid(z)
    return 1 ./ (1 .+ exp.(-z))
end

function sigmoid_derivative(z)
    return sigmoid(z) .* (1 .- sigmoid(z))
end

################################

function tanh(z)
    return 2 ./ ((1 .+ exp.(-2 .* z)) .+ 1 )
end

function tanh_derivative(z)
    return 1 .- tanh(z) .^ 2
end

################################

function relu(x)
    if x < 0
        return 0
    end
    if x >= 0
        return x
    end
end

function derivative_relu(x)
    if x < 0
        return 0
    end
    if x >= 0
        return 1
    end
end

################################

function prelu(alpha, x)
    if x < 0
        return alpha .* x
    end
    if x >= 0
        return x
    end
end

function derivative_prelu(alpha, x)
    if x < 0
        return alpha
    end
    if x >= 0
        return 1
    end
end

################################

function softmax(x)
    return exp.(x) ./ sum(exp.(x))
end

function derivative_softmax(x)
    return (exp.(x[1]) .* (exp.(x[2]) .+ exp.(x[3]))) ./ ((exp.(x[1]) .+ exp.(x[2]) .+ exp.(x[3])) .^ 2 )
end

################################

function elu(alpha, x)
    if x < 0
        return alpha .* (exp.(x) .- 1 )
    end
    if x >= 0
        return x
    end
end

function derivative_elu((alpha, x))
    if x < 0
        return elu(alpha, x) .+ alpha
    end
    if x >= 0
        return 1
    end
end

################################

function identity(x)
    return x
end

function derivative_identity(x)
    return 1
end

################################

function binary(x)
    if x < 0
        return 0
    end
    if x >= 0
        return 1
    end
end

function derivative_binary(x)
    if x != 0
        return 0
    end
    if x == 0
        return nothing
    end
end
