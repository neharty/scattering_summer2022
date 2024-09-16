PASS = ARGS;

#vec = eval(parse(PASS[1]))
#print(vec,"this is a test")

# Second argument, a Vector{Int} in Julia-like format: "[1, 2, 3]"
v = let expr = Meta.parse(ARGS[1])
    @assert expr.head == :vect
    Float64.(expr.args)
end

print(v,"\n")

