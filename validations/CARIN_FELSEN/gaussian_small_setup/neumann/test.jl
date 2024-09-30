
PASS = ARGS;
vvinkel = let expr = Meta.parse(PASS[1])
    @assert expr.head == :vect
    String.(expr.args)
end


print(vvinkel[1],"\n")
print(typeof(vvinkel[1]),"\n")
