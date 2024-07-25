#Will have to test this tomorrow. 
using SpecialFunctions
using PyCall
using LinearAlgebra
using MAT

include("geo_DIR.jl")
include("geo_NEU.jl")
include("helm_DIR.jl")
include("bdry_points.jl")


PASS = ARGS;

xn = parse(Int64,PASS[2]);

fact = float(parse(Int64,PASS[3]));
over = parse(Int64, PASS[4]);

ret = parse(Int64, PASS[5]);

specimen = let expr = Meta.parse(PASS[6])
    @assert expr.head == :vect
    Int64.(expr.args)
end

Tcenter = let expr = Meta.parse(PASS[7])
    @assert expr.head == :vect
    Float64.(expr.args)
end
Tcenter =reshape(Tcenter,(2,length(specimen)) )
center = [];
for i in(1:length(specimen))
    global center=push!(center,Tcenter[:,i])
end

vvinkel = let expr = Meta.parse(PASS[8])
    @assert expr.head == :vect
    Float64.(expr.args)
end

lenny = let expr = Meta.parse(PASS[9])
    @assert expr.head == :vect
    Float64.(expr.args)
end


OPENING = parse(Int64,PASS[10]);

BC = let expr = Meta.parse(PASS[11])
    @assert expr.head == :vect
    Int64.(expr.args)
end


blocks =length(center);



if(OPENING==1)
    β=pi/4; prepre="medium" ;
elseif(OPENING==2)
    β=pi/8; prepre = "small";
elseif(OPENING==3)
    β=pi/3; prepre = "large";
end

if(BC[1]==1)
    if(vvinkel[1]==1)
        vinkel =[pi/2 for i=1:blocks]; pre="UP"
    elseif(vvinkel[1]==2)
        vinkel =[-pi/2 for i=1:blocks]; pre="DOWN"
    else
        vinkel =[0.0 for i=1:blocks]; pre="PLEASE"
    end
elseif(BC[1]==2)
    if(vvinkel[1]==1)
        vinkel =[-pi/2 for i=1:blocks]; pre="UP"
    elseif(vvinkel[1]==2)
        vinkel =[pi/2 for i=1:blocks]; pre="DOWN"
    end
end


nn=2^xn;
hx = 2*pi/nn;
TTT= [0.5*(2*j-1)*hx for j=1:nn]
pari,arc_l,normv,co,dx1,dx2,dx3 = Set_boundary_mix(specimen,BC,β);
blocks =length(center);
n_r,hx_r,PAR,DPR,NOR,MP,DX1,DX2,DX3,N_list,DPAR,DDPR,DNOR,SIN,UNOO,TT_jordan,TT_arc,TT1,TT2=set_bdry_points(nn,specimen,lenny,vinkel,BC,pari,arc_l,normv,dx1,dx2,dx3)





###saves the geometry in a matlab file, with the length of this array being blocks*n.
to_matty=zeros(blocks*xn,2);
for j in(1:blocks)
    for i in(1:xn)
        to_matty[(j-1)*xn+i,:] = PAR[j][i]
    end
end

file = matopen("geo.mat", "w")
write(file, "geo", to_matty)
close(file)

##### saves the geometry in a matlab file, but there is a NaN value between the different
##### geometries to make the plotting in matlab easier. 
to_matty=zeros(blocks*nn+blocks,2);
for j in(1:blocks)
    for i in(1:nn)
            to_matty[(j-1)*nn+i+(j-1)*1,:] = PAR[j][i];
    end
    to_matty[(nn+1)*j,:] = [NaN,NaN];
end

file = matopen("geo_with_NaN.mat", "w")
write(file, "geo_NaN", to_matty)
close(file)

