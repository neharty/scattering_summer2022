#Will have to test this tomorrow. 
using SpecialFunctions
using PyCall
using LinearAlgebra
using MAT

include("geo_many.jl")
include("helm.jl")


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

BBC = let expr = Meta.parse(PASS[11])
    @assert expr.head == :vect
    Int64.(expr.args)
end


include("engine_start.jl")

blocks =length(center);



if(OPENING==1)
    β=12.0; prepre="samesame" ;
elseif(OPENING==2)
    β=24.0; prepre = "smaller";
elseif(OPENING==3)
    β=3.0; prepre = "larger";
end

    C= 1^(1/5);
    #nn = to_e(Int( floor( 2*pi*sqrt(NN/T) ) ) );  
    nn=2^(xn); 
    hx = 2*pi/nn;
    TTT= [0.5*(2*j-1)*hx for j=1:nn]
    #NN = Int( floor(C*T/(2*pi/nn)^(ord/5) )) ;

    PAR=[]; NOR=[]; DPR=[]; MP=[];
    DPAR=[]; DNOR=[]; DDPR=[]; 

    for i in(1:blocks)
        global PAR = push!(PAR,[pari[i](TTT[j],co[i],lenny[i], vinkel[i],center[i]) for j=1:nn]);
        global DPR = push!(DPR,[arc_l[i](TTT[j],co[i],lenny[i],vinkel[i]) for j=1:nn]);
        global NOR = push!(NOR,[normv[i](TTT[j],co[i],lenny[i],vinkel[i]) for j=1:nn]);
            if(length(co[i])!=0)
               global MP=push!(MP,ones(nn))
            elseif(length(co[i])==0)
                global MP=push!(MP,ones(nn))
            end
        Dtemp1=[]; Dtemp2=[];Dtemp3=[];
        for ii in(1:nn)
            Dtemp1 = vcat(Dtemp1, [experiment(TTT[ii] ,hx , nn , pari[i],co[i],lenny[i],vinkel[i],center[i])] ) ;
            Dtemp2 = vcat(Dtemp2, [Oexperiment(TTT[ii] ,hx , nn , arc_l[i],co[i],lenny[i],vinkel[i])] ) ;
            Dtemp3 = vcat(Dtemp3, [Oexperiment(TTT[ii] ,hx , nn , normv[i],co[i],lenny[i],vinkel[i])] ) ;
        end
            global DPAR=push!(DPAR,Dtemp1);
            global DDPR=push!(DDPR,Dtemp2);
            global DNOR=push!(DNOR,Dtemp3);
    end




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

