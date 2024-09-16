#Will have to test this tomorrow. 
using SpecialFunctions
using LaTeXStrings
using PyCall
using LinearAlgebra
using FFTW
using JLD
using Formatting
using IterativeSolvers
using SparseArrays

PASS = ARGS;

which = parse(Int64,PASS[1]);
order = parse(Int64,PASS[2]);

T =  float(parse(Int64,PASS[3]));

xn = parse(Int64,PASS[4]);
tn = parse(Int64,PASS[5]);

fact = float(parse(Int64,PASS[6]));
over = parse(Int64, PASS[7]);

ret = parse(Int64, PASS[8]);

modulate = parse(Float64,PASS[9]);

form = let expr = Meta.parse(PASS[10])
    @assert expr.head == :vect
    Int64.(expr.args)
end

print(length(form)," this is thee length\n")

Tcenter = let expr = Meta.parse(PASS[11])
    @assert expr.head == :vect
    Float64.(expr.args)
end
Tcenter =reshape(Tcenter,(2,length(form)) )
center = [];
for i in(1:length(form))
    global center=push!(center,Tcenter[:,i])
end

vinkel = let expr = Meta.parse(PASS[12])
    @assert expr.head == :vect
    Float64.(expr.args)
end


MOVIE = parse(Int64,PASS[13]);


w_s=1.0;

include("CQ_stuff.jl")
include("helm.jl")
include("geo_many.jl")



################## --------------------------------------- function declaration end -----------------------------------------------------

##------------------Time dependant signal-----------------

if(ret == 1)
    inny = [1,-1]; dir = "REG";
elseif(ret==2)
    inny = [0,-1]; dir = "UPDOWN";
elseif(ret==3)
    inny = [1,0];  dir = "LEFTRIGHT";
elseif(ret==4)
    θ= -pi/4;
    inny=[cos(θ),sin(θ)]; dir="REGA"
elseif(ret==5)
    θ= -pi/2; dir="ANGLE"
    inny=[cos(θ),sin(θ)];
elseif(ret==6)
    θ= 0; dir="LEFTRIGHTA"
    inny=[cos(θ),sin(θ)];
elseif(ret==11)
    θ= pi+ 7*pi/18; dir="CARIN"
    inny=[cos(θ),sin(θ)];
end



print(inny,"direction of the plane wave \n")
#------------------Kernal Definition section---------------

K0= (x,y,ny,WN)-> (1/(2*pi)) * (  dot(x-y,ny)/norm(x-y)^2 )   ;
D = (x,y,ny,WN)-> (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) )  ;
S = (x,y,ny,WN)-> (im/4) *besselh(0,1, WN*norm(x-y))  ;
C = (x,y,ny,WN)-> (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) ) -im*WN*(im/4)*besselh(0,1, WN*norm(x-y));


if(which==1)
    #Single-layer equation.
    k= (x,y,ny,WN)->(im/4) *besselh(0,1, WN*norm(x-y ));
    KK= (p,v,kc)->(im/4)*besselh(0,1,kc*norm(p)) ; 
elseif(which==2)
    #Double-Layer
    k = (x,y,ny,WN)-> (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) ) - (1/(2*pi)) * (  dot(x-y,ny)/norm(x-y)^2 );
    KK= (p,v,kc)-> (im*kc/4)*besselh(1,1,kc*norm(p)) *(dot(p,v)/norm(p));
elseif(which==3)
    #Combined-Layer
    k = (x,y,ny,WN)-> ( (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) ) - (1/(2*pi)) * (  dot(x-y,ny)/norm(x-y)^2 ) ) -
        im*WN*(im/4) *besselh(0,1, WN*norm(x-y ));
    KK=(p,v,kc)->(im*kc/4)*besselh(1,1,kc*norm(p)) *(dot(p,v)/norm(p)) -(im*kc)*(im/4)*besselh(0,1,WN*norm(p));
elseif(which==4)
    k = (x,y,ny,WN)-> (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) ) - (1/(2*pi)) * (  dot(x-y,ny)/norm(x-y)^2 );
end

#

amp=1.0;
Ω=10.0; 
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
lot=8; mu =1; 
smooth_cut=(x,t)->( (abs(retning(x,t))<lot)*exp(-mu/((retning(x,t)-lot)^2 *(retning(x,t)+lot)^2 ) ) + (abs(retning(x,t))>=lot)*(0) );

#λ0=6.4
#λλ=1.35/λ0;
λλ = 3/15;
wm = 2*pi*1/λλ;
g=(xvec,t) -> real(im/(im+(wm/4)*retning(xvec,t))^5)*smooth_cut(xvec,t);


#-------------------------

#width=3/10;
#dis=1.0;
#ϕϕ=pi/3;
#re_ce = (width/2)*[sin(ϕϕ),cos(ϕϕ)]
#center = [[dis*i/2,0]+re_ce for i=0:39]
#blocks =length(center);
#hold = [[dis*(i),0] for i=0:9];
#vinkel =[(-1)^(i-1) *(pi/2-ϕϕ) for i=1:blocks]
#typ=1;
#specimen= typ*ones(blocks)
#pari,arc_l,normv,co  = set_geo(specimen);
#lenny=(width/2)*ones(blocks);

dis=1.0;
width=λλ*1.5;
ϕϕ=pi/3;
re_ce = (width/2)*[sin(ϕϕ),cos(ϕϕ)]

center = [[dis*i/2,0]+re_ce for i=0:39]
blocks =length(center);
hold = [[dis*(i),0] for i=0:9];
vinkel =[(-1)^(i-1) *(pi/2-ϕϕ) for i=1:blocks]
typ=1;
specimen= typ*ones(blocks)
pari,arc_l,normv,co  = set_geo(specimen);
lenny=(width/2)*ones(blocks);



#--------------------DRIVER---------------------
@time begin
    #for i in(StartN:EndN)
    C= 1^(1/5);
    #nn = to_e(Int( floor( 2*pi*sqrt(NN/T) ) ) );  
    nn=2^(xn); 
    hx = 2*pi/nn;
    TTT= [0.5*(2*j-1)*hx for j=1:nn]
    #NN = Int( floor(C*T/(2*pi/nn)^(ord/5) )) ;
    NN =2^tn;  

#    if(NN%2!=0)
#        NN=NN+1;
#    end
    N = Int(over*NN); ##over was 2 now it is being passed off.   

    print("the time discr is ",NN,"\n");
    print("the space discr is ",nn,"\n");

    PAR=[]; NOR=[]; DPR=[]; MP=[];
    DPAR=[]; DNOR=[]; DDPR=[]; 

    for i in(1:blocks)
        global PAR = push!(PAR,[pari[i](TTT[j],co[i],lenny[i], vinkel[i],center[i]) for j=1:nn]);
        global DPR = push!(DPR,[arc_l[i](TTT[j],co[i],lenny[i],vinkel[i]) for j=1:nn]);
        global NOR = push!(NOR,[normv[i](TTT[j],co[i],lenny[i],vinkel[i]) for j=1:nn]);
            if(length(co[i])!=0)
               #global MP=push!(MP,[moid_prime(TTT[j],co[i]) for j=1:nn]);
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


    ########### boundary discr end 
    if (order==2)
        λ , Δt, wave_ns,P,Pinv = setupBDFM(NN,T,w_s) 
    elseif(order==3)
        #λ , Δt, Δ = setupRK3(N,T,NN);
        λ, Δt, wave_ns,P,Pinv= setupRK3(N,T,NN,fact,w_s);
    elseif(order==5)
        #λ , Δt, Δ = setupRK5(N,T,NN,fact); 
        λ, Δt, wave_ns,P,Pinv= setupRK5(N,T,NN,fact,w_s); 
    end

    if (order==2)
        DFT_Glamb = gg_setup(nn,hx,NN,NN,λ,Δt,g,[0],PAR,NOR,DPR,MP,blocks)    
    elseif(order==3)
        DFT_Glamb = gg_setup(nn,hx,NN,N,λ,Δt,g,c3,PAR,NOR,DPR,MP,blocks) 
    elseif(order==5)
        DFT_Glamb = gg_setup(nn,hx,NN,N,λ,Δt,g,c5,PAR,NOR,DPR,MP,blocks)  
    end

    phi_hat = complex( zeros( size(DFT_Glamb) ) );
    choice = 0; #reference to the quadrature rule. Will not change from alpert. 

    if(which==1)
        phi_hat= den_sol(choice ,nn, N,λ,hx,Δt,wave_ns,P,Pinv,DFT_Glamb,k,K0,D,which,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,order)  
    elseif(which==2)
        phi_hat= den_sol(choice ,nn, N,λ,hx,Δt,wave_ns,P,Pinv,DFT_Glamb,k,K0,D,which,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,order)  
    elseif(which==3)
        phi_hat= den_sol(choice ,nn, N,λ,hx,Δt,wave_ns,P,Pinv,DFT_Glamb,k,K0,C,which,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,order)  
    elseif(which==4)
        phi_hat= den_sol(choice ,nn, N,λ,hx,Δt,wave_ns,P,Pinv,DFT_Glamb,k,K0,D,which,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,order) 
    end

    print(NN," total time steps \n")
    print(fact," this is the value of fact by the by\n\n")
    print(over," this is the value of over by the by\n\n")

    #end

    ##discretization



    if(MOVIE==0)
#       
        #local z= [[0,(10/3)*width *35.5]];
        #local z= [[0,dis*5/3]];
        local z= [[0,17.5]];
        #local z= [[0,17.5]];
    
        #local z= [[4.9,width*5/3]];
        
        print(z,"\n")
        print(length(z),"\n")

        #local z = vcat(points1,points2)
        obs = zeros( NN+1,length(z) );
        inc_f=zeros(NN+1,length(z));
       
        #calculates in inc field at the obs point for saving. 
        for i in(1:length(z))
            inc_f[:,i] = [  g( z[i] , Δt*j ) for j=0: NN];
        end

        if(order==2) 
            for i in(1:length(z))
                obs[:,i]= time_freq_sol_BDFM2(NN,nn , λ,z[i] , wave_ns ,phi_hat ,PAR,NOR,DPR , KK,blocks); 
            end
        elseif(order==3||order==5)
            for i in(1:length(z))
                obs[:,i] = time_freq_sol(N,nn , λ, Δt ,z[i] ,phi_hat,wave_ns,P,Pinv,PAR,NOR,DPR, KK,blocks,w_s )[1:NN+1] ;
            end
        end

        td = T/NN;
        ddt=[td*i for i=0:NN];
        post=string(NN);


        #save("z"*string(z[2])*"_40closeramp"*string(amp)*"_mod"*string(modulate)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(ord)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(form)*string(order)*"_"*string(which)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".jld","data",obs);
        save("z17.5_40closeramp"*string(amp)*"_mod"*string(modulate)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(ord)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(form)*string(order)*"_"*string(which)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".jld","data",obs);

    elseif(MOVIE==1)
        ##outputs the entire thing as a JLD file so it can be made into an MP4 movie
        Ngridlen = 2^8; len=4;
        y = range(-len,length=Ngridlen, stop=len);
        x = range(-len,length=Ngridlen, stop=len);
        X = repeat(reshape(x, 1, :), length(y), 1)
        Y = repeat(y, 1, length(x))
        obs = complex(zeros( NN+1, Ngridlen,Ngridlen));

        if(order==2)
            for ii in(1:Ngridlen)
                for jj in(1:Ngridlen)
                    z=[X[ii,jj],Y[ii,jj]];
                    temp = [  g( z , Δt*j ) for j=0: NN];
                    obs[:,ii,jj]= time_freq_sol_BDFM2(NN,nn , λ,z[i] , wave_ns ,phi_hat ,PAR,NOR,DPR , KK,blocks)+temp; 
                end
            end
        elseif(order==3||order==5)
            for ii in(1:Ngridlen)
                for jj in(1:Ngridlen)
                    z=[X[ii,jj],Y[ii,jj]];
                    temp = [  g( z , Δt*j ) for j=0: NN];
                    obs[:,ii,jj] = time_freq_sol(N,nn , λ, Δt ,z[i] ,phi_hat,wave_ns,P,Pinv,PAR,NOR,DPR, KK,blocks,w_s )[1:NN+1] +temp;
                end
            end
        end

        save("MOVamp"*string(amp)*"_mod"*string(modulate)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(ord)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(form)*string(order)*"_"*string(which)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".jld","data",obs);

    end



end
####----------------Driver END-----------------

