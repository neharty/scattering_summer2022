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

Which = parse(Int64,PASS[1]);
order = parse(Int64,PASS[2]);

T =  float(parse(Int64,PASS[3]));

xn = parse(Int64,PASS[4]);
tn = parse(Int64,PASS[5]);

fact = float(parse(Int64,PASS[6]));
over = parse(Int64, PASS[7]);

ret = parse(Int64, PASS[8]);

modulate = parse(Float64,PASS[9]);

specimen = let expr = Meta.parse(PASS[10])
    @assert expr.head == :vect
    Int64.(expr.args)
end

print(length(specimen)," this is thee length\n")

Tcenter = let expr = Meta.parse(PASS[11])
    @assert expr.head == :vect
    Float64.(expr.args)
end
Tcenter =reshape(Tcenter,(2,length(specimen)) )
center = [];
for i in(1:length(specimen))
    global center=push!(center,Tcenter[:,i])
end

vvinkel = let expr = Meta.parse(PASS[12])
    @assert expr.head == :vect
    Float64.(expr.args)
end

lenny = let expr = Meta.parse(PASS[13])
    @assert expr.head == :vect
    Float64.(expr.args)
end



MOVIE = parse(Int64,PASS[14]);




print(center," center\n")
print(lenny," lenny\n")

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
end




sin_shift=(x,co,LL,angle,shift)->sin(x);



amp=1.0;
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
lot=4; mu =1; 
smooth_cut=(x,t)->( (abs(retning(x,t))<lot)*exp(-mu/((retning(x,t)-lot)^2 *(retning(x,t)+lot)^2 ) ) + (abs(retning(x,t))>=lot)*(0) );

Ω=6; 
λλ=0.8;#2.8;
wm = 2*pi*1/λλ;
g=(xvec,t) -> real(im/(im+(wm/4)*retning(xvec,t))^5)#*smooth_cut(xvec,t);
gp =(xvec,t,norr)-> real((5/4)*wm*im/(im+(wm/4)*retning(xvec,t))^6)#*smooth_cut(xvec,t);



#########################



print(inny,"direction of the plane wave \n")
#------------------Kernal Definition section---------------
#################################################################
#################################################################
#################################################################
#################################################################


#D = (x,y,ny,WN)-> (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) )   #In the strip case there is a 1/2 instead of 1/4 in cats code. Im leaving it at 1/4.
#S = (x,y,ny,WN)-> (im/4) *besselh(0,1, WN*norm(x-y)) 
#C = (x,y,ny,WN)-> D(x,y,ny) -im*WN*S(x,y,ny)


K0= (x,y,ny)-> (1/(2*pi)) * (  dot(x-y,ny)/norm(x-y)^2 ) 

################### Kernel Neumann Section #################################
Gk0 =(px,py,nx,ny,WN)->-2( (-im*WN^2 /4)*besselh(0,1,WN*norm( px-py ))*dot(nx,px-py)*dot(ny,px-py)/norm(px-py)^2  
    + (im*WN/4 *norm(px-py)*besselh(1,1,WN*norm(px-py )) -1/(2*pi) )*
    ( 2*dot(px-py,nx)*dot(px-py,ny)/norm(px-py)^4  -dot(nx,ny)/norm(px-py)^2 )
     ) ; 
#Trig integration. 
N__jordan=  (x,y,px,py,ppx) -> -2*dot(ppx,px -py)/norm(px-py)^2 -cot( (y-x)/2); 
N__openArc= (x,y,px,py,px_p,py_p) ->  ( -dot(px_p,py_p)/norm(px-py)^2 +2*(dot(px-py,px_p)*dot(px-py,py_p))/norm(px-py)^4  - 1/(x-y)^2 );   

## for the off block diagonal terms for the double layer;
Gk = (px,py,nx,ny,WN)-> 2*( (1/4)*(im*WN^2 )*besselh(0,1,WN*norm( px-py )) *(dot(nx,px-py)*dot(ny,px-py))/norm(px-py)^2 + 
    (im*WN/4)*besselh(1,1,WN*norm(px-py))*( dot(nx,ny)/norm(px-py) -2*dot(nx,px-py)*dot(ny,px-py)/norm(px-py)^3 )  ) ; 


Ck = (px,py,nx,ny,WN)-> 2*( (1/4)*(im*WN^2 )*besselh(0,1,WN*norm( px-py )) *(dot(nx,px-py)*dot(ny,px-py))/norm(px-py)^2 + 
        (im*WN/4)*besselh(1,1,WN*norm(px-py))*( dot(nx,ny)/norm(px-py) -2*dot(nx,px-py)*dot(ny,px-py)/norm(px-py)^3 )  ) -
        im*WN*( -(im*WN/2)* besselh(1,1,WN*norm(px-py ))*(  dot(px-py,nx)/norm(px-py) )  );


################## for corner purposes #########################
η=im;
Sη = (x,y,ny,nx,WN)-> (im/4) *besselh(0,1, η*norm(x-y))                                                        # No wavenumber here. Can precompute       
Kη = (px,py,nx,ny,WN)->( (im*η/4)* besselh(1,1,η*norm(px-py ))*(  dot(px-py,ny)/norm(px-py) )  );              # No wavenumber here. Can precompute
GkGη = (px,py,nx,ny,WN)-> ( (1/4)*(im*WN^2 )*besselh(0,1,WN*norm( px-py )) *(dot(nx,px-py)*dot(ny,px-py))/norm(px-py)^2 + 
        (im*WN/4)*besselh(1,1,WN*norm(px-py))*( dot(nx,ny)/norm(px-py) -2*dot(nx,px-py)*dot(ny,px-py)/norm(px-py)^3 )  ) -
                        ((1/4)*(im*η^2 )*besselh(0,1,η*norm( px-py )) *(dot(nx,px-py)*dot(ny,px-py))/norm(px-py)^2 + 
        (im*η/4)*besselh(1,1,η*norm(px-py))*( dot(nx,ny)/norm(px-py) -2*dot(nx,px-py)*dot(ny,px-py)/norm(px-py)^3 )  ) ;       # Cannot precompute 

Kk= (px,py,nx,ny,WN)->( (im*WN/4)* besselh(1,1,WN*norm(px-py ))*(  dot(px-py,ny)/norm(px-py) )  );
##################################


#this paramter sets which layer representation I am using;

if(Which==1)
    k= (px,py,nx,ny,WN)->( -(im*WN/2)* besselh(1,1,WN*norm(px-py ))*(  dot(px-py,nx)/norm(px-py) )  );        
    KK = (x,y,ny,WN)-> (im/4) *besselh(0,1, WN*norm(x-y))
elseif(Which==2)
    k=(px,py,nx,ny,WN)-> -2*( (-im*WN^2 /4)*besselh(0,1,WN*norm( px-py ))*dot(nx,px-py)*dot(ny,px-py)/norm(px-py)^2  ##There is a two here not in the strip case. 
        + (im*WN/4 *norm(px-py)*besselh(1,1,WN*norm(px-py )) -1/(2*pi) )*
        ( 2*dot(px-py,nx)*dot(px-py,ny)/norm(px-py)^4  -dot(nx,ny)/norm(px-py)^2 ) ) ; 
    KK = (x,y,ny,WN)-> (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) )
elseif(Which==3)
    k=(px,py,nx,ny,WN)-> -2( (-im*WN^2 /4)*besselh(0,1,WN*norm( px-py ))*dot(nx,px-py)*dot(ny,px-py)/norm(px-py)^2  
        + (im*WN/4 *norm(px-py)*besselh(1,1,WN*norm(px-py )) -1/(2*pi) )*
        ( 2*dot(px-py,nx)*dot(px-py,ny)/norm(px-py)^4  -dot(nx,ny)/norm(px-py)^2 ) ) -
        im*WN*( -(im*WN/2)* besselh(1,1,WN*norm(px-py ))*(  dot(px-py,nx)/norm(px-py) )  );
end


#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################



###### Aux setup for the geometry ##########
#center = [[1.5*i,1.5*j] for i=-2:2,j=-1:1];
blocks =length(center);
#typ=1;
#test= typ*ones(blocks)
pari,arc_l,normv,co = set_geo(specimen);
blocks =length(center);

lenny = [1.0,0.25];

if(vvinkel[1]==1)
    vinkel =[-pi/2 for i=1:blocks]; pre="UP"
elseif(vvinkel[1]==2)
    vinkel =[+pi/2 for i=1:blocks]; pre="DOWN"
end







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

    N = Int(over*NN); ##over was 2 now it is being passed off.   

    print("the time discr is ",NN,"\n");
    print("the space discr is ",nn,"\n");

##################################################################
    n_r=Int(nn/2)-1; hx_r = 2*pi/(nn);

    PAR = [];
    DPR = [];
    NOR = [];
    MP=[];

    DPAR=[]; DNOR=[]; DDPR=[];
    DX1 =[]; DX2 =[]; DX3 =[];
    SIN=[];
    N_list = []
    pari,arc_l,normv,co, dx1,dx2, dx3 = set_geo(specimen);
    TT_jordan = [0.5*(2*j-1)*hx for j=1:nn];
    TT_arc = [(j)*hx_r for j=0:nn];
    TT1= [cos(j*hx_r) for j=0:nn]
    TT2= [sin(j*hx_r) for j=0:nn];
    UNOO=ones(nn);

    ## Discretization section
    for i in(1:blocks) 
        if(specimen[i]>=4)
            global PAR = push!(PAR,[pari[i](TT_jordan[j],co[i],lenny[i], vinkel[i],center[i]) for j=1:nn]);
            global DPR = push!(DPR,[arc_l[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:nn]);
            global NOR = push!(NOR,[normv[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:nn]);

            global DX1 = push!(DX1,[dx1[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:nn])
            global DX2 = push!(DX2,[dx2[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:nn])
            global DX3 = push!(DX3,[dx3[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:nn])
            global N_list = push!(N_list,nn);

            if(length(co[i])!=0)
                global MP=push!(MP,[moid_prime(TT_jordan[j],co[i]) for j=1:nn]);
            elseif(length(co[i])==0)
                global MP=push!(MP,ones(nn))
            end
                Dtemp1=[]; Dtemp2=[];Dtemp3=[];Dtemp4=[]
                for ii in(1:nn)
                    Dtemp1 = vcat(Dtemp1, [experiment(TT_jordan[ii] ,hx , nn , pari[i],co[i],lenny[i],vinkel[i],center[i])] ) ;
                    Dtemp2 = vcat(Dtemp2, [Oexperiment(TT_jordan[ii] ,hx , nn , arc_l[i],co[i],lenny[i],vinkel[i])] ) ;
                    Dtemp3 = vcat(Dtemp3, [Oexperiment(TT_jordan[ii] ,hx , nn , normv[i],co[i],lenny[i],vinkel[i])] ) ;
                end
            global DPAR=push!(DPAR,Dtemp1);
            global DDPR=push!(DDPR,Dtemp2);
            global DNOR=push!(DNOR,Dtemp3);
        elseif(specimen[i]<4)

            global PAR = push!(PAR,[pari[i](TT_arc[j+1],co[i],lenny[i], vinkel[i],center[i]) for j=1:n_r]);
            global DPR = push!(DPR,[arc_l[i](TT_arc[j+1],co[i],lenny[i],vinkel[i]) for j=1:n_r]);
            global NOR = push!(NOR,[normv[i](TT_arc[j+1],co[i],lenny[i],vinkel[i]) for j=1:n_r]);

            global DX1 = push!(DX1,[dx1[i](TT_arc[j+1],co[i],lenny[i],vinkel[i]) for j=1:n_r])
            global DX2 = push!(DX2,[dx2[i](TT_arc[j+1],co[i],lenny[i],vinkel[i]) for j=1:n_r])
            global DX3 = push!(DX3,[dx3[i](TT_arc[j+1],co[i],lenny[i],vinkel[i]) for j=1:n_r])
            global N_list = push!(N_list,n_r);

            global MP=push!(MP,ones(nn))

            Dtemp1=[]; Dtemp2=[];Dtemp3=[];Dtemp4=[]
            for ii in(1:n_r)
                Dtemp1 = vcat(Dtemp1, [experiment(TT_arc[ii+1] ,hx_r , n_r , pari[i],co[i],lenny[i],vinkel[i],center[i])] ) ;
                Dtemp2 = vcat(Dtemp2, [Oexperiment(TT_arc[ii+1] ,hx_r , n_r , arc_l[i],co[i],lenny[i],vinkel[i])] ) ;
                Dtemp3 = vcat(Dtemp3, [Oexperiment(TT_arc[ii+1] ,hx_r , n_r , normv[i],co[i],lenny[i],vinkel[i])] ) ;
                Dtemp4 = vcat(Dtemp4, [experiment(TT_arc[ii+1] ,hx_r , n_r , sin_shift,co[i],lenny[i],vinkel[i],center[i])] ) ;
            end

            global DPAR=push!(DPAR,Dtemp1);
            global DDPR=push!(DDPR,Dtemp2);
            global DNOR=push!(DNOR,Dtemp3);
            global SIN=push!(SIN,Dtemp4);
        end
    end

########################################################################################################################
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
        DFT_G,DFT_GP = gg_setup(nn,n_r,hx,NN,NN,λ,Δt,g,gp,[0],PAR,NOR,DPR,MP,blocks,specimen,co,Which)  
#        print(DFT_GP," boundary done  \n")
    elseif(order==3)
        DFT_G,DFT_GP = gg_setup(nn,n_r,hx,NN,N,λ,Δt,g,gp,c3,PAR,NOR,DPR,MP,blocks,specimen,co,Which) 
    elseif(order==5)
        DFT_G,DFT_GP = gg_setup(nn,n_r,hx,NN,N,λ,Δt,g,gp,c5,PAR,NOR,DPR,MP,blocks,specimen,co,Which)  
    end

    phi_hat = complex( zeros( size(DFT_GP) ) );
    choice = 0; #reference to the quadrature rule. Will not change from alpert. 

    if(Which==1)
        phi_hat= den_sol_S(choice ,nn, N,λ,hx,Δt,wave_ns,P,Pinv,DFT_GP,k,Which,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,order) 
        print(" density done \n")
    elseif(Which==2)
        phi_hat= den_sol_DC(choice ,nn,n_r, N,λ,hx,Δt,wave_ns,P,Pinv,DFT_G,DFT_GP,k,Gk,N__jordan,N__openArc,Sη,Kη,GkGη,Kk,Which,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,DX1,DX2,DX3,blocks,co,order,UNOO,TT1,TT2,specimen,N_list) 
    elseif(Which==3)
    end


    if(MOVIE==0)
        local z= [[0,(10/3)*width *35.5]];
        obs = zeros( NN+1,length(z) );
        inc_f=zeros(NN+1,length(z));
       
        #calculates in inc field at the obs point for saving. 
        for i in(1:length(z))
            inc_f[:,i] = [  g( z[i] , Δt*j ) for j=0: NN];
        end

        if(order==2) 
            for i in(1:length(z))
                obs[:,i]= time_freq_sol_BDFM2(hx,NN,nn , λ,z[i] , wave_ns ,phi_hat ,PAR,NOR,DPR , KK,blocks,specimen); 
            end
        elseif(order==3||order==5)
            for i in(1:length(z))
                obs[:,i] = time_freq_sol(hx,N,nn , λ, Δt ,z[i] ,phi_hat,wave_ns,P,Pinv,PAR,NOR,DPR, KK,blocks,w_s,specimen )[1:NN+1] ;
            end
        end

#        td = T/NN;
#        ddt=[td*i for i=0:NN];
#        name = "amp"*string(amp)*"_mod"*string(modulate)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(ord)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(form)*string(order)*"_"*string(which)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir;
#        ##Writes the scattered field in ascii
#        open(name*".txt", "w") do f
#            for i in 1:NN+1
#                n1 = format(ddt[i]);
#                holder="";
#                for j in(1:length(obs[1,:]))
#                    holder = holder*string(obs[i,j])*" "; 
#                end
#                write(f, "$n1 $holder \n")
#            end
#        end

### the one below is what I should use. 

#        post=string(NN);
#
#        #name ="thresh_erli_rfft_strip_"*string(T)*"_"*post
#        name ="erli_rfft_strip_"*string(T)*"_"*post
#        for j in(1:length(z)) #there is a mistake here. 
#            open("col"*string(j)*name*".txt", "w") do f
#                       for i in 1:NN+1
#                            n1 = format(ddt[i]);
#                            holder="";
#                            holder = holder*string(obs[i,j])*" ";
#                            holder2= string(inc_f[i,j]) 
#                          write(f, "$n1 $holder $holder2 \n")
#                       end
#                   end
#        end


        save("ALTamp"*string(amp)*"_mod"*string(modulate)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(order)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(order)*"_"*string(Which)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".jld","data",obs)

    elseif(MOVIE==1)
        ##outputs the entire thing as a JLD file so it can be made into an MP4 movie
        Ngridlen = 2^8; lenny=4;
        y = range(-lenny,length=Ngridlen, stop=lenny);
        x = range(-lenny,length=Ngridlen, stop=lenny);
        X = repeat(reshape(x, 1, :), length(y), 1)
        Y = repeat(y, 1, length(x))
        obs = complex(zeros( NN+1, Ngridlen,Ngridlen));

        if(order==2)
            for ii in(1:Ngridlen)
                for jj in(1:Ngridlen)
                    z=[X[ii,jj],Y[ii,jj]];
                    temp = [  g( z , Δt*j ) for j=0: NN];
                    obs[:,ii,jj]= time_freq_sol_BDFM2(hx,NN,nn , λ,z , wave_ns ,phi_hat ,PAR,NOR,DPR , KK,blocks,specimen)+temp; 
                                  #time_freq_sol_BDFM2(NN,nn , λ,z , wave_ns ,phi_hat ,PAR,NOR,DPR , KK,blocks,specimen)+temp; 
                end
            end
        elseif(order==3||order==5)
            for ii in(1:Ngridlen)
                for jj in(1:Ngridlen)
                    z=[X[ii,jj],Y[ii,jj]];
                    temp = [  g( z , Δt*j ) for j=0: NN];
                    obs[:,ii,jj] = time_freq_sol(hx,N,nn , λ, Δt ,z ,phi_hat,wave_ns,P,Pinv,PAR,NOR,DPR, KK,blocks,w_s,specimen )[1:NN+1] +temp ;  
                                   #time_freq_sol(N,nn , λ, Δt ,z ,phi_hat,wave_ns,P,Pinv,PAR,NOR,DPR, KK,blocks,w_s,specimen)[1:NN+1] +temp;
                end
            end
        end

        save("OpulseMOVamp"*string(amp)*"_mod"*string(modulate)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(ord)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(specimen)*string(order)*"_"*string(Which)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".jld","data",obs);

    end



end
####----------------Driver END-----------------

