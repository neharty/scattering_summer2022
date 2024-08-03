#Will have to test this tomorrow. 
using SpecialFunctions
using LaTeXStrings
using PyCall
using LinearAlgebra
using FFTW
#using JLD
using Formatting
using IterativeSolvers
using SparseArrays
using DelimitedFiles
using Printf
using PyPlot

Base.show(io::IO, f::Float64) = @printf(io, "%.18f", f)

##these will need to be updated. 
include("geo_DIR.jl")
include("geo_NEU.jl")
include("helm_DIR.jl")
#include("helm_NEU.jl")
include("helm_MIX.jl")
include("bdry_points.jl")
include("mix_kernals.jl")
include("CQ_MIX.jl")
include("engine_start.jl")

PASS = ARGS;

layer_rep = parse(Int64,PASS[1]);
order = parse(Int64,PASS[2]);

T =  float(parse(Int64,PASS[3]));

xn = parse(Int64,PASS[4]);
tn = parse(Int64,PASS[5]);

fact = float(parse(Int64,PASS[6]));
over = parse(Int64, PASS[7]);

ret = parse(Int64, PASS[8]);

specimen = let expr = Meta.parse(PASS[9])
    @assert expr.head == :vect
    Int64.(expr.args)
end

print(length(specimen)," this is thee length\n")

Tcenter = let expr = Meta.parse(PASS[10])
    @assert expr.head == :vect
    Float64.(expr.args)
end
Tcenter =reshape(Tcenter,(2,length(specimen)) )
center = [];
for i in(1:length(specimen))
    global center=push!(center,Tcenter[:,i])
end

vvinkel = let expr = Meta.parse(PASS[11])
    @assert expr.head == :vect
    Float64.(expr.args)
end

lenny = let expr = Meta.parse(PASS[12])
    @assert expr.head == :vect
    Float64.(expr.args)
end

MOVIE = parse(Int64,PASS[13]);

LAYER = let expr = Meta.parse(PASS[14])
    @assert expr.head == :vect
    String.(expr.args)
end

BC = let expr = Meta.parse(PASS[15])
    @assert expr.head == :vect
    Int64.(expr.args)
end


OPENING = parse(Int64,PASS[16]);

if(OPENING==1)
    β=pi/4; prepre="medium" ;
elseif(OPENING==2)
    β=pi/8; prepre = "small";
elseif(OPENING==3)
    β=pi/3; prepre = "large";
end



print(layer_rep," layer rep\n")
print(center," center\n")
print(specimen, "these are the types of scatteres \n")

## Wavespeed 
w_s=1.0;


################## --------------------------------------- function declaration end -----------------------------------------------------

##------------------Time dependant signal-----------------

if(ret == 1)
    inny = [1,-1]; dir = "REG";
elseif(ret==2)
    inny = [0,-1]; dir = "UPDOWN";
elseif(ret==3)
    inny = [1,0];  dir = "LEFTRIGHT";
elseif(ret==4)
    θ= -pi/3;
    inny=[cos(θ),sin(θ)]; dir="REGA"
elseif(ret==5)
    θ= -pi/2; dir="ANGLE"
    inny=[cos(θ),sin(θ)];
elseif(ret==6)
    θ= pi; dir="RIGHTLEFTA"
    inny=[cos(θ),sin(θ)];
end





Gbdr = Function[];
Gbdr = push!(Gbdr,g,gp);
#########################


print(inny,"direction of the plane wave \n")
#------------------Kernal Definition section---------------
#################################################################
#################################################################
#################################################################





K_rep=Function[];
KK=Function[];

if(layer_rep==1)
    K_rep = push!(K_rep,S,(px,py,nx,ny,WN)->( -(im*WN/2)* besselh(1,1,WN*norm(px-py ))*(  dot(px-py,nx)/norm(px-py) )  ));
    KK= push!(KK,S,S);
elseif(layer_rep==2);
    K_rep = push!(K_rep,D,Gk);
    KK= push!(KK,D,D)
end

Ψ = SET_KERNEL(BC,layer_rep);

#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################



###### Aux setup for the geometry ##########
blocks =length(center);
pari,arc_l,normv,co,dx1,dx2,dx3 = Set_boundary_mix(specimen,BC,β);
blocks =length(center);

if(BC[1]==1)
    if(vvinkel[1]==1)
        vinkel =[pi/7 for i=1:blocks]; pre="some_name1"
    elseif(vvinkel[1]==2)
        vinkel =[pi/5 for i=1:blocks]; pre="some_name2"
    elseif(vvinkel[1]==3)
        vinkel =[pi/3 for i=1:blocks]; pre="some_name3"
    elseif(vvinkel[1]==4)
        vinkel =[pi/9 for i=1:blocks]; pre="some_name4"
    elseif(vvinkel[1]==5)
        vinkel =[pi/4 for i=1:blocks]; pre="some_name5"
    elseif(vvinkel[1]==6)
        vinkel =[pi/6 for i=1:blocks]; pre="some_name6"
    elseif(vvinkel[1]==7)
        vinkel =[pi/8 for i=1:blocks]; pre="some_name7"
    elseif(vvinkel[1]==8)
        vinkel =[pi/10 for i=1:blocks]; pre="some_name8"
    elseif(vvinkel[1]==9)
        vinkel =[pi/12 for i=1:blocks]; pre="some_name9"
    elseif(vvinkel[1]==10)
        vinkel =[pi/2 for i=1:blocks]; pre="some_name10"    
    else
#        vinkel =[float(pi) for i=1:blocks]; pre="PLEASE"
        vinkel =[0.0 for i=1:blocks]; pre="PLEASE"
    end
elseif(BC[1]==2)
    if(vvinkel[1]==1)
        vinkel =[pi/7 + float(pi) for i=1:blocks]; pre="some_name1"
    elseif(vvinkel[1]==2)
        vinkel =[pi/5 + float(pi) for i=1:blocks]; pre="some_name2"
    elseif(vvinkel[1]==3)
        vinkel =[pi/3 + float(pi) for i=1:blocks]; pre="some_name3"
    elseif(vvinkel[1]==4)
        vinkel =[pi/9 + float(pi) for i=1:blocks]; pre="some_name4"
    elseif(vvinkel[1]==5)
        vinkel =[pi/4 + float(pi) for i=1:blocks]; pre="some_name5"
    elseif(vvinkel[1]==6)
        vinkel =[pi/6 + float(pi) for i=1:blocks]; pre="some_name6"
    elseif(vvinkel[1]==7)
        vinkel =[pi/8 + float(pi) for i=1:blocks]; pre="some_name7"
    elseif(vvinkel[1]==8)
        vinkel =[pi/10 + float(pi) for i=1:blocks]; pre="some_name8"
    elseif(vvinkel[1]==9)
        vinkel =[pi/12 + float(pi) for i=1:blocks]; pre="some_name9"
    elseif(vvinkel[1]==10)
        vinkel =[pi/2 for i=1:blocks]; pre="some_name10"    
    else
#        vinkel =[float(pi) for i=1:blocks]; pre="PLEASE"
        vinkel =[0.0 for i=1:blocks]; pre="PLEASE"
    end
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
    ## boundary discr section
    n_r,hx_r,PAR,DPR,NOR,MP,DX1,DX2,DX3,N_list,DPAR,DDPR,DNOR,SIN,UNOO,TT_jordan,TT_arc,TT1,TT2=set_bdry_points(nn,specimen,lenny,vinkel,BC,pari,arc_l,normv,dx1,dx2,dx3)

    ########################################################################################################################
    if (order==2)
        print("Second order baby\n")
        λ , Δt, wave_ns,P,Pinv = setupBDFM(NN,T,w_s)
    elseif(order==3)
        print("Third order baby\n")
        λ, Δt, wave_ns,P,Pinv= setupRK3(N,T,NN,fact,w_s);
    elseif(order==5)
        print("Fifth order baby\n")
        λ, Δt, wave_ns,P,Pinv= setupRK5(N,T,NN,fact,w_s); 
    end

    if (order==2)
        DFT_G,DFT_GP = gg_setup(nn,n_r,hx,NN,NN,λ,Δt,Gbdr,[0],PAR,NOR,DPR,MP,blocks,specimen,co,layer_rep,BC) 
    elseif(order==3)
        DFT_G,DFT_GP = gg_setup(nn,n_r,hx,NN,N,λ,Δt,Gbdr,c3,PAR,NOR,DPR,MP,blocks,specimen,co,layer_rep,BC) 
    elseif(order==5)
        DFT_G,DFT_GP = gg_setup(nn,n_r,hx,NN,N,λ,Δt,Gbdr,c5,PAR,NOR,DPR,MP,blocks,specimen,co,layer_rep,BC)  
    end

    print(norm(DFT_G)," ----  dftg\n")
    print(norm(DFT_GP)," --  dftgp\n")

    phi_hat = complex( zeros( size(DFT_GP) ) );
    choice = 0; #reference to the quadrature rule. Will not change from alpert. 


    phi_hat= den_sol_MIX(choice ,nn,n_r, N,λ,hx,Δt,wave_ns,P,Pinv,DFT_G,DFT_GP,Ψ,K_rep,Gk,N__jordan,N__openArc,Sη,Kη,GkGη,Kk,K0,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,DX1,DX2,DX3,blocks,co,order,UNOO,TT1,TT2,specimen,N_list,layer_rep,BC) 

    print(norm(phi_hat), "----------- one of the density\n")

    if(MOVIE==0)
        
        
        obs = zeros( NN+1,length(z) );
        inc_f=zeros(NN+1,length(z));
       
        #calculates in inc field at the obs point for saving. 
        for i in(1:length(z))
            inc_f[:,i] = [  g( z[i] , Δt*j ) for j=0: NN];
        end

        if(order==2)
            print("do as I say damnit!!! \n")
            for i in(1:length(z))
                obs[:,i]= MIX_time_freq_sol_BDFM2(hx,NN,nn , λ,z[i] , wave_ns ,phi_hat ,PAR,NOR,DPR , KK,blocks,specimen,layer_rep); 
            end
        elseif(order==3||order==5)
            for i in(1:length(z))
                obs[:,i] = MIX_time_freq_sol(hx,N,nn , λ, Δt ,z[i] ,phi_hat,wave_ns,P,Pinv,PAR,NOR,DPR, KK,blocks,w_s,specimen,layer_rep)[1:NN+1] ;
            end
        end
        
        print(norm(obs), "   the norm of the field      \n")
        
	#The two lines below do the same thing. Only difference is which format the data is stored as. 
        save_field_txt(T,NN,z,inc_f,obs)
        #save(prepre*LAYER[1]*"amp"*string(amp)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(order)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(order)*"_"*string(layer_rep)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".jld","data",obs)

    elseif(MOVIE==1)
        ##outputs the entire thing as a JLD file so it can be made into an MP4 movie
        y = range(y_lenny_left, length=Ngridlen, stop=y_lenny_right);
        x = range(x_lenny_left, length=Ngridlen, stop=x_lenny_right);
        X = repeat(reshape(x, 1, :), length(y), 1)
        Y = repeat(y, 1, length(x))
        obs = complex(zeros( NN+1, Ngridlen,Ngridlen));

        if(order==2)
            for ii in(1:Ngridlen)
                for jj in(1:Ngridlen)
                    local z=[X[ii,jj],Y[ii,jj]];
                    temp = [  g( z , Δt*j ) for j=0: NN];
                    obs[:,ii,jj]= MIX_time_freq_sol_BDFM2(hx,NN,nn , λ,z , wave_ns ,phi_hat ,PAR,NOR,DPR , KK,blocks,specimen,layer_rep)+temp; 
                end
            end
        elseif(order==3||order==5)
            for ii in(1:Ngridlen)
                for jj in(1:Ngridlen)
                    local z=[X[ii,jj],Y[ii,jj]];
                    temp = [  g( z , Δt*j ) for j=0: NN];
                    obs[:,ii,jj] = MIX_time_freq_sol(hx,N,nn , λ, Δt ,z ,phi_hat,wave_ns,P,Pinv,PAR,NOR,DPR, KK,blocks,w_s,specimen,layer_rep)[1:NN+1] +temp ;  
                end
            end
        end

        

        ##Saves the output as jld file. Just because its a good backup. ###
        ###################################################################
        save(prepre*LAYER[1]*"MOVamp"*string(amp)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(ord)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(specimen)*string(order)*"_"*string(layer_rep)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".jld","data",obs);

        ## This block makes the png images ##############################  
        ################################################################
        levels = range(zmin,stop=zmax,length=35) 
        
        
        for i in(1:NN+1)
        #     imshow(abs.(field[i,:,:]));#colorbar();
        #     clim(0,2)
            if(order==2)
                if(all(in(BC),1))
                    #title("BDF2 Final time T="*string(T)*" \n t="*string((i-1)*Δt));
                    title("BDF2 Dirichlet BC");
                elseif(all(in(BC),2))
                    title("BDF2 Neumann BC");
                else
                    title("BDF2 Mix BC");
                end
            elseif(order==3)
                if(all(in(BC),1))
                    title("IRK3 Dirichlet BC");
                elseif(all(in(BC),2))
                    title("IRK3 Neumann BC");
                else
                    title("IRK3 Mix BC");
                end
            elseif(order==5)
                if(all(in(BC),1))
                    title("IRK3 Dirichlet BC");
                elseif(all(in(BC),2))
                    title("IRK3 Neumann BC");
                else
                    title("IRK3 Mix BC");
                end
            end
            contourf(X,Y,abs.(obs[i,:,:]),levels=levels,cmap="jet")
            # colorbar(); ##uncomment this line if a colorbar is desired. 

            for j in(1:blocks)
                xx = [PAR[j][i][1] for i=1:nn]; yy = [PAR[j][i][2] for i=1:nn];
                plot(xx,yy,"w")
                if(specimen[j]>=4)
                    fill(xx,yy,"gray")
                end
            end


            axis("scaled")
            savefig(""*string(i)*".png", dpi=600)
            close();
            print(i," ")

#            # If dsesired uncomment the below lines to get rid of all whitespaces and make a .png transparent. My suggestions would be to remove the title also. 
#            axis("scaled")
#            axis("off")
#            savefig(string(i)*".png",transparent="True",dpi=600,bbox_inches="tight")



        end
	## this is a terminal command that converts the png to mp4. With this particular codec the movie should be playable on macs. 
    	run(`ffmpeg -framerate 24 -i %00d.png -s:v 1280x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p movie.mp4`)
    end
    
end
####----------------Driver END-----------------

