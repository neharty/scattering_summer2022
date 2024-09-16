#Will have to test this tomorrow. 
using SpecialFunctions
using LaTeXStrings
using PyCall
using LinearAlgebra
using FFTW
using JLD
using Formatting
using DelimitedFiles
using Printf
using PyPlot

Base.show(io::IO, f::Float64) = @printf(io, "%.18f", f)


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
include("save_txt.jl")
include("engine_start.jl")

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


#Ω=97/6;
#Ω=4-73/6 *cos(θ)
#retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
#conny =5.5;
#plads = 0.10
#amp=2.0; 
#g = (xvec,t)->-(amp*exp(-modulate*(conny*retning(xvec,t))^2)-exp(-(modulate-1)*(conny*retning(xvec,t+plads))^2)-exp(-(modulate-1)*(conny*retning(xvec,t-plads))^2) );


amp=1.0;
retning =(xvec,t)-> w_s*t- xvec'*inny -Ω;
Ω=5.0; 
aa = 2*pi/0.3;
g1 =(xvec,t)-> -sin(aa*retning(xvec,t))*(0.0<=retning(xvec,t)<0.15);
g2 =(xvec,t)-> -2*sin(aa*retning(xvec,t))*(0.15<=retning(xvec,t)<0.3);
g3 =(xvec,t)-> -sin(aa*retning(xvec,t))*(0.3<=retning(xvec,t)<0.45);

g = (xvec,t)-> g1(xvec,t) + g2(xvec,t) + g3(xvec,t);

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
######## Aux setup for the geometry ##########
###center = [[1.5*i,1.5*j] for i=-2:2,j=-1:1];
##blocks =length(center);
###typ=1;
###test= typ*ones(blocks)
##pari,arc_l,normv,co = set_geo(form);
##blocks =length(center);
###vinkel =[i*pi/2.235 for i=1:blocks]
##lenny[i]=0.5;
#
#
#
#center = [[(5/3)*i,0] for i=-7:7];
#blocks =length(center);
#typ=1;
#specimen= typ*ones(blocks)
#pari,arc_l,normv,co = set_geo(specimen);
#vinkel =[0.0 for i=1:blocks]
#lenny=0.5*ones(blocks);
#


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
        save_field_txt(T,NN,z,inc_f,obs)
#        save("amp"*string(amp)*"_mod"*string(modulate)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(ord)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(form)*string(order)*"_"*string(which)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".jld","data",obs);

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
                    obs[:,ii,jj] = time_freq_sol(N,nn , λ, Δt ,z ,phi_hat,wave_ns,P,Pinv,PAR,NOR,DPR, KK,blocks,w_s )[1:NN+1] +temp;
                end
            end
        end

        ##Saves the output as jld file. Just because its a good backup. ###
        ###################################################################
        save("MOVamp"*string(amp)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(ord)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(specimen)*string(order)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".jld","data",obs);

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

