using PyPlot
using JLD
using LinearAlgebra


PASS = ARGS;


BBC = let expr = Meta.parse(PASS[1])
    @assert expr.head == :vect
    Int64.(expr.args)
end




include("engine_start.jl")

obs=load("MOVamp1.0_start_6_ws_1.0_pp4_ord4_over1_fact5_16mult_[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]2_2_nn256_NN256_ANGLE.jld")["data"];

levels = range(0,stop=2.0,length=35) 

        Ngridlen = 2^8; lenny=4;
        y = range(-lenny,length=Ngridlen, stop=lenny);
        x = range(-lenny,length=Ngridlen, stop=lenny);
        X = repeat(reshape(x, 1, :), length(y), 1)
        Y = repeat(y, 1, length(x))



NN = length(obs[:,1,1])-1; 
order=2;
#BC=[1,1];


for i in(1:NN+1)
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
    #contourf(X,Y,abs.(obs[i][:,:]),levels=levels,cmap="jet")
    # colorbar();

#    for j in(1:blocks)
#        xx = [PAR[j][i][1] for i=1:nn]; yy = [PAR[j][i][2] for i=1:nn];
#        plot(xx,yy,"w")
#        if(specimen[j]>=4)
#            fill(xx,yy,"gray")
#        end
#    end
#

    axis("scaled")
    savefig(""*string(i)*".png")
    close();
end
## this is a terminal command that converts the png to mp4. With this particular codec the movie should be playable on macs. 
run(`ffmpeg -framerate 24 -i %00d.png -s:v 1280x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -y movie.mp4`)


