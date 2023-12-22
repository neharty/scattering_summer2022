function Set_boundary_mix(which::Array{Int64},BC::Array{Int64},β::Float64)
    fff=Function[]; Arc=Function[]; norm_v=Function[]; CO=Array[]; dx1 = Function[]; dx2= Function[]; dx3= Function[];
    for i in(1:length(which))
        if(BC[i]==1)
            pari,arc_l,normv,co, dx1_t,dx2_t, dx3_t = DIR_set_geo([which[i]],β);
        elseif(BC[i]==2)
            pari,arc_l,normv,co, dx1_t,dx2_t, dx3_t = NEU_set_geo([which[i]],β);
        end
        fff = push!(fff,pari);
        Arc = push!(Arc,arc_l);
        norm_v = push!(norm_v,normv);
        CO = push!(CO,co);
        dx1 = push!(dx1,dx1_t);
        dx2 = push!(dx2,dx2_t);
        dx3 = push!(dx3,dx3_t);
    end

    return fff,Arc,norm_v,CO,dx1,dx2,dx3; 
end

function set_bdry_points(n::Int64,specimen::Array{Int64},lenny::Array{Float64},vinkel::Array{Float64},
                         BC::Array{Int64},pari::Array{Function},arc_l::Array{Function},normv::Array{Function},
                         dx1::Array{Function},sx2::Array{Function},dx3::Array{Function})
    
    sin_shift=(x,co,LL,angle,shift)->sin(x);

    n_r=Int(n/2)-1; hx_r = 2*pi/(n);

    PAR = [];
    DPR = [];
    NOR = [];
    MP=[];

    DPAR=[]; DNOR=[]; DDPR=[];
    DX1 =[]; DX2 =[]; DX3 =[];
    SIN=[];

    N_list = []

    TT_jordan = [0.5*(2*j-1)*hx for j=1:n];

    TT_arc = [(j)*hx_r for j=0:n];
    TT1= [cos(j*hx_r) for j=0:n]
    TT2= [sin(j*hx_r) for j=0:n];

    UNOO=ones(n);
    ## Discretization section
    for i in(1:blocks) 
        if(specimen[i]>=4)
            
            PAR = push!(PAR,[pari[i](TT_jordan[j],co[i],lenny[i], vinkel[i],center[i]) for j=1:n]);
            DPR = push!(DPR,[arc_l[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:n]);
            NOR = push!(NOR,[normv[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:n]);

            DX1 = push!(DX1,[dx1[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:n])
            DX2 = push!(DX2,[dx2[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:n])
            DX3 = push!(DX3,[dx3[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:n])
            N_list = push!(N_list,n);

            if(length(co[i])!=0)
                MP=push!(MP,[moid_prime(TT_jordan[j],co[i]) for j=1:n]);
            elseif(length(co[i])==0)
                MP=push!(MP,ones(n))
            end
                Dtemp1=[]; Dtemp2=[];Dtemp3=[];Dtemp4=[]
                for ii in(1:n)
                    Dtemp1 = vcat(Dtemp1, [experiment(TT_jordan[ii] ,hx , n , pari[i],co[i],lenny[i],vinkel[i],center[i])] ) ;
                    Dtemp2 = vcat(Dtemp2, [Oexperiment(TT_jordan[ii] ,hx , n , arc_l[i],co[i],lenny[i],vinkel[i])] ) ;
                    Dtemp3 = vcat(Dtemp3, [Oexperiment(TT_jordan[ii] ,hx , n , normv[i],co[i],lenny[i],vinkel[i])] ) ;
                end
            
            DPAR=push!(DPAR,Dtemp1);
            DDPR=push!(DDPR,Dtemp2);
            DNOR=push!(DNOR,Dtemp3);
        elseif(specimen[i]<4)
            
            if(BC[i]==1)
                PAR = push!(PAR,[pari[i](TT_jordan[j],co[i],lenny[i], vinkel[i],center[i]) for j=1:n]);
                DPR = push!(DPR,[arc_l[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:n]);
                NOR = push!(NOR,[normv[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:n]);

                DX1 = push!(DX1,[dx1[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:n])
                DX2 = push!(DX2,[dx2[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:n])
                DX3 = push!(DX3,[dx3[i](TT_jordan[j],co[i],lenny[i],vinkel[i]) for j=1:n])
                N_list = push!(N_list,n);

                if(length(co[i])!=0)
                    MP=push!(MP,[moid_prime(TT_jordan[j],co[i]) for j=1:n]);
                elseif(length(co[i])==0)
                    MP=push!(MP,ones(n))
                end
                    Dtemp1=[]; Dtemp2=[];Dtemp3=[];Dtemp4=[]
                    for ii in(1:n)
                        Dtemp1 = vcat(Dtemp1, [experiment(TT_jordan[ii] ,hx , n , pari[i],co[i],lenny[i],vinkel[i],center[i])] ) ;
                        Dtemp2 = vcat(Dtemp2, [Oexperiment(TT_jordan[ii] ,hx , n , arc_l[i],co[i],lenny[i],vinkel[i])] ) ;
                        Dtemp3 = vcat(Dtemp3, [Oexperiment(TT_jordan[ii] ,hx , n , normv[i],co[i],lenny[i],vinkel[i])] ) ;
                    end

                DPAR=push!(DPAR,Dtemp1);
                DDPR=push!(DDPR,Dtemp2);
                DNOR=push!(DNOR,Dtemp3);
                
                
            elseif(BC[i]==2)
                PAR = push!(PAR,[pari[i](TT_arc[j+1],co[i],lenny[i], vinkel[i],center[i]) for j=1:n_r]);
                DPR = push!(DPR,[arc_l[i](TT_arc[j+1],co[i],lenny[i],vinkel[i]) for j=1:n_r]);
                NOR = push!(NOR,[normv[i](TT_arc[j+1],co[i],lenny[i],vinkel[i]) for j=1:n_r]);

                DX1 = push!(DX1,[dx1[i](TT_arc[j+1],co[i],lenny[i],vinkel[i]) for j=1:n_r])
                DX2 = push!(DX2,[dx2[i](TT_arc[j+1],co[i],lenny[i],vinkel[i]) for j=1:n_r])
                DX3 = push!(DX3,[dx3[i](TT_arc[j+1],co[i],lenny[i],vinkel[i]) for j=1:n_r])
                N_list = push!(N_list,n_r);

                MP=push!(MP,ones(n))

                Dtemp1=[]; Dtemp2=[];Dtemp3=[];Dtemp4=[]
                for ii in(1:n_r)
                    Dtemp1 = vcat(Dtemp1, [experiment(TT_arc[ii+1] ,hx_r , n_r , pari[i],co[i],lenny[i],vinkel[i],center[i])] ) ;
                    Dtemp2 = vcat(Dtemp2, [Oexperiment(TT_arc[ii+1] ,hx_r , n_r , arc_l[i],co[i],lenny[i],vinkel[i])] ) ;
                    Dtemp3 = vcat(Dtemp3, [Oexperiment(TT_arc[ii+1] ,hx_r , n_r , normv[i],co[i],lenny[i],vinkel[i])] ) ;
                    Dtemp4 = vcat(Dtemp4, [experiment(TT_arc[ii+1] ,hx_r , n_r , sin_shift,co[i],lenny[i],vinkel[i],center[i])] ) ;
                end

                DPAR=push!(DPAR,Dtemp1);
                DDPR=push!(DDPR,Dtemp2);
                DNOR=push!(DNOR,Dtemp3);
                SIN=push!(SIN,Dtemp4);
            end
            
        end
    end
    return n_r,hx_r,PAR,DPR,NOR,MP,DX1,DX2,DX3,N_list,DPAR,DDPR,DNOR,SIN,UNOO,TT_jordan,TT_arc,TT1,TT2;
end



function save_field_txt(T::Float64,NN::Int64,z::Array,inc_f::Array{Float64},obs::Array{Float64})
    td = T/NN;
    ddt=[td*i for i=0:NN];
    #name = "amp"*string(amp)*"_mod"*string(modulate)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(ord)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(form)*string(order)*"_"*string(which)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir;
#   name =prepre*LAYER[1]*"amp"*string(amp)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(order)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(order)*"_"*string(layer_rep)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir
#    

    name =LAYER[1]*"_"*string(layer_rep)
    
    ##Writes the scattered field in ascii
    open(name*".txt", "w") do f
        for i in 1:NN+1
            n1 = format(ddt[i]);
            holder="";
            for j in(1:length(obs[1,:]))
                holder = holder*string(obs[i,j])*"  "; 
            end
            write(f, "$n1     $holder     \n")
        end
    end
end

#function save_field_txt(T::Float64,NN::Int64,z::Array,inc_f::Array{Float64},obs::Array{Float64})
#    
#    td = T/NN;
#    ddt=[td*i for i=0:NN];
#    name =prepre*LAYER[1]*"amp"*string(amp)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(order)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(order)*"_"*string(layer_rep)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".txt";
#
#    open(name, "w") do io
#           writedlm(io, [ddt obs])
#    end;
#
#end
#


