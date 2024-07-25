#Needed for the different CQ stepping methods. 
c3 = [1/3 1];
c5 = [(2/5 - sqrt(6)/10) (2/5 +sqrt(6)/10 ) (1)]; 



###Need a setup BDF2
function to_e(x)
    if(x%2!=0)
        x+=1;
    end
    return x
end

##I think this one is fine enough
function setupBDFM(N::Integer,T::Float64,w_s::Float64)  
    delt = T/(N); 
    lamb = max( delt^(3/(N)) ,eps(1.0)^(1/(2*(N))));     
    fun = (x)-> (1/2)*(x^2-4*x+3);
    zl = [lamb*exp(-2*pi*im *l/(N+1))  for l =0:N]; 
    kvec = [im*fun(zl[j])/(w_s*delt) for j=1:N+1];
    P=[]; Pinv=[]; #these are dummy variables. Don't need them
    return lamb, delt, kvec,P,Pinv;
end

### I think these functions below are correct, but will need to test

function setupRK3(N::Integer,T::Float64,NN::Integer,factor::Float64,w_s::Float64)
    delt = T/(NN);
    lamb = max( delt^(4.5/(NN)) ,eps(1.0)^(1/(2*(NN))));
    zl = [lamb*exp(-2*pi*im *l/(N+1))  for l =0:N];
    A = [5/12 -1/12 ; 3/4  1/4];
    b = [3/4 1/4];
    uno = [1 1];
    fun = z->inv(A+ (z/(1-z))*uno'*b);
    K = [fun(zl[i]) for i=1:N+1];

    wave_ns=complex( zeros(N+1,length(K[1][:,1])) );
    P = complex( zeros(N+1,length(K[1][:,1]),length(K[1][:,2]) ) );
    Pinv = complex( zeros(N+1,length(K[1][:,1]),length(K[1][:,2]) ) ); 
    
    for j in(1:N+1)
        wave_ns[j,:] = (im*eigvals(K[j]))/(w_s*delt) ;
        P[j,:,:] = eigvecs(K[j]);
        Pinv[j,:,:] = inv(P[j,:,:])
    end

    return lamb, delt, wave_ns,P,Pinv;
end

function setupRK5(N::Integer,T::Float64, NN::Integer, factor::Float64,w_s::Float64)
    delt = T/(NN);
    lamb = max( delt^(factor/(NN)) ,eps(1.0)^(1/(2*(NN)))); ##  the factor was 5.   
    zl = [lamb*exp(-2*pi*im *l/(N+1))  for l =0:N];
    A = [(11/45 - 7*sqrt(6)/360) (37/225 - 169*sqrt(6)/1800) (-2/225 + sqrt(6)/75); (37/225 + 169*sqrt(6)/1800)  (11/45 + 7*sqrt(6)/360) (-2/225 -sqrt(6)/75);
        (4/9 - sqrt(6)/36) (4/9 + sqrt(6)/36) (1/9)] ;
    b= [(4/9 -sqrt(6)/36) (4/9+sqrt(6)/36) (1/9)];
    uno = [1 1 1];
    fun = z->inv(A+ (z/(1-z))*uno'*b);
    K = [fun(zl[i]) for i=1:N+1];

    wave_ns=complex( zeros(N+1,length(K[1][:,1])) );
    P = complex( zeros(N+1,length(K[1][:,1]),length(K[1][:,2]))) #, length(K[1][:,3]) ) );
    Pinv = complex( zeros(N+1,length(K[1][:,1]),length(K[1][:,2]))) #, length(K[1][:,3]) ) ); 
    
    for j in(1:N+1)
        wave_ns[j,:] = (im*eigvals(K[j]))/(w_s*delt) ;
        P[j,:,:] = eigvecs(K[j]);
        Pinv[j,:,:] = inv(P[j,:,:])
    end

    return lamb, delt, wave_ns,P,Pinv;

end

# baring any mistakes, I think this is correct now. 
function gg_setup(nn::Integer,n_r::Integer,hx::Float64,NN::Integer,N::Integer,λ::Float64,Δt::Float64, g::Function,gp::Function,c::Array, par::Array,
                  nor::Array,dpr::Array,MP::Array,blocks::Int64,specimen::Array,co::Array,which::Int64)
    if(which==1)
        G = [] ;
        DFT_G = [];
    
        GP = zeros(length(c),blocks*nn,N+1); 
        DFT_GP = complex(zeros(length(c),blocks*nn,Int((N)/2 +1) ) );
	for k in(1:length(c))
            for pp in(0:blocks-1)
                GP[k,(pp%blocks)*nn+1:(pp%blocks+1)*nn,1:NN+1] = -2*[ λ^(j)*dpr[pp+1][i]*gp( par[pp+1][i] , Δt*j + c[k]*Δt,nor[pp+1][i]) for i=1:nn, j=0: NN]; ##removed MP 
#                Glamb[k,(pp%blocks)*nn+1:(pp%blocks+1)*nn,1:NN+1] = -1*[ (abs(g( par[pp+1][i] , Δt*j + c[k]*Δt))>=1e-10  )*λ^(j)*MP[pp+1][i] * g( par[pp+1][i] , Δt*j + c[k]*Δt) +
#                                                (abs(g( par[pp+1][i] , Δt*j + c[k]*Δt))<1e-10  )*0 for i=1:nn, j=0: NN];  
            end

            for i in(1:blocks*nn)
                    DFT_GP[k,i,:] = rfft(GP[k,i,:])  
            end
	end       
    elseif(which==2 ||which==3)
        G = zeros(length(c),blocks*nn,N+1);
        GP = zeros(length(c),blocks*nn,N+1);
        DFT_G = complex(zeros(length(c),blocks*nn,Int((N)/2 +1) ) );
        DFT_GP = complex(zeros(length(c),blocks*nn,Int((N)/2 +1) ) );
        
        for k in(1:length(c))
            for pp in(0:blocks-1)
                left =pp%blocks*nn+1;
                right=(pp%blocks+1)*nn;

                if(specimen[pp+1]<4)
                    LLeft = (pp%blocks)*nn+1
                    RRight = (pp%blocks+1)*(nn)-n_r-2;
                    GP[k,LLeft:RRight,1:NN+1] = -[ λ^(j)*dpr[pp+1][i]*gp( par[pp+1][i] , Δt*j + c[k]*Δt,nor[pp+1][i]) for i=1:n_r, j=0: NN];

                elseif(specimen[pp+1]>=4)

                    if(length(co[pp+1])==0)
                        GP[k,left:right,1:NN+1] = -2*[ λ^(j)*dpr[pp+1][i]*gp( par[pp+1][i] , Δt*j + c[k]*Δt,nor[pp+1][i]) for i=1:nn, j=0: NN];

                    elseif(length(co[pp+1])!=0)
                        G[k,left:right,1:NN+1] = [ λ^(j)*g( par[pp+1][i] , Δt*j + c[k]*Δt) for i=1:nn, j=0: NN]; #2 does not appear here, nor does the dpr 
                        GP[k,left:right,1:NN+1] = [ λ^(j)*gp( par[pp+1][i] , Δt*j + c[k]*Δt,nor[pp+1][i]) for i=1:nn, j=0: NN]; #2 does not appear here, nor does the dpr 
                    end
                end
            end
            for i in(1:blocks*nn)
                    DFT_G[k,i,:] = rfft(G[k,i,:])  
                    DFT_GP[k,i,:] = rfft(GP[k,i,:])  
            end
        end
    end
    return DFT_G,DFT_GP;
end


function den_sol_S(choice::Int, nn::Integer, N::Integer, λ::Float64, hx::Float64 ,Δt::Float64, waveNS::Array,P::Array,Pinv::Array, g_hat::Array,k::Function,which::Int64, 
                 PAR::Array,NOR::Array,DPR::Array,MP::Array,DPAR::Array,DNOR::Array,DDPR::Array,blocks::Int64,co::Array,order::Int64)

        ϕ_hat = complex( zeros(size( g_hat ) ) );
         alt_N = length(g_hat[1,1,:])

        ## BDFM block
        if(order==2)
            for j in(1:alt_N) #
                if maximum(abs.(g_hat[1,:,j]) ) >=6.5e-10    
                    print(waveNS[j]," ")
                    ϕ_hat[1,:,j] = scatSYSS(choice,hx,nn,g_hat[1,:,j],k,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,waveNS[j]);     
                    print("2nd order Sing we are at iteration ",j, "\n");
                end
            end
        ## IRK schemes. 
        elseif(order==3)
            for j in(1:alt_N) #
                if maximum(abs.(g_hat[1,:,j]) ) >=6.5e-10 #&&  norm(g_hat[2,:,j])>=1e-10   
                    resvec=Pinv[j,:,:]*g_hat[:,:,j];
                    for i in(1:length(waveNS[j,:]))
                        ϕ_hat[i,:,j] =  scatSYSS(choice,hx,nn, resvec[i,:],k,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,waveNS[j,i]);    
                    end
                    print(string(order)*"rd order Sing we are at iteration ",j, "\n");
                end
            end
        elseif(order==5)
            for j in(1:alt_N) #
                if maximum(abs.(g_hat[1,:,j])) >=6.5e-10  #change this error tollerence. e-6.   
                    resvec=Pinv[j,:,:]*g_hat[:,:,j];
                    for i in(1:length(waveNS[j,:]))
                        ϕ_hat[i,:,j] =  scatSYSS(choice,hx,nn, resvec[i,:],k,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,waveNS[j,i]);     
                    end
                        print(string(order)*"th Sing we are at iteration ",j, "\n");
                end
            end
        end

    print("the size of ϕ ofr BDFM is ",size(ϕ_hat),"\n")
    if(order==2)
        ϕ_hat = reshape(ϕ_hat, (blocks*nn,alt_N));
    end
	return ϕ_hat;
end

function den_sol_DC(choice::Int, nn::Integer,n_r::Integer, N::Integer, λ::Float64, hx::Float64 ,Δt::Float64, waveNS::Array,P::Array,Pinv::Array,g_hat::Array, gp_hat::Array,k::Function,Gk::Function,N__jordan::Function,
                   N__openArc::Function,Sη::Function,Kη,GkGη::Function,Kk::Function,Which::Int64, PAR::Array,NOR::Array,DPR::Array,MP::Array,DPAR::Array,DNOR::Array,DDPR::Array,
                   DX1::Array,DX2::Array,DX3::Array,blocks::Int64,co::Array,order::Int64,UNO::Array,TT1::Array,TT2::Array,specimen::Array,N_list::Array)

       #Sη_mat = ALPERT_jordan(hx,n,Sη,PAR[pp+1],NOR[pp+1],DPR[pp+1],MP[pp+1],DPAR[pp+1],DNOR[pp+1],DDPR[pp+1],UNO); 
       #Kη_mat  =  ALPERT_jordan(h,N,Kη,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],UNO); 

       print(size(UNO)," size of UNO\n")

        Sη_mat=[];
        Kη_mat=[];
        if(any(x->x!=0,co))
            for pp in(1:blocks)
                if(length(co[pp])!=0)
                    print(co[pp], "\n")
                    Sη_mat = push!(Sη_mat,ALPERT_jordan(hx,nn,Sη,PAR[pp],NOR[pp],DPR[pp],MP[pp],DPAR[pp],DNOR[pp],DDPR[pp],UNO,0+im*0)); 
                    Kη_mat = push!(Kη_mat,ALPERT_jordan(hx,nn,Kη,PAR[pp],NOR[pp],DPR[pp],MP[pp],DPAR[pp],DNOR[pp],DDPR[pp],UNO,0+im*0)); 
                elseif(length(co[pp])==0)
                    Sη_mat = push!(Sη_mat,[]); 
                    Kη_mat = push!(Kη_mat,[]); 
                end
            end
        end

        ϕ_hat = complex( zeros(size( gp_hat ) ) );
        alt_N = length(gp_hat[1,1,:])


       ## BDFM block
        if(order==2)
            for j in(1:alt_N)
                if maximum(abs.(gp_hat[1,:,j]) ) >=6.5e-10    
                    #ϕ_hat[1,:,j] = scatSYSD(choice,hx,hx_r,nn, n_r,gp_hat[1,:,j],k,Gk,N__jordan,N__openArc,Sη,Kη,GkGη,PAR,DX1,DX2,DX3,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,specimen,TT1,TT2,SIN,N_list,waveNS[j],UNO); 
                    ϕ_hat[1,:,j] = scatSYSD(choice,hx,hx_r,nn,n_r,g_hat[1,:,j],gp_hat[1,:,j],k,Gk,N__jordan,N__openArc,Sη_mat,Kη_mat,GkGη,Kk,PAR,DX1,DX2,DX3,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,specimen,TT1,TT2,SIN,N_list,waveNS[j],UNO); 

                    print("2nd order D&C we are at iteration ",j, "\n");
                end
            end
        ## IRK schemes. 
        elseif(order==3)
            for j in(1:alt_N) #
                if maximum(abs.(gp_hat[1,:,j]))  >=6.5e-10 #&&  norm(gp_hat[2,:,j])>=1e-10   
                    resvec1=Pinv[j,:,:]*g_hat[:,:,j];
                    resvec2=Pinv[j,:,:]*gp_hat[:,:,j];
                    for i in(1:length(waveNS[j,:]))
                        #ϕ_hat[i,:,j] = scatSYSD(choice,hx,hx_r,nn, n_r,resvec[i,:],k,Gk,N__jordan,N__openArc,Sη,Kη,GkGη,PAR,DX1,DX2,DX3,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,specimen,TT1,TT2,SIN,N_list,waveNS[j,i],UNO);
                        ϕ_hat[i,:,j] = scatSYSD(choice,hx,hx_r,nn, n_r,resvec1[i,:],resvec2[i,:],k,Gk,N__jordan,N__openArc,Sη_mat,Kη_mat,GkGη,Kk,PAR,DX1,DX2,DX3,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,specimen,TT1,TT2,SIN,N_list,waveNS[j,i],UNO);
                    end
                    print(string(order)*"rd  D&C we are at iteration ",j, "\n");
                end
            end
        elseif(order==5)
            for j in(1:alt_N) #
                if maximum(abs.(gp_hat[1,:,j])) >=6.5e-10 #&&  norm(gp_hat[2,:,j])>=1e-10   &&  norm(gp_hat[3,:,j])>=1e-10 
                    resvec1=Pinv[j,:,:]*g_hat[:,:,j];
                    resvec2=Pinv[j,:,:]*gp_hat[:,:,j];
                    for i in(1:length(waveNS[j,:]))
                        #ϕ_hat[i,:,j] = scatSYSD(choice,hx,hx_r,nn, n_r,resvec[i,:],k,Gk,N__jordan,N__openArc,Sη,Kη,GkGη,PAR,DX1,DX2,DX3,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,specimen,TT1,TT2,SIN,N_list,waveNS[j,i],UNO);
                        ϕ_hat[i,:,j] = scatSYSD(choice,hx,hx_r,nn, n_r,resvec1[i,:],resvec2[i,:],k,Gk,N__jordan,N__openArc,Sη_mat,Kη_mat,GkGη,Kk,PAR,DX1,DX2,DX3,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,specimen,TT1,TT2,SIN,N_list,waveNS[j,i],UNO);
                    end
                    print(string(order)*" D&C we are at iteration ",j, "\n");
                end
            end
        end

    print("the size of ϕ ofr BDFM is ",size(ϕ_hat),"\n")
    if(order==2)
        ϕ_hat = reshape(ϕ_hat, (blocks*nn,alt_N));
    end
	return ϕ_hat;

end



function time_freq_sol_BDFM2(h::Float64,N::Integer,M::Integer, λ::Float64,x::Array,s_l::Array,ϕ::Array,par::Array,nor::Array,dpr::Array,k::Function, blocks::Int64,specimen::Array)
    alt_N = length(ϕ[1,:]) #this extra 1 really shoulndt be here
    ϕ_hat = complex(zeros(alt_N));

    Lam = (λ.^(0:N)) ;
    for j in(1:alt_N)
        ϕ_hat[j] = ker_eval_jordan(h,M,ϕ[:,j],x,s_l[j],par,nor,dpr,k,blocks,specimen) #should be good. 
    end
    
#    for j=2:Int(floor(N/2))+1  ##Now what do I do here?
#        ϕ_hat[N+3-j] = conj(ϕ_hat[j]);
#    end;

    u = Lam.^(-1).*irfft(ϕ_hat,N+1);
    return u;
end

function time_freq_sol(h::Float64,N::Integer,M::Integer, λ::Float64,Δt::Float64,x::Array,ϕ::Array,WN::Array,P::Array,Pinv::Array,par::Array,nor::Array,dpr::Array,k::Function,blocks::Int64,w_s::Float64,specimen::Array)    
    alt_N=length(ϕ[1,1,:]) 
    temp = complex( zeros( size(ϕ[:,1,1]) ) );  
    ϕ_hat = complex(zeros(alt_N));
    Lam = (λ.^(0:N)) ; 

    for j in(1:alt_N)
        for ii in(1:length(P[1,1,:]))
            temp[ii] = P[j,length(P[1,1,:]),ii] * ker_eval_jordan(h,M,ϕ[ii,:,j],x, WN[j,ii] ,par,nor,dpr,k,blocks,specimen); 
            ϕ_hat[j] =ϕ_hat[j] + temp[ii];
        end
        ϕ_hat[j] =  λ*exp(-2*pi*im/(N+1))^(j-1) *ϕ_hat[j];
    end

#    for j=2:Int(floor(N/2))+1  ##Now what do I do here?
#        ϕ_hat[N+3-j] = conj(ϕ_hat[j]);
#    end;

    u = Lam.^(-1).*irfft(ϕ_hat,N+1);
    return u;
end


