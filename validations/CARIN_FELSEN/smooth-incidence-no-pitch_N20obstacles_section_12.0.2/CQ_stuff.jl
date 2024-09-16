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

#function setupRK3(N::Integer,T::Float64,NN::Integer)
#    delt = T/(NN);
#    lamb = max( delt^(4.5/(NN)) ,eps(1.0)^(1/(2*(NN))));  
#    zl = [lamb*exp(-2*pi*im *l/(N+1))  for l =0:N];
#    A = [5/12 -1/12 ; 3/4  1/4];
#    b = [3/4 1/4];
#    uno = [1 1];
#    fun = z->inv(A+ (z/(1-z))*uno'*b);
#    K = [fun(zl[i]) for i=1:N+1];
#    return lamb, delt, K;
#end

#function setupRK5(N::Integer,T::Float64, NN::Integer, factor::Float64)
#    delt = T/(NN);
#    lamb = max( delt^(factor/(NN)) ,eps(1.0)^(1/(2*(NN)))); ##  the factor was 5.   
#    zl = [lamb*exp(-2*pi*im *l/(N+1))  for l =0:N];
#    A = [(11/45 - 7*sqrt(6)/360) (37/225 - 169*sqrt(6)/1800) (-2/225 + sqrt(6)/75); (37/225 + 169*sqrt(6)/1800)  (11/45 + 7*sqrt(6)/360) (-2/225 -sqrt(6)/75);
#        (4/9 - sqrt(6)/36) (4/9 + sqrt(6)/36) (1/9)] ;
#    b= [(4/9 -sqrt(6)/36) (4/9+sqrt(6)/36) (1/9)];
#    uno = [1 1 1];
#    fun = z->inv(A+ (z/(1-z))*uno'*b);
#    K = [fun(zl[i]) for i=1:N+1];
#    return lamb, delt, K;
#end
#

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



function gg_setup(nn::Integer,hx::Float64,NN::Integer,N::Integer,λ::Float64,Δt::Float64, g::Function,c::Array, par::Array,
                  nor::Array,dpr::Array,MP::Array,blocks::Int64)
   # here is the needed modification.  
    Glamb = zeros(length(c),blocks*nn,N+1); 
    DFT_Glamb = complex(zeros(length(c),blocks*nn,Int((N)/2 +1) ) );

	for k in(1:length(c))
            for pp in(0:blocks-1)
                Glamb[k,(pp%blocks)*nn+1:(pp%blocks+1)*nn,1:NN+1] = -[ λ^(j)*MP[pp+1][i] * g( par[pp+1][i] , Δt*j + c[k]*Δt) for i=1:nn, j=0: NN]; 
#                Glamb[k,(pp%blocks)*nn+1:(pp%blocks+1)*nn,1:NN+1] = -1*[ (abs(g( par[pp+1][i] , Δt*j + c[k]*Δt))>=1e-10  )*λ^(j)*MP[pp+1][i] * g( par[pp+1][i] , Δt*j + c[k]*Δt) +
#                                                (abs(g( par[pp+1][i] , Δt*j + c[k]*Δt))<1e-10  )*0 for i=1:nn, j=0: NN];  
            end
            #This should work. 
            for i in(1:blocks*nn)
                    DFT_Glamb[k,i,:] = rfft(Glamb[k,i,:])  
            end
	end                                                                                                      
    return DFT_Glamb;
end


##Function handles need to be changed. 
function den_sol(choice::Int, nn::Integer, N::Integer, λ::Float64, hx::Float64 ,Δt::Float64, waveNS::Array,P::Array,Pinv::Array, g_hat::Array,k::Function,K0::Function,KER::Function,which::Int64, 
                 PAR::Array,NOR::Array,DPR::Array,MP::Array,DPAR::Array,DNOR::Array,DDPR::Array,blocks::Int64,co::Array,order::Int64)

        ϕ_hat = complex( zeros(size( g_hat ) ) );
         alt_N = length(g_hat[1,1,:])

        ## BDFM block
        if(order==2)
            if(which==1)
                    for j in(1:alt_N) #
                            if maximum(abs.(g_hat[1,:,j]) ) >=1.0e-9    
                                    ϕ_hat[1,:,j] = scatSYSS(choice,hx,nn,g_hat[1,:,j],k,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,waveNS[j]);  #scatSYSS(choice,hx,nn,g_hat[1,:,j], k, par,nor,dpr,blocks)    
                                    print("2nd order Sing we are at iteration ",j, "\n");
                            end
                    end
            elseif(which==2 || which==3 || which==4)
                    for j in(1:alt_N)
                            if maximum(abs.(g_hat[1,:,j]) ) >=1.0e-9    
                                ϕ_hat[1,:,j] = scatSYSD(choice,hx,nn,g_hat[1,:,j],k,K0,KER,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,coi,waveNS[j]); # there are cases when this needs to be a C.  
                                print("2nd order D&C we are at iteration ",j, "\n");
                            end
                    end
            end
        ## IRK schemes. 
        elseif(order==3)
            if(which==1)
                    for j in(1:alt_N) #
                        if maximum(abs.(g_hat[1,:,j]) ) >=1.0e-9 #&&  norm(g_hat[2,:,j])>=1e-10   
                                    resvec=Pinv[j,:,:]*g_hat[:,:,j];
                                    for i in(1:length(waveNS[j,:]))
                                            ϕ_hat[i,:,j] =  scatSYSS(choice,hx,nn, resvec[i,:],k,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,waveNS[j,i]);    
                                    end
                                    print(string(order)*"rd order Sing we are at iteration ",j, "\n");
                            end
                    end
            elseif(which==2 || which==3 || which==4)
                    for j in(1:alt_N) #
                            if maximum(abs.(g_hat[1,:,j]))  >=1.0e-9 #&&  norm(g_hat[2,:,j])>=1e-10   
                                    resvec=Pinv[j,:,:]*g_hat[:,:,j];
                                    for i in(1:length(waveNS[j,:]))
                                            ϕ_hat[i,:,j] =  scatSYSD(choice,hx,nn,resvec[i,:],k,K0,KER,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,waveNS[j,i]); # there are cases when D needs to be a C.  
                                    end
                                    print(string(order)*"rd  D&C we are at iteration ",j, "\n");
                            end
                    end
            end
        elseif(order==5)
            if(which==1)
                    for j in(1:alt_N) #
                        if maximum(abs.(g_hat[1,:,j])) >=1.0e-9  #change this error tollerence. e-6.   
                                    resvec=Pinv[j,:,:]*g_hat[:,:,j];
                                    for i in(1:length(waveNS[j,:]))
                                            ϕ_hat[i,:,j] =  scatSYSS(choice,hx,nn, resvec[i,:],k,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,waveNS[j,i]);     
                                    end
                                    print(string(order)*"th Sing we are at iteration ",j, "\n");
                            end
                    end
            elseif(which==2 || which==3 || which==4)
                    for j in(1:alt_N) #
                            if maximum(abs.(g_hat[1,:,j])) >=1.0e-9 #&&  norm(g_hat[2,:,j])>=1e-10   &&  norm(g_hat[3,:,j])>=1e-10 
                                    resvec=Pinv[j,:,:]*g_hat[:,:,j];
                                    for i in(1:length(waveNS[j,:]))
                                            ϕ_hat[i,:,j] =  scatSYSD(choice,hx,nn,resvec[i,:],k,K0,KER,PAR,NOR,DPR,MP,DPAR,DNOR,DDPR,blocks,co,waveNS[j,i]); # This D needs to be a C 
                                    end
                                    print(string(order)*" D&C we are at iteration ",j, "\n");
                            end
                    end
            end
        end

  

    print("the size of ϕ ofr BDFM is ",size(ϕ_hat),"\n")
    if(order==2)
        ϕ_hat = reshape(ϕ_hat, (blocks*nn,alt_N));
    end
	return ϕ_hat;
end

function ker_eval(ON::Int64,phi::Array,z::Array,sl::Complex, par::Array,nor::Array,dpr::Array,k::Function, blocks::Int64)  
    h = 2*pi/ON;
    hold = Float64(complex(0));
    for p in(1:ON)
        for jj in(0:blocks-1)
            xtp = par[jj+1][p];
            ntp = nor[jj+1][p]; 
            hold = hold + k(z-xtp,ntp,sl)*phi[p+jj*ON]*dpr[jj+1][p]; 
        end
    end
    hold = h*hold; 
    return hold;
end

function time_freq_sol_BDFM2(N::Integer,M::Integer, λ::Float64,x::Array,s_l::Array,ϕ::Array,par::Array,nor::Array,dpr::Array,k::Function, blocks::Int64)
    alt_N = length(ϕ[1,:])
    ϕ_hat = complex(zeros(alt_N));

    Lam = (λ.^(0:N)) ;
    for j in(1:alt_N)
        ϕ_hat[j] = ker_eval(M,ϕ[:,j],x,s_l[j],par,nor,dpr,k,blocks) #should be good. 
    end
    
#    for j=2:Int(floor(N/2))+1  ##Now what do I do here?
#        ϕ_hat[N+3-j] = conj(ϕ_hat[j]);
#    end;

    u = Lam.^(-1).*irfft(ϕ_hat,N+1);
    return u;
end

function time_freq_sol(N::Integer,M::Integer, λ::Float64,Δt::Float64,x::Array,ϕ::Array,WN::Array,P::Array,Pinv::Array,par::Array,nor::Array,dpr::Array,k::Function,blocks::Int64,w_s::Float64)    
    alt_N=length(ϕ[1,1,:]) 
    temp = complex( zeros( size(ϕ[:,1,1]) ) );  
    ϕ_hat = complex(zeros(alt_N));
    Lam = (λ.^(0:N)) ; 

    for j in(1:alt_N)
        for ii in(1:length(P[1,1,:]))
            temp[ii] = P[j,length(P[1,1,:]),ii] * ker_eval(M,ϕ[ii,:,j],x, WN[j,ii] ,par,nor,dpr,k,blocks); 
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
































#function time_freq_sol_BDFM2(N::Integer,M::Integer, λ::Float64,x::Array,s_l::Array,ϕ::Array,par::Array,nor::Array,dpr::Array,k::Function, blocks::Int64)
#    ϕ_hat = complex(zeros(N+1));
#    Lam = (λ.^(0:N)) ;
#    for j in(1:N+1) 
#        ϕ_hat[j] = ker_eval(M,ϕ[:,j],x,s_l[j],par,nor,dpr,k,blocks) #should be good. 
#    end
#    u = Lam.^(-1).*ifft(ϕ_hat);
#    return u;
#end
#
#function time_freq_sol(N::Integer,M::Integer, λ::Float64,Δt::Float64,x::Array,ϕ::Array,WN::Array,P::Array,Pinv::Array,par::Array,nor::Array,dpr::Array,k::Function,blocks::Int64,w_s::Float64)    
#    temp = complex( zeros( size(ϕ[:,1,1]) ) );  
#    ϕ_hat = complex(zeros(N+1));
#    Lam = (λ.^(0:N)) ; 
#    for j in(1:N+1) 
#        #γ = eigvals(Δ[j]); P = eigvecs(Δ[j]);
#        for ii in(1:length(P[1,1,:]))
#            #temp[ii] = P[length(γ),ii] * ker_eval(M,ϕ[ii,:,j],x, (im*γ[ii]/(w_s*Δt)) ,par,nor,dpr,k,blocks); 
#            temp[ii] = P[j,length(P[1,1,:]),ii] * ker_eval(M,ϕ[ii,:,j],x, WN[j,ii] ,par,nor,dpr,k,blocks); 
#            ϕ_hat[j] =ϕ_hat[j] + temp[ii];
#        end
#        ϕ_hat[j] =  λ*exp(-2*pi*im/(N+1))^(j-1) *ϕ_hat[j];
#    end
#    u = Lam.^(-1).*ifft(ϕ_hat);
#    return u;
#end
#
