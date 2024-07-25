#############################################################################

function Deriv_mat(h::Float64,N::Int64)
    M = zeros(N,N);
    for ii in(1:N)
        for jj in(1:N)
            if((ii-jj)%N==0)
                M[ii,jj] = 0;
            elseif((ii-jj)%N!=0)
                M[ii,jj] = 0.5*(-1)^(ii-jj) *cot((ii-jj)*h/2);
            end
        end
    end
    return M;
end

function matrix_coeff_T(n::Int,j::Int);
    if j==0 
        t =-n/2;
    elseif rem(j,2) == 0 
        t = 0;
    else
        t = 1/(2 * n * (sin(j * pi/2/n))^2);
    end;
    return t;
end


function Tmat(M::Int64)
    N = Int(M/2); h=pi/N; 
    A = complex(zeros(M,M)) 
    for ii in(1:M)
        tii = (pi/N)*(ii-1);
        for jj in(1:M)
            tjj = (pi/N)*(jj-1);
            A[ii,jj] = T_(h,N, ii,jj,tii,tjj);
        end
    end
    return A;     
end

function T_(h::Float64, N::Int64,i::Int64, j::Int64,ti::Float64,tj::Float64)
     hold =0.0;
        if(abs(i-j)%2==1)
          hold = (2*N*sin( (ti-tj)/2 )^2 )^(-1);
        elseif(abs(i-j)%2==0 && abs(i-j)!=0)
         hold=0.0;
     elseif(abs(i-j)==0)
         hold = -N/2
      end
     return hold
end

function Hilbert(n::Int64,n_r::Int64)
    Tits = zeros(n);
    for j=0:(n-1)
        Tits[j + 1] = matrix_coeff_T(n_r+1, j);
    end;

    AAtest = zeros(n,n);
    for i=1:n
        for j=1:n
            AAtest[i,j] = Tits[abs(i-j)+1];
        end
    end
    
    B = zeros(n_r,n_r)
    for i=1:n_r
        for j=1:n_r
            B[i,j] = AAtest[i+1,j+1] - AAtest[i+1,n-j+1];
        end;
    end;    
    return B;
end

function ALPERT_jordan(h::Float64,N_r::Int64, k::Function,Γ::Array,nvec::Array, dΓ::Array,MP::Array,DPAR::Array,DNOR::Array,DDPR::Array,UDPR::Array,WW::Complex)     
    A = complex(zeros(N_r,N_r));
    B = complex(zeros(N_r,N_r));
    C = complex(zeros(N_r,N_r));
    D = complex(zeros(N_r,N_r));
    for i in(1:N_r)
        for j in(1:N_r)
            if( abs(ll(i,j,N_r))>=a )
                xi = Γ[i];
                xj = Γ[j];
                B[i,j] = k(xi ,xj ,nvec[i],nvec[j],WW)*dΓ[j]*UDPR[i] ; 
            end
        end
    end
    for i in(1:N_r)
        for p in(1:2*m)                 
            for j in(1:N_r)
                if(any(x->x==ll(i,j,N_r),O[p,:]))   
                    noo= np_a(p,ll(i,j,N_r)) 
                    C[i,j] = C[i,j]+k(Γ[i], DPAR[i][p],nvec[i],DNOR[i][p],WW)*ws[p]*Lb11_aa(Int(noo),O[p,:],p)*UDPR[i]*DDPR[i][p];        
                end
            end
        end
    end
    
    A = h*(B + C)
    return  A;
    
end

 function ALPERT_arc(h::Float64,N_r::Int64, k::Function,Γ::Array,nvec::Array, dΓ::Array,MP::Array,DPAR::Array,DNOR::Array,DDPR::Array,TT1::Array,TT2::Array,SIN::Array,WW::Complex)     
    A = complex(zeros(N_r,N_r));
    B = complex(zeros(N_r,N_r));
    C = complex(zeros(N_r,N_r));
    D = complex(zeros(N_r,N_r));
    for i in(1:N_r)
        for j in(1:N_r)
            if( abs(ll(i,j,N_r))>=a )
                xi = Γ[i];
                xj = Γ[j];
                B[i,j] = k(xi ,xj ,nvec[i],nvec[j],WW )*dΓ[j]*dΓ[i]; 
            end
        end
    end
    for i in(1:N_r)
        for p in(1:2*m)                 
            for j in(1:N_r)
                if(any(x->x==ll(i,j,N_r),O[p,:]))   
                    noo= np_a(p,ll(i,j,N_r)) 
                    C[i,j] = C[i,j]+k(Γ[i], DPAR[i][p],nvec[i],DNOR[i][p],WW)*ws[p]*Lb11_aa(Int(noo),O[p,:],p)*dΓ[i]*DDPR[i][p];        
                end
            end
        end
    end
    
    A = h*(B + C)
    return  A;
    
end

function trap_arc(h::Float64,n_r::Int,k::Function,pr::Array,pr_p::Array, pr_pp::Array,pr_ppp::Array,TT1::Array,TT2::Array) 
    Mat = complex(zeros(n_r,n_r));
    for i in(1:n_r)
        for j in(1:n_r)
            if i!=j
                Mat[i,j] = TT2[i+1]*TT2[j+1]*k( TT1[i+1], TT1[j+1], pr[i], pr[j], pr_p[i], pr_p[j] ); 
            elseif i==j 
                Mat[i,j] = TT2[i+1]*TT2[i+1]*(- 0.5*dot(pr_p[i],pr_pp[i])^2/norm(pr_p[i])^4 +(1/4)*dot(pr_pp[i],pr_pp[i])/norm(pr_p[i])^2+ (1/6)*dot(pr_p[i],pr_ppp[i])/norm(pr_p[i])^2 );
            end
        end
    end 
    return -h*Mat/2 *(1/(2*pi));
end

function trap_jordan(h,n,k::Function,pr::Array,prp::Array, prpp::Array) 
    Mat = complex(zeros(n,n));
    for i in(1:n)
        for j in(1:n)
            if i!=j
                Mat[i,j] = k( 0.5*(2*i-1)*h, 0.5*(2*j-1)*h, pr[i], pr[j], prp[i]  );
            elseif i==j
                Mat[i,j] = -dot(prpp[i],prp[i])/norm(prp[i])^2; 
            end
        end
    end 
    return h*Mat;
end
function NEU_reflections(h::Float64,N::Int64,G::Function,par1::Array,par2::Array,nor1::Array,nor2::Array,dΓ1::Array,dΓ2::Array,MP::Array,WW::Complex) 
    A = complex(zeros(N,N));
    for i in(1:N)
        for j in(1:N)
            A[i,j] = G(par2[i],par1[j],nor2[i],nor1[j],WW)*dΓ1[j]*dΓ2[i]; #dΓ2[i] neeeds to be one in the DL case.    
        end
    end
    return h*(A);
end

function NEU_reflections_test(h::Float64,N::Int64,N1::Int64,N2::Int64,G::Function,par1::Array,par2::Array,nor1::Array,nor2::Array,dΓ1::Array,dΓ2::Array,MP::Array, TT1::Array, TT2::Array,WW::Complex) 
    A = complex(zeros(N,N));
    for i in(1:N2)
        for j in(1:N1)
            A[i,j] = dΓ2[i]*G(par2[i],par1[j],nor2[i],nor1[j],WW)*dΓ1[j]; #removed TT2   
        end
    end
    return h*(A)
end


function scatSYS(choice::Int64,h::Float64,h_r::Float64,N::Int64,N_r::Int64,f1::Array,f2::Array,G::Array{Function},K_rep::Array{Function},Gk::Function,N__::Function,N__arc::Function,Sη::Array,Kη::Array,GkGη::Function,Kk::Function,G0::Function,par::Array,
        parp::Array,parpp::Array,parppp::Array,nor::Array,dpr::Array,mp::Array,DPAR::Array,DNOR::Array,DDPR::Array,blocks::Int64,co::Array,specimen::Array,TT1::Array,TT2::Array,SIN::Array,Ns::Array,WW::Complex,UNO::Array,
        layer_rep::Int64,BC::Array{Int64})
    
    result = complex(zeros(blocks*N)) 
    A=complex(diagm(ones(blocks*N))) 
    
    AS =sparse(complex(diagm(ones(blocks*N))));
    
    hol1 = complex(zeros(blocks*N,blocks*N)) ;
    hol2 = complex(zeros(blocks*N));


    
    if(layer_rep==1)
        for i in(0:blocks-1)
            for j in(0:blocks-1)
                if(i==j)
                    if(BC[i+1]==1)
                        hold=al_nu(h,N,G[i+1],par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],WW);                     
                        A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] = hold;
                        AS[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] =inv(hold);
                    elseif(BC[i+1]==2)
                        hold=-diagm(dpr[i+1])+ALPERT_jordan(h,N,G[i+1],par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],dpr[i+1],WW);
                        A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] = hold;
                        AS[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] =inv(hold);
                    end
                elseif(i!=j)
                    if(BC[i+1]==1)
                        ##dir 
                        A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(j%blocks+1)*N] =DIR_reflections(h,N,K_rep[BC[i+1]],par[j+1],par[i+1],nor[j+1],dpr[j+1],mp[i+1],WW);
                    elseif(BC[i+1]==2)
                        ##neu
                        A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(j%blocks+1)*N] =NEU_reflections(h,N,K_rep[BC[i+1]],par[j+1],par[i+1],nor[j+1],nor[i+1],dpr[j+1],dpr[i+1],mp[i+1],WW);
                    end
                end
            end
        end

    elseif(layer_rep==2)
        for i in(0:blocks-1)
            for j in(0:blocks-1)
                leftx=i%blocks*N+1 ; rightx =(i%blocks+1)*N; 
                lefty=j%blocks*N+1 ; righty =(j%blocks+1)*N; 
                
                
                rleftx=i%blocks*N_r+1 ; rrightx =(i%blocks+1)*N_r; 
                rlefty=j%blocks*N_r+1 ; rrighty =(j%blocks+1)*N_r;
                
                leftXX = (i%blocks)*N+1
                rightXX = (i%blocks+1)*(N)-N_r-2
                
                leftYY = (j%blocks)*N+1
                rightYY = (j%blocks+1)*(N)-N_r-2
                
                if(i==j)               
                    if(BC[i+1]==1)
                        if(length(co[i+1])==0)
                            hold=0.5*I+al_nu(h,N,D,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],WW);
                            A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] = hold;
                            AS[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] =inv(hold);

                        elseif(length(co[i+1])!=0)
                            hold =trap(h,N,G0,par[i+1],nor[i+1],dpr[i+1],mp[i+1],WW) + al_nu(h,N,K_rep[BC[i+1]],par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],WW);
                            A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] = hold;
                            AS[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] =inv(hold);
                        end
                    elseif(BC[i+1]==2)
                        if(length(co[i+1])==0)
                            if(specimen[i+1]>=4)
                                AA =ALPERT_jordan(h,N,G[i+1],par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],dpr[i+1],WW);
                                B=Tmat(N);
                                C= (1/(2*pi))*trap_jordan(h,N,N__,par[i+1],parp[i+1],parpp[i+1])*Deriv_mat(h,N);               
                                hold = AA+B+C
                                A[ leftx:rightx , lefty:righty] = hold;
                                AS[ leftx:rightx , lefty:righty] = inv(hold);
                                
                            elseif(specimen[i+1]<4)                     
                                AA = 2*ALPERT_jordan(h_r,N_r,G[i+1],par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],dpr[i+1],WW)/2 #This /2 bothers me 
                                B=Hilbert(N,N_r);
                                C= 2*trap_arc(h_r,N_r,N__arc,par[i+1],parp[i+1],parpp[i+1],parppp[i+1],TT1,TT2);
                                hold = (B+C+AA);
                                A[ leftXX:rightXX , leftYY:rightYY] = hold;  
                                AS[ leftXX:rightXX , leftYY:rightYY] = inv(hold);  
                    
                            end
                            
                        elseif(length(co[i+1])!=0)  
                            Kk_mat =   ALPERT_jordan(h,N,Kk,  par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],UNO,WW); 
                            GkGη_mat = ALPERT_jordan(h,N,GkGη,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],UNO,WW); 
                            hold =  ( I-Kk_mat + ( -2*Sη[i+1]*GkGη_mat-2*Kη[i+1]*Kη[i+1]) )
                            A[leftx:rightx,lefty:righty] =hold ;
                            AS[leftx:rightx,lefty:righty] =inv(hold);

                            f1[leftx:rightx] = f1[leftx:rightx]+2*Sη[i+1]*f2[leftx:rightx];##LHS changed to f1..... 
                        end
                    end
                elseif(i!=j) #this is for the reflections
                    if(BC[i+1]==1)
                        A[ leftx:rightx, lefty:righty]=DIR_reflections_onstrip(h_r,N,Ns[j+1],Ns[i+1],K_rep[BC[i+1]],par[j+1],par[i+1],nor[j+1],dpr[j+1],mp[i+1],TT1,TT2,WW);;
                    elseif(BC[i+1]==2)
                        if(length(co[i+1])==0)
                            A[ leftx:rightx, lefty:righty]=NEU_reflections_test(h_r,N,Ns[j+1],Ns[i+1],Gk,par[j+1],par[i+1],nor[j+1],nor[i+1],dpr[j+1],dpr[i+1],mp[i+1],TT1,TT2,WW);;
                        elseif(co[i+1]!=0)
                            Gk_mat =NEU_reflections_test(h_r,N,Ns[j+1],Ns[i+1],Gk,par[j+1],par[i+1],nor[j+1],nor[i+1],dpr[j+1],UNO,mp[i+1],TT1,TT2,WW);;
                            Kk_mat =NEU_reflections_test(h_r,N,Ns[j+1],Ns[i+1],Kk,par[j+1],par[i+1],nor[j+1],nor[i+1],dpr[j+1],UNO,mp[i+1],TT1,TT2,WW);;
                            A[ leftx:rightx, lefty:righty] =-Kk_mat-Sη[i+1]*Gk_mat;                
                        end
                    end
                end
            end
        end
    end
  
    ###here was changed to f1
    mul!(hol1,AS,A) ;
    mul!(hol2,AS,f1) ;
    result=gmres(hol1,hol2,abstol=1e-12);
#    result=A\f1

    return result;
end


function ker_eval_mix(h::Float64,ON::Int64,phi::Array,z::Array,κ::Complex, par::Array,nor::Array,dpr::Array,k::Function, blocks::Int64,specimen::Array,layer_rep::Int64) 
    hold = Float64(complex(0));
    
    for jj in(0:blocks-1)
        if(specimen[jj+1]<4 && layer_rep==2)
            for p in(1:Int(ON/2)-1)
                xtp = par[jj+1][p];
                ntp = nor[jj+1][p];
                #hold = hold + 2*k(z,xtp,ntp,κ)*phi[p+jj*ON]*dpr[jj+1][p];
                hold = hold + k(z,xtp,ntp,κ)*phi[p+jj*ON]*dpr[jj+1][p]; ## I removed the 2. I still don't think its supposed to be there. 

         ##       print(2*k(z,xtp,ntp,κ)*phi[p+jj*ON]*dpr[jj+1][p], "this should not be zero\n")
            end
        else
            for p in(1:ON)
                xtp = par[jj+1][p];
                ntp = nor[jj+1][p];
                hold = hold + k(z,xtp,ntp,κ)*phi[p+jj*ON]*dpr[jj+1][p]; 
            end
        end
    end    
    
    hold = h*hold;
    #print(hold,"\n")
    #print(norm(phi)," norm of this density\n")
    return hold;
end

