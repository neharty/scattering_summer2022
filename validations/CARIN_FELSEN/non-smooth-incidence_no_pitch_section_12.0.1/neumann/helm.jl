pp=4;


################## 3rd order Alpert ###########################
# const chi=[1.150395811972836e-01
#             9.365464527949632e-01
#             ];
#
# const w=[3.913373788753340e-01
#             1.108662621124666e+00
#             ];
# const m=2; #order
# const a=2; #some parameter that accompanies the order of the method. 
# const TT =2*pi;
# const M = m+3;
# ord=3

######################### 4th order Alpert ##################################
const chi=[2.379647284118974e-02
            2.935370741501914e-01
            1.023715124251890e+00
            ];
     
const w=[8.795942675593887e-02
            4.989017152913699e-01
            9.131388579526912e-01
            ];

const m=3; #order
const a=2; #some parameter that accompanies the order of the method. 
const TT =2*pi;
const M = m+3;
ord=4



####################### 5th order Alpert Quadrature #######################
#const chi=[2.339013027203800e-02
#            2.854764931311984e-01
#            1.005403327220700e+00
#            1.994970303994294e+00
#            ];
#                                         
#const w=[8.609736556158105e-02
#            4.847019685417959e-01
#            9.152988869123725e-01
#            1.013901778984250e+00
#        ];
#
#const m=4; #order
#const a=3; #some parameter that accompanies the order of the method. 
#const TT =2*pi;
#const M = m+3;
#ord=5
#

########################### 6th order Alpert Quadrature ######################
#const chi=[4.004884194926570e-03
#            7.745655373336686e-02
#            3.972849993523248e-01
#            1.075673352915104e+00
#             2.003796927111872e+00
#            ];
#     
#const w=[1.671879691147102e-02
#             1.636958371447360e-01
#             4.981856569770637e-01
#             8.372266245578912e-01
#             9.841730844088381e-01
#            ];
#
#const m=5; #order
#const a=3; #some parameter that accompanies the order of the method. 
#const TT =2*pi;
#const M = m+3;
#ord=6
# #############################################################################

########################## 10th -Order Alpert ###################################
#const chi = [1.175089381227308e-03
#        1.877034129831289e-02
#        9.686468391426860e-02 
#        3.004818668002884e-01
#        6.901331557173356e-01
#        1.293695738083659e+00 
#        2.090187729798780e+00
#        3.016719313149212e+00
#        4.001369747872486e+00
#        5.000025661793423e+00 
#        ];
#const w =  [4.560746882084207e-03
#        3.810606322384757e-02
#        1.293864997289512e-01
#        2.884360381408835e-01
#        4.958111914344961e-01
#        7.077154600594529e-01
#        8.741924365285083e-01
#        9.661361986515218e-01
#        9.957887866078700e-01
#        9.998665787423845e-01        
#        ];
#const ww3= [7.832432020568779e+00
#        -4.565161670374749e+01
#        1.452168846354677e+02
#        -2.901348302886379e+02
#        3.870862162579900e+02
#        -3.523821383570681e+02
#        2.172421547519342e+02
#        -8.707796087382991e+01
#        2.053584266072635e+01
#        -2.166984103403823e+00
#        ];
#
#const m=10; #order
#const a=6;  #some parameter that accompanies the order of the method. 
#const TT =2*pi;
#const M = m+3
######################################
function l(x,y,q) #works from previous file check
    temp = mod(y-x,q)
    if temp>q/2
        temp = temp-q
    end
    return temp
end



#-----------------------------------------------------Alpert Lagrange----------------------------------
function mid_sum(N,y,h) #Works from previous file check
    sum = 0;
    for p in(a:N-a)
        sum = sum + k(y,y+ p*h)*sig(y+p*h)
    end
    sum = sum*h;
    return sum
end

#Matrix for index of alpert method. 
const chis = zeros(2*m);const ws = zeros(2*m);
for j=1:m
    chis[j] = chi[j];
    chis[m+j] = - chi[j];

    ws[j] = w[j];
    ws[m+j] = w[j];
end;

const O= zeros(2*m,M)
for p in(1:2*m)
    O[p,:]=[round(chis[p]-M/2)+i-1 for i=1:M];
end

function ll(x,y,q)
    a= mod(x-y,q)
    b=mod(y-x,q)
    res=0;
    if(a<b)
        res=-a;
    else
        res=b;
    end
    return res;
end

function o_a(p,u)  
    temp = round(chis[p]-M/2) + u
    return temp
    end

function np_a(p,q) 
    temp = q - round(chis[p]-M/2)+1
    return temp
end

function Lb11_aa(n,os,p) 
    res = 1.0
    for j in(1:length(os))
        if(j != n)
            res = res * ( ( chis[p] - os[j])/(os[n] - os[j] ) );   
        end
    end
    return res
end

function al_nu(h::Float64,N::Int64, k::Function,Γ::Array,nvec::Array, dΓ::Array,MP::Array,DPAR::Array,DNOR::Array,DDPR::Array,WN::Complex)     
    A = complex(zeros(N,N));
    B = complex(zeros(N,N));
    C = complex(zeros(N,N));

    for i in(1:N)
        for j in(1:N)
            if( abs(ll(i,j,N))>=a )
                xi = Γ[i];
                xj = Γ[j];
                B[i,j] = k(xi ,xj ,nvec[j],WN)*dΓ[j] *MP[i]; 
            end
        end
    end
    for i in(1:N)
        for p in(1:2*m)                 
            for j in(1:N)
                if(any(x->x==ll(i,j,N),O[p,:]))   
                    noo= np_a(p,ll(i,j,N))  
                    C[i,j] = C[i,j]+k(Γ[i], DPAR[i][p],DNOR[i][p],WN)*ws[p]*Lb11_aa(Int(noo),O[p,:],p)*dΓ[j] *MP[i];  ; #this works for smooth
                end
            end
        end
    end

    A = h*(B + C)
    return A
end

######################################## Sigmoid ###########################
function indexfinder_general(t::Float64, corner_list::Array)
    if (t<0)
        t = 2* pi +t;
    end
    if (t>2*pi)
        t = -2* pi +t;
    end
    j =0; #was zero
    for ll in(1:length(corner_list) - 1)
        if (t > corner_list[ll])
            j = j+1;
        end
    end
    return j
end

function moid(s,corner_list)
    jjj = indexfinder_general(s, corner_list);
    Ti = corner_list[jjj] ; Tiplus = corner_list[jjj + 1];
    v=(s)->(1/pp-1/2)*((Ti+Tiplus -2*s)/(Tiplus-Ti))^3 +(1/pp)*(2*s-Ti-Tiplus)/(Tiplus-Ti)+1/2;
    temp = (Tiplus*v(s)^pp +Ti *( 1-v(s) )^pp)/(v(s)^pp+(1-v(s))^pp );
    return temp;
    temp =0;
end

function moid_prime(s,corner_list)
    Ti = 0;
    Tiplus = 0;
    jjj = indexfinder_general(s, corner_list); 
    Ti = corner_list[jjj] ; Tiplus = corner_list[jjj + 1];
    v=(s)->(1/pp-1/2)*((Ti+Tiplus -2*s)/(Tiplus-Ti))^3 +(1/pp)*(2s-Ti-Tiplus)/(Tiplus-Ti)+1/2;
    vp=(s)->-6*(1/pp-1/2)* (Ti+Tiplus-2*s)^2 /(Tiplus-Ti)^3 + (2/pp)*(1/(Tiplus-Ti))
    temp = (pp*vp(s)*(Tiplus-Ti)*(v(s)-v(s)^2)^(pp-1))/(v(s)^pp + (1-v(s))^pp )^2
    return temp;
    temp =0;
end

function moid_pp(s,corner_list)
    temp= complex(0.0);
    Ti = 0;
    Tiplus = 0;
    jjj = indexfinder_general(s, corner_list); 
    v  = (s)->(1/pp-1/2)*((Ti+Tiplus -2*s)/(Tiplus-Ti))^3 +(1/pp)*(2*s-Ti-Tiplus)/(Tiplus-Ti)+1/2;
    vp = (s)->-6*(1/pp-1/2)* (Ti+Tiplus-2*s)^2 /(Tiplus-Ti)^3 + (2/pp)*(1/(Tiplus-Ti))
    vpp= (s)->( -12*(pp-2)*(Ti+Tiplus-2*s) )/(pp*(Tiplus-Ti)^3);
    temp = (Tiplus-Ti)*pp* ((v(s)-1)*v(s))^pp *( ( v(s)^pp *(2*v(s)-1-pp) +(1-v(s))^pp *(pp-1+2*v(s)) )*vp(s)^2 + (1-v(s))*v(s)*(v(s)^pp +(1-v(s))^pp )*vpp(s) )/
       ((v(s) -1)^2 *v(s)^2 *( (1-v(s))^pp +v(s)^pp )^3 );
    return temp;
end
##############################################################################

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
function reflections(h::Float64,N::Int64,G::Function,par1::Array,par2::Array,nor1::Array,nor2::Array,dΓ1::Array,dΓ2::Array,MP::Array,WW::Complex) 
    A = complex(zeros(N,N));
    for i in(1:N)
        for j in(1:N)
            A[i,j] = G(par2[i],par1[j],nor2[i],nor1[j],WW)*dΓ1[j]*dΓ2[i]; #dΓ2[i] neeeds to be one in the DL case.    
        end
    end
    return h*(A);
end

function reflections_test(h::Float64,N::Int64,N1::Int64,N2::Int64,G::Function,par1::Array,par2::Array,nor1::Array,nor2::Array,dΓ1::Array,dΓ2::Array,MP::Array, TT1::Array, TT2::Array,WW::Complex) 
    A = complex(zeros(N,N));
    for i in(1:N2)
        for j in(1:N1)
            A[i,j] = dΓ2[i]*G(par2[i],par1[j],nor2[i],nor1[j],WW)*dΓ1[j]; #removed TT2   
        end
    end
    return h*(A)
end
function scatSYSS(choice::Int64,h::Float64,N::Int64,f::Array,G::Function,par::Array,nor::Array,dpr::Array,mp::Array,DPAR::Array,DNOR::Array,DDPR::Array,blocks::Int64,co::Array,WW::Complex)
        result = complex(zeros(blocks*N))
        A =complex(zeros(blocks*N,blocks*N))
        for i in(0:blocks-1)
            for j in(0:blocks-1)
                if(i==j)
                    A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] = -diagm(dpr[i+1])+ALPERT_jordan(h,N,G,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],dpr[i+1],WW);
                elseif(i!=j)
                     A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(j%blocks+1)*N] =reflections(h,N,G,par[j+1],par[i+1],nor[j+1],nor[i+1],dpr[j+1],dpr[i+1],mp[i+1],WW);
                end
            end
        end
	result = (A)\f;
	return result;
end



## so mistake cannot be here. 
#function scatSYSD(choice::Int64,h::Float64,h_r::Float64,N::Int64,N_r::Int64,f::Array,G::Function,Gk::Function,N__::Function,N__arc::Function,Sη::Function,Kη::Function,GkGη::Function,par::Array,
#        parp::Array,parpp::Array,parppp::Array,nor::Array,dpr::Array,mp::Array,DPAR::Array,DNOR::Array,DDPR::Array,blocks::Int64,co::Array,specimen::Array,TT1::Array,TT2::Array,SIN::Array,Ns::Array,WW::Complex,UNO::Array)
function scatSYSD(choice::Int64,h::Float64,h_r::Float64,N::Int64,N_r::Int64,f::Array,G::Function,Gk::Function,N__::Function,N__arc::Function,Sη::Array,Kη::Array,GkGη::Function,par::Array,
        parp::Array,parpp::Array,parppp::Array,nor::Array,dpr::Array,mp::Array,DPAR::Array,DNOR::Array,DDPR::Array,blocks::Int64,co::Array,specimen::Array,TT1::Array,TT2::Array,SIN::Array,Ns::Array,WW::Complex,UNO::Array)
    
    result = complex(zeros(blocks*N)) 
    
#     A =complex(zeros(blocks*N,blocks*N))
#     A =complex(zeros(blocks*N_r,blocks*N_r))
    A=complex(diagm(ones(blocks*N))) 
    
    
#     holdi = complex(diagm(ones(N)));
    
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
               if(length(co[i+1])==0)
                    if(specimen[i+1]>=4)
                        AA =ALPERT_jordan(h,N,G,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],dpr[i+1],WW);
                        B=Tmat(N);
                        C= (1/(2*pi))*trap_jordan(h,N,N__,par[i+1],parp[i+1],parpp[i+1])*Deriv_mat(h,N);               
                        A[ leftx:rightx , lefty:righty] = AA+B+C; 
                        
                    elseif(specimen[i+1]<4)                     
#                         AA = 2*ALPERT_arc(h_r,N_r,G,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],TT1,TT2,SIN[1])/2 #
                        AA = 2*ALPERT_jordan(h_r,N_r,G,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],dpr[i+1],WW)/2 #
                        B=Hilbert(N,N_r);
                        C= 2*trap_arc(h_r,N_r,N__arc,par[i+1],parp[i+1],parpp[i+1],parppp[i+1],TT1,TT2);
                        A[ leftXX:rightXX , leftYY:rightYY] = (B+C+AA);  
            
                    end
                    
                elseif(length(co[i+1])!=0)  
                    ## Dirichlet + neumann trance
#                    Sη_mat = ALPERT_jordan(h,N,Sη,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],UNO,WW); 
#                    GkGη_mat = ALPERT_jordan(h,N,GkGη,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],UNO,WW); 
#                    Kη_mat  =  ALPERT_jordan(h,N,Kη,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],UNO,WW); 
#                    A[leftx:rightx,lefty:righty] = ( 0.5*I -2*Sη_mat*GkGη_mat-2*Kη_mat*Kη_mat );
#                    f[leftx:rightx] = 2*Sη_mat*f[leftx:rightx] ;


                    GkGη_mat = ALPERT_jordan(h,N,GkGη,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],UNO,WW); 
                    A[leftx:rightx,lefty:righty] = ( 0.5*I -2*Sη[i+1]*GkGη_mat-2*Kη[i+1]*Kη[i+1] );
                    f[leftx:rightx] = 2*Sη[i+1]*f[leftx:rightx] ;


                end
            elseif(i!=j) #this is for the reflections
                if(length(co[i+1])==0)
                                                   
                    A[ leftx:rightx, lefty:righty]=reflections_test(h_r,N,Ns[j+1],Ns[i+1],Gk,par[j+1],par[i+1],nor[j+1],nor[i+1],dpr[j+1],dpr[i+1],mp[i+1],TT1,TT2,WW);;
                
                elseif(co[i+1]!=0)
                    ###Neumann trace  ## Need to do this one as well. 
#                   #  Gk_mat = reflections(h,N,Gk,par[j+1],par[i+1],nor[j+1],nor[i+1],dpr[j+1],UNO,mp[i+1]); ##same deal
#                    Gk_mat =reflections_test(h_r,N,Ns[j+1],Ns[i+1],Gk,par[j+1],par[i+1],nor[j+1],nor[i+1],dpr[j+1],UNO,mp[i+1],TT1,TT2,WW);;
#                    Sη_mat = ALPERT_jordan(h,N,Sη,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],UNO,WW); 
#                    A[ leftx:rightx, lefty:righty] =-Sη_mat*Gk_mat;                

                    Gk_mat =reflections_test(h_r,N,Ns[j+1],Ns[i+1],Gk,par[j+1],par[i+1],nor[j+1],nor[i+1],dpr[j+1],UNO,mp[i+1],TT1,TT2,WW);;
                    A[ leftx:rightx, lefty:righty] =-Sη[i+1]*Gk_mat;                

                end

            end
        end
#         print("\n")
    end
#     imshow(abs.(A))
    result = (A)\f;
#     result = gmres(A,f);
    return result;
end







function ker_eval_jordan(h::Float64,ON::Int64,phi::Array,z::Array,κ::Complex, par::Array,nor::Array,dpr::Array,k::Function, blocks::Int64,specimen::Array)  
    hold = Float64(complex(0));
    
    for jj in(0:blocks-1)
        if(specimen[jj+1]<4)
            for p in(1:Int(ON/2)-1)
                xtp = par[jj+1][p];
                ntp = nor[jj+1][p];
                hold = hold + k(z,xtp,ntp,κ)*phi[p+jj*ON]*dpr[jj+1][p]; ##I dont actually know if the two is supposed to be here or not 
            end
        elseif(specimen[jj+1]>=4)
            for p in(1:ON)
                xtp = par[jj+1][p];
                ntp = nor[jj+1][p];
                hold = hold + k(z,xtp,ntp,κ)*phi[p+jj*ON]*dpr[jj+1][p]; 
            end
        end
    end    
    
    hold = h*hold; 
    return hold;
end

function experiment(y::Float64,h::Float64,N::Int64,Γ::Function,co::Array,LL::Float64,angle::Float64,shift::Array)
    PP=[];
    for p in(1:2*m)  
        PP=vcat(PP,[ Γ( y + chis[p]*h,co,LL,angle,shift) ])           
    end

    return PP
end

function Oexperiment(y::Float64,h::Float64,N::Int64,Γ::Function,co::Array,LL::Float64,angle::Float64)
    PP=[];
    for p in(1:2*m)  
        PP=vcat(PP,[ Γ( y + chis[p]*h,co,LL,angle) ])           
    end

    return PP
end

