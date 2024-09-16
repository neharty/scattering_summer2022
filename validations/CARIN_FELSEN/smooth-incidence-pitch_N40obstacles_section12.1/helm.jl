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

#function moid(s,corner)
#        mult = floor(s/(2*pi));
#        s=s-mult*2*pi;
#        lenny = length(corner);
#        Ti = 0;
#        Tiplus = 0;
#        for i in(1:lenny-1)
#            if(corner[i]<=s<=corner[i+1])
#                Ti = corner[i];
#                Tiplus = corner[i+1];
#                break
#            end
#        end
#        
#        v=(s)->(1/pp-1/2)*((Ti+Tiplus -2*s)/(Tiplus-Ti))^3 +(1/pp)*(2s-Ti-Tiplus)/(Tiplus-Ti)+1/2;
#        temp = (Tiplus*v(s)^pp +Ti *( 1-v(s) )^pp)/(v(s)^pp+(1-v(s))^pp );
#        return temp;
#        temp =0;
#end
#
#function moid_prime(s,corner)
#        mult = floor(s/(2*pi));
#        s=s-mult*2*pi;
#        lenny = length(corner);
#        Ti = 0;
#        Tiplus = 0;
#        for i in(1:lenny-1)
#            if(corner[i]<=s<=corner[i+1])
#                Ti = corner[i];
#                Tiplus = corner[i+1];
#                break
#            end
#        end
#        
#        v=(s)->(1/pp-1/2)*((Ti+Tiplus -2*s)/(Tiplus-Ti))^3 +(1/pp)*(2s-Ti-Tiplus)/(Tiplus-Ti)+1/2;
#        vp=(s)->-6*(1/pp-1/2)* (Ti+Tiplus-2*s)^2 /(Tiplus-Ti)^3 + (2/pp)*(1/(Tiplus-Ti))
#        temp = (pp*vp(s)*(Tiplus-Ti)*(v(s)-v(s)^2)^(pp-1))/(v(s)^pp + (1-v(s))^pp )^2
#        return temp;
#        temp =0;
#end
#

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
#mult = floor(s/(2*pi));
#         s=s-mult*2*pi;
#         lenny = length(corner);
#         Ti = 0;
#         Tiplus = 0;
#         for i in(1:lenny-1)
#             if(corner[i]<=s<=corner[i+1])
#                 Ti = corner[i];
#                 Tiplus = corner[i+1];
#                 break
#             end
#         end
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



function trap(h::Float64,n::Int64,k::Function, PAR::Array,NOR::Array,DPR::Array,MP::Array,WN::Complex) #will be the specific trapezoidal rule for G0 in 5.18 in cats paper. 
    Mat = complex(zeros(n,n));
    for i in(1:n)
        for j in(1:n)
            if (i!=j)
                Mat[i,j] = k(PAR[i], PAR[j],NOR[j],WN)*DPR[j]*MP[i];
            elseif (i==j)
                off_d =  [k(PAR[i],PAR[jj],NOR[jj],WN)*DPR[jj]*MP[jj] for jj=1:n if jj!=i] 
                Mat[i,j] = -1*sum(off_d);
            end
        end
    end
    return h*Mat;
end


function reflections(h::Float64,N::Int64,G::Function,par1::Array,par2::Array,nor1::Array,dΓ1::Array,MP::Array,WN::Complex) 
    A = complex(zeros(N,N));
    for i in(1:N)
        for j in(1:N)
            A[i,j] = G(par2[i],par1[j],nor1[j],WN)*dΓ1[j]*MP[i]; #I feel as if this MP needs to be here....but im not sure how.   
        end
    end
    return h*(A);
end

function scatSYSS(choice::Int64,h::Float64,N::Int64,f::Array,G::Function,par::Array,nor::Array,dpr::Array,mp::Array,DPAR::Array,DNOR::Array,DDPR::Array,blocks::Int64,co::Array,WN::Complex)                          
        result = complex(zeros(blocks*N)) 
        A =complex(zeros(blocks*N,blocks*N))  
        for i in(0:blocks-1)
            for j in(0:blocks-1)
                if(i==j)
                    A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] =al_nu(h,N,G,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],WN); 
                elseif(i!=j)
                    A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(j%blocks+1)*N] =reflections(h,N,G,par[j+1],par[i+1],nor[j+1],dpr[j+1],mp[i+1],WN);
                end
            end
        end
        #print(cond(A),"this is the condition  \n")
	result = (A)\f;
	return result;

#        result = complex(zeros(blocks*N)) 
#        A =complex(zeros(blocks*N,blocks*N))
#        AA =sparse(complex(zeros(blocks*N,blocks*N)));
#        hol1 = complex(zeros(blocks*N,blocks*N)) ; 
#        hol2 = complex(zeros(blocks*N));      
#
#        for i in(0:blocks-1)
#            for j in(0:blocks-1)
#                if(i==j)
#                    hold=al_nu(h,N,G,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],WN);                     
#                    A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] = hold;
#                    AA[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] =inv(hold); 
#                elseif(i!=j)
#                    A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(j%blocks+1)*N] =reflections(h,N,G,par[j+1],par[i+1],nor[j+1],dpr[j+1],mp[i+1],WN);
#
#                end
#            end
#        end
#    
#    mul!(hol1,AA,A) ;
#    mul!(hol2,AA,f) ;
#    result=gmres(hol1,hol2,abstol=1e-5);

    return result;
end

#This works for multiple domains smooth, corners, whatever, but I have not found a way to generalize the coding.
function scatSYSD(choice::Int64,h::Float64,N::Int64,f::Array,G::Function,G0::Function,D::Function,par::Array,nor::Array,dpr::Array,mp::Array,DPAR::Array,DNOR::Array,DDPR::Array,blocks::Int64,co::Array,WN::Complex)
    result = complex(zeros(blocks*N)) 
    A =complex(zeros(blocks*N,blocks*N))   
    for i in(0:blocks-1)
        for j in(0:blocks-1)
            if(i==j)
                if(length(co[i+1])==0)
                    ##For smooth domains
                    A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] =0.5*I+al_nu(h,N,D,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],WN); 
                elseif(length(co[i+1])!=0)
                    ##for corner domains
                    A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(i%blocks+1)*N] =trap(h,N,G0,par[i+1],nor[i+1],dpr[i+1],mp[i+1],WN) + al_nu(h,N,G,par[i+1],nor[i+1],dpr[i+1],mp[i+1],DPAR[i+1],DNOR[i+1],DDPR[i+1],WN); 
                end
            elseif(i!=j)
                A[ i%blocks*N+1:(i%blocks+1)*N , j%blocks*N+1:(j%blocks+1)*N] =reflections(h,N,D,par[j+1],par[i+1],nor[j+1],dpr[j+1],mp[i+1],WN); #changed form G to D
            end
        end
    end
result = (A)\f;
return result;
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

