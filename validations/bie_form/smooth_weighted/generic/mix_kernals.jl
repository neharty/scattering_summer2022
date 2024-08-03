
################### Kernel Dirichlet Section ################
K0= (x,y,ny,WN)-> (1/(2*pi)) * (  dot(x-y,ny)/norm(x-y)^2 )   ;
D = (x,y,ny,WN)-> (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) )  ;
S = (x,y,ny,WN)-> (im/4) *besselh(0,1, WN*norm(x-y))  ;
C = (x,y,ny,WN)-> (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) ) -im*WN*(im/4)*besselh(0,1, WN*norm(x-y));

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




function SET_KERNEL(DoRN::Array{Int64},layer::Int64) #DoRN stands Dirichlet or Neumann
    
    kernals = Function[];
    
    for i in(1:length(DoRN)) ## for-loop begin ## --------------------------------
        
        if(DoRN[i]==1)
            if(layer==1) #Dirichlet trace operators
                #Single-layer equation.
                ker_hold= (x,y,ny,WN)->(im/4) *besselh(0,1, WN*norm(x-y ));
            elseif(layer==2)
                #Double-Layer
                ker_hold = (x,y,ny,WN)-> (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) ) - (1/(2*pi)) * (  dot(x-y,ny)/norm(x-y)^2 );
            elseif(layer==3)
                #Combined-Layer
                ker_hold = (x,y,ny,WN)-> ( (im*WN/4) * besselh(1,1,WN*norm(x-y ))*(  dot(x-y,ny)/norm(x-y) ) - (1/(2*pi)) * (  dot(x-y,ny)/norm(x-y)^2 ) ) -
                    im*WN*(im/4) *besselh(0,1, WN*norm(x-y ));
            end
        elseif(DoRN[i]==2) #Neumann Trace operators
            if(layer==1)
                ker_hold= (px,py,nx,ny,WN)->( -(im*WN/2)* besselh(1,1,WN*norm(px-py ))*(  dot(px-py,nx)/norm(px-py) )  );        
            elseif(layer==2)
                ker_hold=(px,py,nx,ny,WN)-> -2*( (-im*WN^2 /4)*besselh(0,1,WN*norm( px-py ))*dot(nx,px-py)*dot(ny,px-py)/norm(px-py)^2  ##There is a two here not in the strip case. 
                    + (im*WN/4 *norm(px-py)*besselh(1,1,WN*norm(px-py )) -1/(2*pi) )*
                        ( 2*dot(px-py,nx)*dot(px-py,ny)/norm(px-py)^4  -dot(nx,ny)/norm(px-py)^2 ) ) ; 
            elseif(layer==3) 
                #### This combined layer for the Neumann technically doesnt work. yes
                k=(px,py,nx,ny,WN)-> -2( (-im*WN^2 /4)*besselh(0,1,WN*norm( px-py ))*dot(nx,px-py)*dot(ny,px-py)/norm(px-py)^2  
                    + (im*WN/4 *norm(px-py)*besselh(1,1,WN*norm(px-py )) -1/(2*pi) )*
                    ( 2*dot(px-py,nx)*dot(px-py,ny)/norm(px-py)^4  -dot(nx,ny)/norm(px-py)^2 ) ) -
                    im*WN*( -(im*WN/2)* besselh(1,1,WN*norm(px-py ))*(  dot(px-py,nx)/norm(px-py) )  );
            end

        end
        
        kernals = push!(kernals,ker_hold);
                
        end ## for-loop end ## ----------------------------------------------
    
    return kernals;
    
end
