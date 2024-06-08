function NEU_set_geo(which::Array, β::Float64)

    rot = angle->[cos(angle) -sin(angle)
                    sin(angle) cos(angle)];

    
    
    ## auxiliarry functions #########
    pn =1.0;  
    lp  = (x)-> ( abs(cos(x))^(pn) + abs(sin(x))^(pn) )^(1/pn)
    lpp = (x)-> sin(x)*cos(x)*( abs(sin(x))^(pn-2) - abs(cos(x))^(pn-2) )*lp(x)^(1-pn);
    lp2p = (x)-> (abs(cos(x))^pn + abs(sin(x))^pn )^(1/pn -2) *( 2*abs(sin(x)*cos(x))^pn *( sin(2*x)^2 *(pn-2) *( abs(sin(x) )^4 + abs(cos(x))^4 +2*abs(sin(x)*cos(x))^2 ) +
            4cos(2*x)* abs(sin(x)*cos(x))^2 *(abs(cos(x))^2 - abs(sin(x))^2 ) ) + abs(cos(x))^4 * abs(sin(x))^(2*pn) * (8*cos(2*x)*abs(sin(x))^2 +cos(4*x)-1) +
            abs(sin(x))^4 * abs(cos(x))^(2*pn) * (-8*cos(2*x)*abs(cos(x))^2 +cos(4*x) -1 ) ) /
            (8*abs(sin(x))^4 *abs(cos(x))^4 );
    

    ##Starfish
#     rr =t-> (1+  sin(5 * t)/4) 
#     r_p = (x)-> (5/4)*cos(5*x);
#     r_pp = (x)-> -(25/4)*sin(5*x);
#    r = t->( 9/20 - (1/9)*cos(5*t) );
#    r_p = (x)-> (5/9)*sin(5*x);
#    r_pp = (x)-> (25/9)*cos(5*x);
    r = t->( 9.0/10.0 - (1.0/10.0)*cos(5*t) );
    r_p = (x)-> (0.5)*sin(5*x);
    r_pp = (x)-> (2.5)*cos(5*x);
    
    ##Teardrop
    alpha = 1/2;
    beta = tan(alpha * pi/2);
    aux =t-> [2 * sin(t/2),-beta * sin(t)];
    #############
    
    ####cavity
    cl=4;
    Y =t-> sin(t) + sin(2*t) + sin(3*t)/cl;
    Yp =t-> cos(t) + 2*cos(2*t) + 3*cos(3*t)/cl;
    Yp_prime =t-> -sin(t) - 4*sin(2*t) - 9*sin(3*t)/cl;
    Ys =t-> -4*sin(t) + 7*sin(2*t) - 6*sin(3*t) + 2*sin(4*t);
    Ysp =t-> -4*cos(t) + 14*cos(2*t) - 18*cos(3*t) + 8*cos(4*t);
    Ysp_prime =t-> 4*sin(t) - 28*sin(2*t) + 54*sin(3*t) - 32*sin(4*t);
    
 
    fff=[]; 
    Arc=[];
    norm_v=[];
    CO=[];
    dx1 = [];
    dx2= [];
    dx3= [];
    
#      for i in(1:length(which))
     for i in(1:1)
        if(which[i]==1)
            ##strips
            co=[]; 
            f=(x,co,LL,angle,shift)->LL*rot(angle)*[cos(x),0]+shift;  
            arc = (x,co,LL,angle)->LL*sin(x); #inserted sin(x)
            nor =(x,co,LL,angle)->rot(angle)*[0,-1]; 
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*[1,0];
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*[0,0];
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0];
        elseif(which[i]==2)
            ##sqiggle
            co=[];    
            f=(x,co,LL,angle,shift)->LL*rot(angle)*[cos(x),sin(pi*cos(x))]+shift; 
            arc = (x,co,LL,angle)->LL*sqrt(1 + pi^2 *cos(pi*cos(x))^2)*sin(x); #inserted sin
            nor =(x,co,LL,angle)->rot(angle)*[pi*cos(pi*cos(x)),-1]/sqrt(1+pi^2 *cos(pi*cos(x))^2 ) ;
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*[1,pi*cos(pi*cos(x))];
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*[0,-pi^2 * sin(pi*cos(x))];
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,-pi^3 * cos(pi *cos(x))];
        elseif(which[i]==3)
            #partial-circle
            co=[];
            #β = pi/4;
            α = pi-β;
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[cos(α*cos(x)),sin(α*cos(x))]+shift;
            arc = (x,co,LL,angle)->LL*α*sqrt(cos(α*cos(x))^2+sin(α*cos(x))^2)*sin(x); #inserted sin
            nor =(x,co,LL,angle)->rot(angle)*LL*α*[cos(α*cos(x)),sin(α*cos(x))]/(LL*α*sqrt(cos(α*cos(x))^2+sin(α*cos(x))^2));
            dx1_t= (x,co,LL,angle)->α *rot(angle)*LL*[-sin(α*cos(x)),cos(α *cos(x))];
            dx2_t= (x,co,LL,angle)->-α^2 *rot(angle)*LL*[cos(α*cos(x)),sin(α*cos(x))];
            dx3_t= (x,co,LL,angle)->α^3 *rot(angle)*LL*[sin(α*cos(x)),-cos(α*cos(x))];
        elseif(which[i]==4)
            ##circle
            co=[]; #[0,0]
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[cos(x),sin(x)]+shift;
            arc = (x,co,LL,angle)->LL;
            nor =(x,co,LL,angle)->rot(angle)*[cos(x),sin(x)];
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*[-sin(x),cos(x)];
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*[-cos(x),-sin(x)];
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0];
        elseif(which[i]==5)
            ##boomerang
            co=[];#[0,0];
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[(cos(x)+0.65*cos(2*x)) , 1.5*sin(x)]+shift
            arc = (x,co,LL,angle)->LL*sqrt((sin(x)+1.3*sin(2*x) )^2 + (1.5*cos(x) )^2  );
            nor =(x,co,LL,angle)->rot(angle)*LL*[1.5*cos(x),sin(x)+1.3*sin(2*x)]/(  LL*sqrt((sin(x)+1.3*sin(2*x) )^2 + (1.5*cos(x) )^2  )  );
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*[(-sin(x)-1.3*sin(2*x)) , 1.5*cos(x)];
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*[(-cos(x)-2.6*cos(2*x)) , -1.5*sin(x)];
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0];           
        elseif(which[i]==6)
            ##Diamond
            co = [0,pi/2,pi, 3*pi/2, 2*pi];  
            uno = (x)-> -( lp(moid(x,co))*lp2p(moid(x,co))*moid_prime(x,co) + lp(moid(x,co))*lpp(moid(x,co))*moid_pp(x,co) -2*lpp(moid(x,co))*moid_prime(x,co) )/lp(moid(x,co))^3 ;
            dos = (x)-> (lp(moid(x,co))*moid_prime(x,co) -moid(x,co)*lp(moid(x,co)))/lp(moid(x,co))^2 ;
            f=(x,co,LL,angle,shift)->rot(angle)*LL* (1/lp(moid(x,co)))*[cos(moid(x,co)) , sin(moid(x,co))] +shift;
            arc = (x,co,LL,angle)->LL* (1/lp(moid(x,co)))^2 * sqrt(lp(moid(x,co))^2 + lpp(moid(x,co))^2 )*moid_prime(x,co);                        
            nor =(x,co,LL,angle)->rot(angle)*( (1/sqrt(  lp(moid(x,co))^2 + lpp(moid(x,co))^2  )) * [(cos(moid(x,co))*lp(moid(x,co)) - sin(moid(x,co))*lpp(moid(x,co)) ) ,(sin(moid(x,co))*lp(moid(x,co)) + 
                    cos(moid(x,co))*lpp(moid(x,co)))] );  
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*( -lp(moid(x,co))^(-2) *lpp(moid(x,co))*moid_prime(x,co)*[cos(moid(x,co)),sin(moid(x,co))] + lp(moid(x,co))^(-1) *moid_prime(x,co)*[-sin(moid(x,co)),cos(moid(x,co))]) ;
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*( (uno(x) - moid_prime(x,co)^2 /lp(moid(x,co)) )*[cos(moid(x,co)),sin(moid(x,co))] + 
                    ( -(lp(moid(x,co))*moid_prime(x,co))/lp(moid(x,co))^2 +dos(x))*[-sin(moid(x,co)),cos(moid(x,co))]);
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0];           
        elseif(which[i]==7)                     
            #starfish
            co=[];
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[ r(x)*cos(x),r(x)*sin(x)] +shift; 
            arc = (x,co,LL,angle)->LL*sqrt(r(x)^2 + r_p(x)^2);
            nor =(x,co,LL,angle)->rot(angle)*[r_p(x)*sin(x) + r(x)*cos(x), r(x)*sin(x) - r_p(x)*cos(x)]/sqrt(r(x)^2 + r_p(x)^2); 
            dx1_t =(x,co,LL,angle)->LL*rot(angle)* (r_p(x)*[cos(x),sin(x)] + r(x)*[-sin(x),cos(x)]); 
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*((r_pp(x)-r(x))*[cos(x),sin(x)]+ 2*r_p(x)*[-sin(x),cos(x)]);
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0]; # do not need this for closed boundaries
            
        elseif(which[i]==8)
            #I think this is a teardrop
            co =[0,2*pi];
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[2*sin(0.5*moid(x,co)), -sin(moid(x,co))]+shift 
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*[cos(0.5*moid(x,co)), -cos(moid(x,co))]*moid_prime(x,co);
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*([-0.5*sin(0.5*moid(x,co)) , sin(moid(x,co))]*moid_prime(x,co)+[cos(0.5*moid(x,co)), -cos(moid(x,co))]*moid_pp(x,co));
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0];
            arc = (x,co,LL,angle)-> LL*sqrt(cos(0.5*moid(x,co))^2 + cos(moid(x,co))^2)*moid_prime(x,co);
            nor =(x,co,LL,angle)-> rot(angle)*[-cos(moid(x,co)) , -cos(0.5*moid(x,co))]/sqrt(cos(0.5*moid(x,co))^2 + cos(moid(x,co))^2);
        elseif(which[i]==9)
            ##cavity
            co=[]
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[(cos(x)+2*cos(2*x))/2.5,(Y(x)-Ys(x)/24)]+shift; #24->14
            arc = (x,co,LL,angle)->LL*sqrt( (-(sin(x)+4*sin(2*x))/2.5 )^2 + ( (Yp(x)-Ysp(x)/24) )^2);
            nor =(x,co,LL,angle)->rot(angle)*[(Yp(x)-Ysp(x)/24),(sin(x)+4*sin(2*x))/2.5]/sqrt( (-(sin(x)+4*sin(2*x))/2.5 )^2 + ( (Yp(x)-Ysp(x)/24) )^2);
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*[-(sin(x)+4*sin(2*x))/2.5,(Yp(x)-Ysp(x)/24)];
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*[-(cos(x)+8*cos(2*x))/2.5,(Yp_prime(x)-Ysp_prime(x)/24)];
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0]; #do not need this for closed bounderies.
        elseif(which[i]==10)
            ## heart
            co=[0,pi,2*pi];
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[16*sin(moid(x,co))^3, 13*cos(moid(x,co))-5cos(2*moid(x,co))-2*cos(3*moid(x,co))-cos(4*moid(x,co))]+shift;
            arc = (x,co,LL,angle)->LL*sqrt((48*sin(moid(x,co))^2*cos(moid(x,co)))^2 +(-13*cos(moid(x,co))+10*sin(2*moid(x,co))+6*sin(3*moid(x,co))+4*sin(4*moid(x,co)))^2)*moid_prime(x,co); #need moid prime
            nor =(x,co,LL,angle)->rot(angle)*[-13*cos(moid(x,co))+10*sin(2*moid(x,co))+6*sin(3*moid(x,co))+4*sin(4*moid(x,co)), -48*sin(moid(x,co))^2 *cos(moid(x,co))]/sqrt((48*sin(moid(x,co))^2*cos(moid(x,co)))^2 +(-13*cos(moid(x,co))+10*sin(2*moid(x,co))+6*sin(3*moid(x,co))+4*sin(4*moid(x,co)))^2);
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*moid_prime(x,co)*[48*sin(moid(x,co))^2 *cos(moid(x,co)),-13*cos(moid(x,co))+10*sin(2*moid(x,co))+6*sin(3*moid(x,co))+4*sin(4*moid(x,co))];
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*(moid_pp(x,co)*[48*sin(moid(x,co))^2 *cos(moid(x,co)),-13*cos(moid(x,co))+10*sin(2*moid(x,co))+6*sin(3*moid(x,co))+4*sin(4*moid(x,co))]+ 
                                moid_prime(x,co)*[48*sin(moid(x,co))*(2*cos(moid(x,co))^2-sin(moid(x,co))^2),-13*cos(moid(x,co))+20*cos(2*moid(x,co))+18*cos(3*moid(x,co))+16*cos(4*moid(x,co))]) ;
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0]; 
        elseif(which[i]==11) 
            ## A weird domain with two corners. 
            co=[0,pi,2*pi];
            αα=(x,co)->( 2-abs(sin(moid(x,co))) );
            αα_p=(x,co)->-sin(moid(x,co))*cos(moid(x,co))/abs(sin(moid(x,co))) ;
            αα_pp=(x,co)->sin(moid(x,co))^2/abs(sin(moid(x,co))) ;
            
            f=(x,co,LL,angle,shift)->rot(angle)*LL*αα(x,co)*[cos(moid(x,co)),sin(moid(x,co))] +shift
            arc = (x,co,LL,angle)->LL*sqrt(αα_p(x,co)^2 +αα(x,co)^2 )*moid_prime(x,co);
            nor =(x,co,LL,angle)->rot(angle)*[αα_p(x,co)*sin(moid(x,co))+αα(x,co)*cos(moid(x,co)), αα(x,co)*sin(moid(x,co))-αα_p(x,co)*cos(moid(x,co))]/sqrt( αα_p(x,co)^2 +αα(x,co)^2);
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*(αα_p(x,co)*[cos(moid(x,co)),sin(moid(x,co))] +αα(x,co)*[-sin(moid(x,co)),cos(moid(x,co))])*moid_prime(x,co);
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*( moid_pp(x,co)*( αα_p(x,co)*[cos(moid(x,co)),sin(moid(x,co))]+ αα(x,co)*[-sin(moid(x,co)),cos(moid(x,co))]) 
                                    +moid_prime(x,co)^2 *((αα_pp(x,co)-αα(x,co))*[cos(moid(x,co)),sin(moid(x,co))] +2*αα_p(x,co)*[-sin(moid(x,co)),cos(moid(x,co))] )   );
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0]; 
        elseif(which[i]==12)
            ## alternate square, but it the αα_p and αα_pp are not correct. 
            pn =1.0;  
            co = [0,pi/2,pi, 3*pi/2, 2*pi];
            αα=(x,co)->1.0/( abs(cos(moid(x,co)))^(pn) + abs(sin(moid(x,co)))^(pn) )^(1/pn);
            αα_p=(x,co)->sin(moid(x,co))*cos(moid(x,co))*(-(abs(sin(moid(x,co)))^(pn-2) - abs(cos(moid(x,co)))^(pn-2)))*(abs(sin(moid(x,co)))^pn + abs(cos(moid(x,co)))^pn)^(-(pn + 1)/pn);
            αα_pp=(x,co)->(abs(cos(moid(x,co)))^pn + abs(sin(moid(x,co)))^pn )^(1/pn -2) *( 2*abs(sin(moid(x,co))*cos(moid(x,co)))^pn *( sin(2*x)^2 *(pn-2) *( abs(sin(moid(x,co)) )^4 + abs(cos(moid(x,co)))^4 +2*abs(sin(moid(x,co))*cos(moid(x,co)))^2 ) +
                    4*cos(2*moid(x,co))* abs(sin(moid(x,co))*cos(moid(x,co)))^2 *(abs(cos(moid(x,co)))^2 - abs(sin(moid(x,co)))^2 ) ) + abs(cos(moid(x,co)))^4 * abs(sin(moid(x,co)))^(2*pn) * (8*cos(2*moid(x,co))*abs(sin(moid(x,co)))^2 +cos(4*moid(x,co))-1) +
                    abs(sin(moid(x,co)))^4 * abs(cos(moid(x,co)))^(2*pn) * (-8*cos(2*x)*abs(cos(moid(x,co)))^2 +cos(4*moid(x,co)) -1 ) ) /
                    (8*abs(sin(moid(x,co)))^4 *abs(cos(moid(x,co)))^4 ) ;

            ## fairly sure the below is correct. 
            f=(x,co,LL,angle,shift)->rot(angle)*LL*αα(x,co)*[cos(moid(x,co)),sin(moid(x,co))] +shift
            arc = (x,co,LL,angle)->LL*sqrt(αα_p(x,co)^2 +αα(x,co)^2 )*moid_prime(x,co);
            nor =(x,co,LL,angle)->rot(angle)*[αα_p(x,co)*sin(moid(x,co))+αα(x,co)*cos(moid(x,co)), αα(x,co)*sin(moid(x,co))-αα_p(x,co)*cos(moid(x,co))]/sqrt( αα_p(x,co)^2 +αα(x,co)^2);
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*(αα_p(x,co)*[cos(moid(x,co)),sin(moid(x,co))] +αα(x,co)*[-sin(moid(x,co)),cos(moid(x,co))])*moid_prime(x,co);
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*( moid_pp(x,co)*( αα_p(x,co)*[cos(moid(x,co)),sin(moid(x,co))]+ αα(x,co)*[-sin(moid(x,co)),cos(moid(x,co))]) 
                                    +moid_prime(x,co)^2 *((αα_pp(x,co)-αα(x,co))*[cos(moid(x,co)),sin(moid(x,co))] +2*αα_p(x,co)*[-sin(moid(x,co)),cos(moid(x,co))] )   );
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0];     
        elseif(which[i]==13)
            ## A weird domain with two corners. 
            co = [0,pi/2,pi, 3*pi/2, 2*pi];  
            ββ=2.0;
            αα=(x,co)->( 2-abs(sin(ββ*moid(x,co))) );
            αα_p=(x,co)->-ββ*sin(ββ*moid(x,co))*cos(ββ*moid(x,co))/abs(sin(ββ*moid(x,co))) ;
            αα_pp=(x,co)->ββ*sin(ββ*moid(x,co))^2/abs(sin(ββ*moid(x,co))) ;
            
            f=(x,co,LL,angle,shift)->rot(angle)*LL*αα(x,co)*[cos(moid(x,co)),sin(moid(x,co))] +shift
            arc = (x,co,LL,angle)->LL*sqrt(αα_p(x,co)^2 +αα(x,co)^2 )*moid_prime(x,co);
            nor =(x,co,LL,angle)->rot(angle)*[αα_p(x,co)*sin(moid(x,co))+αα(x,co)*cos(moid(x,co)), αα(x,co)*sin(moid(x,co))-αα_p(x,co)*cos(moid(x,co))]/sqrt( αα_p(x,co)^2 +αα(x,co)^2);
            dx1_t =(x,co,LL,angle)->LL*rot(angle)*(αα_p(x,co)*[cos(moid(x,co)),sin(moid(x,co))] +αα(x,co)*[-sin(moid(x,co)),cos(moid(x,co))])*moid_prime(x,co);
            dx2_t =(x,co,LL,angle)->LL*rot(angle)*( moid_pp(x,co)*( αα_p(x,co)*[cos(moid(x,co)),sin(moid(x,co))]+ αα(x,co)*[-sin(moid(x,co)),cos(moid(x,co))]) 
                                    +moid_prime(x,co)^2 *((αα_pp(x,co)-αα(x,co))*[cos(moid(x,co)),sin(moid(x,co))] +2*αα_p(x,co)*[-sin(moid(x,co)),cos(moid(x,co))] )   );
            dx3_t =(x,co,LL,angle)->LL*rot(angle)*[0,0]; 
            
        end
                
        fff = push!(fff,f);
        Arc = push!(Arc,arc);
        norm_v = push!(norm_v,nor);
        CO = push!(CO,co);
        dx1 = push!(dx1,dx1_t);
        dx2 = push!(dx2,dx2_t);
        dx3 = push!(dx3,dx3_t);
        
    end
                
    return fff[1],Arc[1],norm_v[1],CO[1],dx1[1],dx2[1],dx3[1]; 
end

