
### different way of doing the geometry
function DIR_set_geo(which::Array,β::Float64)

    rot = angle->[cos(angle) -sin(angle)
                    sin(angle) cos(angle)];

    ## auxiliarry functions #########
    
    #Diamond
    pn =1;  #with a value of one we have a diamond.·
    lp  = (x)-> ( abs(cos(x))^(pn) + abs(sin(x))^(pn) )^(1/pn)
    lpp = (x)-> sin(x)*cos(x)*( abs(sin(x))^(pn-2) - abs(cos(x))^(pn-2) )*lp(x)^(1-pn);
    
    ##Starfish
    rr =t-> (4.0/5.0-cos(5*t)/5.0) 
    r_p = (x)-> sin(5*x);
#    rr =t-> (1+ 0.25 * sin(5 * t)) 
#    r_p = (x)-> (1.25)*cos(5*x);
#     rr = t->( 9/20 - (1/9)*cos(5*t) );
#     r_p = (x)-> (5/9)*sin(5*x);
    
    ##Teardrop
    alpha = 1/2;
    #beta = tan(alpha * pi/2);
    #aux =t-> [2 * sin(t/2),-beta * sin(t)];
    beta = tan(alpha*pi/2)/2;
    aux=t->[sin(t/2), -beta*sin(t)];

    #############
   
    ####cavity
    cl=4;
    Y =t-> sin(t) + sin(2*t) + sin(3*t)/cl;
    Yp =t-> cos(t) + 2*cos(2*t) + 3*cos(3*t)/cl;
    Ys =t-> -4 * sin(t) + 7 * sin(2*t) - 6 * sin(3*t) + 2*sin(4*t);
    Ysp =t-> -4 * cos(t) + 14*cos(2*t) - 18*cos(3*t) + 8*cos(4*t);
    
    fff=[]; 
    Arc=[];
    norm_v=[];
    CO=[];
    
    dx1 =[];
    dx2= [];
    dx3= [];
    
#      for i in(1:length(which))
     for i in(1:1)
        if(which[i]==1)
            ##strips
            co=[0,2*pi];
            f=(x,co,LL,angle,shift)->LL*[(1-moid(x,co)/(2*pi))*cos(angle)+(moid(x,co)/(2*pi))*cos(angle+pi),(1-moid(x,co)/(2*pi))*sin(angle)+(moid(x,co)/(2*pi))*sin(angle+pi) ]+shift;
            arc = (x,co,LL,angle)->LL/pi;
            nor =(x,co,LL,angle)->[sin(angle),-cos(angle)];
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
        elseif(which[i]==2)
             ##a squigely strip;
            co=[0,2*pi];
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[moid(x,co)-pi,sin(moid(x,co))]+shift;
            arc = (x,co,LL,angle)->LL*sqrt(1+cos(moid(x,co))^2);
            nor =(x,co,LL,angle)->rot(angle)*LL*[cos(moid(x,co)),-1]/arc(x,co,LL,angle);
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
            
        elseif(which[i]==3)
            co=[0,2*pi]; ###SO this parametrization of open aperture will only work for anlge
            #β = pi/4;
            α = (2*pi-2*β)/(2*pi);
            f=(x,co,LL,angle,shift)->rot(angle+β)*LL*[cos(moid(x,co)*α),sin(moid(x,co)*α)]+shift;
            arc = (x,co,LL,angle)->LL*sqrt(cos(moid(x,co)*α)^2+sin(moid(x,co)*α)^2);
            nor =(x,co,LL,angle)->rot(angle+β)*LL*[cos(moid(x,co)*α),sin(moid(x,co)*α)]/arc(x,co,LL,angle);
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
            
        elseif(which[i]==4)
            ##circle
            co=[]; #[0,0]
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[cos(x),sin(x)]+shift;
            arc = (x,co,LL,angle)->LL;
            nor =(x,co,LL,angle)->rot(angle)*[cos(x),sin(x)];
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
            
        elseif(which[i]==5)
            ##boomerang
            co=[];#[0,0];
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[(cos(x)+0.65*cos(2*x)) , 1.5*sin(x)]+shift
            arc = (x,co,LL,angle)->LL*sqrt((sin(x)+1.3*sin(2*x) )^2 + (1.5*cos(x) )^2  );
            nor =(x,co,LL,angle)->rot(angle)*LL*[1.5*cos(x),sin(x)+1.3*sin(2*x)]/(  LL*sqrt((sin(x)+1.3*sin(2*x) )^2 + (1.5*cos(x) )^2  )  );
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
            
            
        elseif(which[i]==6)
            ##Diamond
            co=[0,pi/2,pi, 3*pi/2, 2*pi];
            f=(x,co,LL,angle,shift)->rot(angle)*LL*(1/lp(moid(x,co)))*[cos(moid(x,co)) , sin(moid(x,co))]+shift; ##Diamond
            arc = (x,co,LL,angle)->LL*(1/lp(moid(x,co)))^2 * sqrt(lp(moid(x,co))^2 + lpp(moid(x,co))^2) ;
            nor =(x,co,LL,angle)->rot(angle)*(1/sqrt(  lp(moid(x,co))^2 + lpp(moid(x,co))^2  )) * [(cos(moid(x,co))*lp(moid(x,co)) - sin(moid(x,co))*lpp(moid(x,co)) ) ,(sin(moid(x,co))*lp(moid(x,co)) + cos(moid(x,co))*lpp(moid(x,co)))];
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
            
        elseif(which[i]==7)
            ## starfish
            co=[];#[0,0];
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[ rr(x)*cos(x),rr(x)*sin(x)]+shift;
            arc = (x,co,LL,angle)->LL*sqrt(rr(x)^2 + r_p(x)^2);
            nor =(x,co,LL,angle)->rot(angle)*LL*[r_p(x)*sin(x) + rr(x)*cos(x), rr(x)*sin(x) - r_p(x)*cos(x)]/(LL*sqrt(rr(x)^2 + r_p(x)^2)) ; ##issue has to be here
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
            
        elseif(which[i]==8)
            ##teardrop 
            co=[0,2*pi]
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[2*sin(0.5*moid(x,co)), -sin(moid(x,co))] +shift;#rot(angle)*LL*[2 * sin(moid(x,co)/2),-beta * sin(moid(x,co))]+shift;
            arc = (x,co,LL,angle)->LL*sqrt(cos(0.5*moid(x,co))^2 + cos(moid(x,co))^2);#LL*sqrt(cos(moid(x,co)/2)^2+ beta^2 * cos(moid(x,co))^2); ###THIS ONE I NEED TO CHECK;
            nor =(x,co,LL,angle)->rot(angle)*LL*[-cos(moid(x,co)) , -cos(0.5*moid(x,co))]/sqrt(cos(0.5*moid(x,co))^2 + cos(moid(x,co))^2);#rot(angle)*LL*[-beta*cos(moid(x,co)),-cos(moid(x,co)/2)]/(  LL*sqrt(cos(moid(x,co)/2)^2+beta^2 * cos(moid(x,co))^2)  );
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
 
        elseif(which[i]==9)
           ##cavity
            co=[]
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[(cos(x)+2*cos(2*x))/2.5,(Y(x)-Ys(x)/24)]+shift; #24->14
            arc = (x,co,LL,angle)->LL*sqrt( (-(sin(x)+4*sin(2*x))/2.5 )^2 + ( (Yp(x)-Ysp(x)/24) )^2);
            nor =(x,co,LL,angle)->rot(angle)*LL*[(Yp(x)-Ysp(x)/24),(sin(x)+4*sin(2*x))/2.5]/arc(x,co,LL,angle);
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
            
        elseif(which[i]==10)
            ##a Heart;
            co=[0,pi,2*pi];
            rad =(x,co)->( (sin(moid(x,co)-pi/2)*sqrt(abs(cos(moid(x,co)-pi/2))) )/(sin(moid(x,co)-pi/2) +7/5 ) -2*sin(moid(x,co)-pi/2) +2 );
            radp =(x,co)-> (1/2)*cos(moid(x,co)-pi/2)*(-(5*sin(moid(x,co)-pi/2)^2)/((5*sin(moid(x,co)-pi/2) + 7)*abs(cos(moid(x,co)-pi/2))^(3/2)) + (70*sqrt(abs(cos(moid(x,co) -pi/2))))/(5*sin(moid(x,co)-pi/2) + 7)^2 - 4);
            f=(x,co,LL,angle,shift)->rot(angle)*LL*rad(x,co)*[cos(moid(x,co)-pi/2),sin(moid(x,co)-pi/2)]+shift;
            arc = (x,co,LL,angle)->LL*sqrt(radp(x,co)^2 +rad(x,co)^2);
            nor = (x,co,LL,angle)->rot(angle)*[radp(x,co)*sin(moid(x,co)-pi/2)+rad(x,co)*cos(moid(x,co)-pi/2),-radp(x,co)*cos(moid(x,co)-pi/2)+rad(x,co)*sin(moid(x,co)-pi/2)]/sqrt(radp(x,co)^2 +rad(x,co)^2);
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
        elseif(which[i]==11)
            co=[0,2*pi];
            f=(x,co,LL,angle,shift)->rot(angle)*LL*[3/2 *sin(3/2 *moid(x,co)),sin(moid(x,co))]+shift;
            arc = (x,co,LL,angle)->LL*sqrt( (9/4 *cos(3/2 *moid(x,co)) )^2 + cos(moid(x,co))^2);
            nor = (x,co,LL,angle)->rot(angle)*[cos(moid(x,co)),-9/4 *cos(3/2 *moid(x,co))]/sqrt( (9/4 *cos(3/2 *moid(x,co)))^2 + cos(moid(x,co))^2) ;
            dx1_t =(x,co,LL,angle)->0;
            dx2_t =(x,co,LL,angle)->0;
            dx3_t =(x,co,LL,angle)->0;
        elseif(which[i]==12)
#             co=[0,2*pi]; ###SO this parametrization of open aperture will only work for anlge
#             β = pi/4;
#             α = (2*pi-2*β)/(2*pi);
#             f=(x,co,LL,angle,shift)->rot(angle+β)*LL*[cos(moid(x,co)*α),sin(moid(x,co)*α)]+shift;
#             arc = (x,co,LL,angle)->LL*sqrt(cos(moid(x,co)*α)^2+sin(moid(x,co)*α)^2);
#             nor =(x,co,LL,angle)->rot(angle+β)*LL*[cos(moid(x,co)*α),sin(moid(x,co)*α)]/arc(x,co,LL,angle);
#             dx1_t =(x,co,LL,angle)->0;
#             dx2_t =(x,co,LL,angle)->0;
#             dx3_t =(x,co,LL,angle)->0;
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

