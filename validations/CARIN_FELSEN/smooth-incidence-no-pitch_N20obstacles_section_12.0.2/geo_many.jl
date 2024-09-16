function set_geo(which::Array)
    ####auxiliarry functions 
    rot = angle->[cos(angle) -sin(angle)
                            sin(angle) cos(angle)];


    ##Diamond
    pn =1;  
    lp  = (x)-> ( abs(cos(x))^(pn) + abs(sin(x))^(pn) )^(1/pn)
    lpp = (x)-> sin(x)*cos(x)*( abs(sin(x))^(pn-2) - abs(cos(x))^(pn-2) )*lp(x)^(1-pn);

    ##Starfish
    rr =t-> (1+ 0.25 * sin(5 * t)) 
    r_p = (x)-> (1.25)*cos(5*x);
    #rr = t->( 9/20 - (1/9)*cos(5*t) );
    #r_p = (x)-> (5/9)*sin(5*x);

    ##Teardrop
    alpha = 1/2;
    beta = tan(alpha * pi/2);
    aux =t-> [2 * sin(t/2),-beta * sin(t)];

    ##cavity
    Y =t-> sin(t) + sin(2*t) + sin(3*t)/2;
    Yp =t-> cos(t) + 2*cos(2*t) + 3*cos(3*t)/2;
    Ys =t-> -4 * sin(t) + 7 * sin(2*t) - 6 * sin(3*t) + 2*sin(4*t);
    Ysp =t-> -4 * cos(t) + 14*cos(2*t) - 18*cos(3*t) + 8*cos(4*t);


    fff=[]; 
    Arc=[];
    norm_v=[];
    CO=[];

    for i in(1:length(which))
    if(which[i]==1)
        #strips
        co=[0,2*pi];
        f=(x,co,LL,angle,shift)->LL*[(1-moid(x,co)/(2*pi))*cos(angle)+(moid(x,co)/(2*pi))*cos(angle+pi),(1-moid(x,co)/(2*pi))*sin(angle)+(moid(x,co)/(2*pi))*sin(angle+pi) ]+shift;
        arc = (x,co,LL,angle)->1.0#LL/pi;
        nor =(x,co,LL,angle)->[sin(angle),-cos(angle)];
    elseif(which[i]==2)
        #circle
        co=[]; 
        f=(x,co,LL,angle,shift)->rot(angle)*LL*[cos(x),sin(x)]+shift;
        arc = (x,co,LL,angle)->LL;
        nor =(x,co,LL,angle)->rot(angle)*[cos(x),sin(x)];
    elseif(which[i]==3)
        #boomerang
        co=[];
        f=(x,co,LL,angle,shift)->rot(angle)*LL*[(cos(x)+0.65*cos(2*x)) , 1.5*sin(x)]+shift
        arc = (x,co,LL,angle)->LL*sqrt((sin(x)+1.3*sin(2*x) )^2 + (1.5*cos(x) )^2  );
        nor =(x,co,LL,angle)->rot(angle)*LL*[1.5*cos(x),sin(x)+1.3*sin(2*x)]/(  LL*sqrt((sin(x)+1.3*sin(2*x) )^2 + (1.5*cos(x) )^2  )  );
    elseif(which[i]==4)
        #Diamond
        co=[0,pi/2,pi, 3*pi/2, 2*pi];
        f=(x,co,LL,angle,shift)->rot(angle)*LL*(1/lp(moid(x,co)))*[cos(moid(x,co)) , sin(moid(x,co))]+shift; ##Diamond
        arc = (x,co,LL,angle)->LL*(1/lp(moid(x,co)))^2 * sqrt(lp(moid(x,co))^2 + lpp(moid(x,co))^2) ;
        nor =(x,co,LL,angle)->rot(angle)*(1/sqrt(  lp(moid(x,co))^2 + lpp(moid(x,co))^2  )) * [(cos(moid(x,co))*lp(moid(x,co)) - sin(moid(x,co))*lpp(moid(x,co)) ) ,(sin(moid(x,co))*lp(moid(x,co)) + cos(moid(x,co))*lpp(moid(x,co)))];
    elseif(which[i]==5)
        #starfish
        co=[];
        f=(x,co,LL,angle,shift)->rot(angle)*LL*[ rr(x)*cos(x),rr(x)*sin(x)]+shift;
        arc = (x,co,LL,angle)->LL*sqrt(rr(x)^2 + r_p(x)^2);
        nor =(x,co,LL,angle)->rot(angle)*LL*[r_p(x)*sin(x) + rr(x)*cos(x), rr(x)*sin(x) - r_p(x)*cos(x)]/(LL*sqrt(rr(x)^2 + r_p(x)^2)) ; 
    elseif(which[i]==6)
        #teardrop 
        co=[0,2*pi]
        f=(x,co,LL,angle,shift)->rot(angle)*LL*[2*sin(0.5*moid(x,co)), -sin(moid(x,co))] +shift;
        arc = (x,co,LL,angle)->LL*sqrt(cos(0.5*moid(x,co))^2 + cos(moid(x,co))^2);
        nor =(x,co,LL,angle)->rot(angle)*LL*[-cos(moid(x,co)) , -cos(0.5*moid(x,co))]/sqrt(cos(0.5*moid(x,co))^2 + cos(moid(x,co))^2);
    elseif(which[i]==7)
        #cavity
        co=[]
        f=(x,co,LL,angle,shift)->rot(angle)*LL*[(cos(x)+2*cos(2*x))/2.5,(Y(x)-Ys(x)/24)]+shift; #24->14
        arc = (x,co,LL,angle)->LL*sqrt( (-1*(sin(x)+4*cos(x))/2.5 )^2 + ( (Yp(x)-Ysp(x))/24 )^2);
        nor =(x,co,LL,angle)->[0,0];
    end

    fff = push!(fff,f);
    Arc = push!(Arc,arc);
    norm_v = push!(norm_v,nor);
    CO = push!(CO,co);
    end

    return fff,Arc,norm_v,CO; 
end


##### --------------- Geo End ________--------___________----------_________


