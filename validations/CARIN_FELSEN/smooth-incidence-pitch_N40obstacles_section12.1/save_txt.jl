function save_field_txt(T::Float64,NN::Int64,z::Array,inc_f::Array{Float64},obs::Array{Float64})
    
    td = T/NN;
    ddt=[td*i for i=0:NN];
    name ="amp"*string(amp)*"_start_"*string(Ω)*"_ws_"*string(w_s)*"_pp"*string(pp)*"_ord"*string(order)*"_over"*string(over)*"_fact"*string(Int(fact))*"_"*string(Int(T))*"mult_"*string(order)*"_nn"*string(nn)*"_NN"*string(NN)*"_"*dir*".txt";

    open(name, "w") do io
           writedlm(io, [ddt obs])
    end;

end



