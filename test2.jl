
using SharedArrays

mutable struct fields2d_shared
		
	densityCells::SharedArray{Float64,1}

end


dens = SharedArray{Float64}(100); 


zzz = fields2d_shared(dens)


