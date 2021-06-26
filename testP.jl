
using Distributed;
using PyPlot;
using SharedArrays;

## PUT this on top !!!!
const numThreads = 3;
if (numThreads != 1)

	if (nprocs() == 1)
		addprocs(numThreads,lazy=false); 
		display(workers());
	end
		
end


@everywhere using SharedArrays;

@everywhere function dummy1(beginCell::Int64, endCell::Int64, a::SharedArray{Float64,2},b::SharedArray{Float64,2},c::SharedArray{Float64,2})

	 for i = beginCell:endCell
		 @inbounds c[i] = a[i] + b[i] + 0.5;
	 end
	
end


	

function dummyCalc(	nThreads::Int64, cellsPart::Array{Int64,2}, 
	a::SharedArray{Float64,2},b::SharedArray{Float64,2},c::SharedArray{Float64,2})


	@sync @distributed for p in workers()	
	
		beginCell::Int64 = cellsThreadsX[p-1,1];
		endCell::Int64 = cellsThreadsX[p-1,2];
		println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);
	
		# println("worker: ",p,"\ttermoX.gamma: ", thermoX.Gamma);
		# println("worker: ",p,"\ttestMeshX: ", testMeshX.nCells, "\t", testMeshX.nNodes );
		# println("worker: ",p,"\ttestfields2dX: ", testfields2dX.densityCells[299] );
	
		dummy1(beginCell,endCell, a, b, c);
	
	end
	
	@everywhere finalize(dummy1);
					
end
	
	
function prime()


	N = 9;

	a = SharedArray{Float64}(N,1);
	b = SharedArray{Float64}(N,1);
	c = SharedArray{Float64}(N,1);


	for i = 1:N
		a[i] = 1.0;
		b[i] = 2.0;
		c[i] = 0.0;
	end

	numThreads = 3;

	cellsThreads = Array{Int64,2}(undef,numThreads,2);
	cellsThreads[1,1] = 1;
	cellsThreads[1,2] = 3;

	cellsThreads[2,1] = 4;
	cellsThreads[2,2] = 6;

	cellsThreads[3,1] = 7;
	cellsThreads[3,2] = 9;

	@everywhere numThreadX = $numThreads;
	@everywhere cellsThreadsX = $cellsThreads;


	dummyCalc(numThreads,cellsThreads,a,b,c);

	
	println("a: ", a)
	println("b: ", b)
	println("c: ", c)
	
	for i = 1:N
		a[i] = 1.0;
		b[i] = 2.0;
		c[i] = 0.0;
	end
	
	println("a: ", a)
	println("b: ", b)
	println("c: ", c)
	
	dummyCalc(numThreads,cellsThreads,a,b,c);
	
	println("a: ", a)
	println("b: ", b)
	println("c: ", c)
	
	
	
end

prime()

				