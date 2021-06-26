

using Distributed;
using PyPlot;

const pname = "testTriMesh2d.bson"
const numThreads = 4;


if (numThreads != 1)

	if (nprocs() == 1)
		addprocs(numThreads,lazy=false); 
		display(workers());
	end
		
end


@everywhere using PyPlot;
@everywhere using WriteVTK;
@everywhere using CPUTime;
@everywhere using DelimitedFiles;
@everywhere using Printf
@everywhere using BSON: @load
@everywhere using BSON: @save
@everywhere using SharedArrays;

include("primeObjects.jl");
include("thermo.jl"); #setup thermodynamics
include("utilsIO.jl");
include("RoeFlux2d.jl")
include("AUSMflux2d.jl"); #AUSM+ inviscid flux calculation 
include("utilsFVM2dp.jl"); #FVM utililities

include("init2d.jl");






function godunov2dthreads(pname::String, numThreads::Int64)

	
	#@load "testTriMesh2d.bson" testMesh
	#@load "testQuadMesh2d.bson" testMesh
	#@load "testMixedMesh2d.bson" testMesh
	
	@load pname testMesh ## load mesh
	
	
	cellsThreads = distibuteCellsInThreadsSA(numThreads, testMesh.nCells); ## partition mesh 
	

	include("setupSolver2DoblickShock2d.jl"); #setup FVM and numerical schemes
	
	
	## init primitive variables 
	println("set initial and boundary conditions ...");
	testfields2d = createFields2d(testMesh, thermo);
	
	
	
	
	## init conservative variables 
	## UconsCellsOld = zeros(Float64,testMesh.nCells,4);
	## UconsCellsNew = zeros(Float64,testMesh.nCells,4);
	##Delta = zeros(Float64,testMesh.nCells,4);
	
	
	UconsCellsOldX = SharedArray{Float64}(testMesh.nCells,4);
	UconsCellsNewX = SharedArray{Float64}(testMesh.nCells,4);
	DeltaX = SharedArray{Float64}(testMesh.nCells,4);
	iFLUX  = SharedArray{Float64}(testMesh.nCells,4);
	iFLUXdist  = SharedArray{Float64}(testMesh.nCells,4);
	
	n = size(testMesh.mesh_connectivity,2);
	mesh_connectivity = SharedArray{Float64}(testMesh.nCells,n);
	
	n = size(testMesh.cell_edges_length,2);
	cell_edges_length = SharedArray{Float64}(testMesh.nCells,n);
	
	n = size(testMesh.cell_edges_Nx,2);
	cell_edges_Nx = SharedArray{Float64}(testMesh.nCells,n);
	cell_edges_Ny = SharedArray{Float64}(testMesh.nCells,n);
	
	n = size(testMesh.cell_stiffness,2);
	cell_stiffness = SharedArray{Float64}(testMesh.nCells,n);
	
	n = size(testMesh.Z,2);
	Z = SharedArray{Float64}(testMesh.nCells,n);
	
	for i = 1:testMesh.nCells
	
		mesh_connectivity[i,:] = testMesh.mesh_connectivity[i,:];
		cell_edges_length[i,:] = testMesh.cell_edges_length[i,:];
		cell_edges_Nx[i,:] = testMesh.cell_edges_Nx[i,:];
		cell_edges_Ny[i,:] = testMesh.cell_edges_Ny[i,:];
		cell_stiffness[i,:] = testMesh.cell_stiffness[i,:];
		Z[i] = testMesh.Z[i];
	
	end
	
	# display(mesh_connectivity)
	# display(cell_edges_length)
	# display(cell_edges_Nx)
	# display(cell_edges_Ny)
	# display(Z)
	
	
	UphysCellsX = SharedArray{Float64}(testMesh.nCells,4);
	for i = 1:testMesh.nCells
		UphysCellsX[i,1] = testfields2d.densityCells[i];
		UphysCellsX[i,2] = testfields2d.UxCells[i];
		UphysCellsX[i,3] = testfields2d.UyCells[i];
		UphysCellsX[i,4] = testfields2d.pressureCells[i];
	end
	
	
	
	#UconsCellsOldX = phs2dcns2dcells(testfields2d, thermo.Gamma); #old vector
	#UconsCellsNewX = deepcopy(UconsCellsOldX); #new  vector 
	
	phs2dcns2dcellsSA(UconsCellsOldX,testfields2d, thermo.Gamma);	
	phs2dcns2dcellsSA(UconsCellsNewX,testfields2d, thermo.Gamma);	
		
	println("Start calculations ...");
	println(output.header);
	

	
	@everywhere testMeshX = $testMesh; 
	@everywhere thermoX   = $thermo;
	@everywhere cellsThreadsX = $cellsThreads;
	@everywhere testfields2dX  = $testfields2d;
		
	@everywhere dynControlsX = $dynControls;
	@everywhere solControlsX = $solControls;
	@everywhere pControlsX = $pControls;
	@everywhere outputX = $output;
	
	# @everywhere timeVector = [];
	# @everywhere residualsVector1 = []; 
	# @everywhere residualsVector2 = []; 
	# @everywhere residualsVector3 = []; 
	# @everywhere residualsVector4 = []; 
	# @everywhere residualsVectorMax = ones(Float64,4);
	# @everywhere convergenceCriteria= [1e-5;1e-5;1e-5;1e-5;];

	timeVector = [];
	residualsVector1 = []; 
	residualsVector2 = []; 
	residualsVector3 = []; 
	residualsVector4 = []; 
	residualsVectorMax = ones(Float64,4);
	convergenceCriteria= [1e-5;1e-5;1e-5;1e-5;];
	

	
	
		#for h = 1:2
		while (dynControlsX.isRunSimulation == 1)
			CPUtic();	
			
			
			# PROPAGATE STAGE: 
			(dynControlsX.velmax,id) = findmax(testfields2dX.VMAXCells);
			# #dynControls.tau = solControls.CFL * testMesh.maxEdgeLength/(max(dynControls.velmax,1.0e-6)); !!!!
			dynControlsX.tau = solControlsX.CFL * testMeshX.maxArea/(max(dynControlsX.velmax,1.0e-6));
		
				
			@sync @distributed for p in workers()	
	
				beginCell::Int64 = cellsThreadsX[p-1,1];
				endCell::Int64 = cellsThreadsX[p-1,2];
				#println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);
				# # println("worker: ",p,"\ttermoX.gamma: ", thermoX.Gamma);
				# # println("worker: ",p,"\ttestMeshX: ", testMeshX.nCells, "\t", testMeshX.nNodes );
				# # println("worker: ",p,"\ttestfields2dX: ", testfields2dX.densityCells[299] );
				
				#inviscidFlux(beginCell, endCell, 1.0, testMeshX, testfields2dX, thermoX, solControlsX, dynControlsX, UconsCellsNewX, UconsCellsOldX, iFLUX); 
				
				 inviscidFluxMM(beginCell, endCell, 1.0, 
						mesh_connectivity,cell_edges_length,cell_edges_Nx,cell_edges_Ny,cell_stiffness, Z, 
						thermoX.Gamma, solControlsX, dynControlsX, UconsCellsNewX, UconsCellsOldX, UphysCellsX, iFLUXdist); 

					 
					
			end
			
			#@everywhere finalize(inviscidFlux);			
					
			#inviscidFlux(1, testMeshX.nCells, 1.0, testMeshX, testfields2dX, thermoX, solControlsX, dynControlsX, iFLUX); 			
			#display(iFLUX-iFLUXdist)
			
			
			
			for i=1:testMeshX.nCells
			
				# DeltaX[i,1] = UconsCellsOldX[i,1]  - iFLUX[i,1];
				# DeltaX[i,2] = UconsCellsOldX[i,2]  - iFLUX[i,2];
				# DeltaX[i,3] = UconsCellsOldX[i,3]  - iFLUX[i,3];
				# DeltaX[i,4] = UconsCellsOldX[i,4]  - iFLUX[i,4];

				# DeltaX[i,1] = UconsCellsOldX[i,1]  - iFLUXdist[i,1];
				# DeltaX[i,2] = UconsCellsOldX[i,2]  - iFLUXdist[i,2];
				# DeltaX[i,3] = UconsCellsOldX[i,3]  - iFLUXdist[i,3];
				# DeltaX[i,4] = UconsCellsOldX[i,4]  - iFLUXdist[i,4];

							
				# UconsCellsNewX[i,1] = DeltaX[i,1];
				# UconsCellsNewX[i,2] = DeltaX[i,2];
				# UconsCellsNewX[i,3] = DeltaX[i,3];
				# UconsCellsNewX[i,4] = DeltaX[i,4];
				
				
				# UphysCellsX[i,1]  = UconsCellsNewX[i,1];
				# UphysCellsX[i,2]  = UconsCellsNewX[i,2]/UconsCellsNewX[i,1];
				# UphysCellsX[i,3]  = UconsCellsNewX[i,3]/UconsCellsNewX[i,1];
				# UphysCellsX[i,4]  = (thermoX.Gamma-1.0)*( UconsCellsNewX[i,4] - 0.5*( UconsCellsNewX[i,2]*UconsCellsNewX[i,2] + UconsCellsNewX[i,3]*UconsCellsNewX[i,3] )/UconsCellsNewX[i,1] );
	
				testfields2dX.densityCells[i] = UconsCellsNewX[i,1];
				testfields2dX.UxCells[i] 	  = UconsCellsNewX[i,2]/UconsCellsNewX[i,1];
				testfields2dX.UyCells[i] 	  = UconsCellsNewX[i,3]/UconsCellsNewX[i,1];
				testfields2dX.pressureCells[i] = (thermoX.Gamma-1.0)*( UconsCellsNewX[i,4] - 0.5*( UconsCellsNewX[i,2]*UconsCellsNewX[i,2] + UconsCellsNewX[i,3]*UconsCellsNewX[i,3] )/UconsCellsNewX[i,1] );

				testfields2dX.aSoundCells[i] = sqrt( thermoX.Gamma * testfields2dX.pressureCells[i]/testfields2dX.densityCells[i] );
				testfields2dX.VMAXCells[i]  = sqrt( testfields2dX.UxCells[i]*testfields2dX.UxCells[i] + testfields2dX.UyCells[i]*testfields2dX.UyCells[i] ) + testfields2dX.aSoundCells[i];
		
				DeltaX[i,1] = UconsCellsNewX[i,1] - UconsCellsOldX[i,1];
				DeltaX[i,2] = UconsCellsNewX[i,2] - UconsCellsOldX[i,2];
				DeltaX[i,3] = UconsCellsNewX[i,3] - UconsCellsOldX[i,3];
				DeltaX[i,4] = UconsCellsNewX[i,4] - UconsCellsOldX[i,4];

		
				UconsCellsOldX[i,1] = UconsCellsNewX[i,1];
				UconsCellsOldX[i,2] = UconsCellsNewX[i,2];
				UconsCellsOldX[i,3] = UconsCellsNewX[i,3];
				UconsCellsOldX[i,4] = UconsCellsNewX[i,4];
				
				UphysCellsX[i,1] = testfields2dX.densityCells[i];
				UphysCellsX[i,2] = testfields2dX.UxCells[i];
				UphysCellsX[i,3] = testfields2dX.UyCells[i];
				UphysCellsX[i,4] = testfields2dX.pressureCells[i];
				
		
			end
	
			#UconsCellsOldX = deepcopy(UconsCellsNewX);
			
			# display(UconsCellsNewX)
			# display(UconsCellsOldX)
			# display(iFLUX)

		
	
			cells2nodesSolutionReconstructionWithStencilsImplicit!(testMeshX, testfields2dX); 
			
		
	
			(dynControlsX.rhoMax,id) = findmax(testfields2dX.densityCells);
			(dynControlsX.rhoMin,id) = findmin(testfields2dX.densityCells);
			

			push!(timeVector, dynControlsX.flowTime); 
			dynControlsX.curIter += 1; 
			dynControlsX.verIter += 1;
			
			
			updateResidual(DeltaX, 
				residualsVector1,residualsVector2,residualsVector3,residualsVector4, residualsVectorMax,  
				convergenceCriteria, dynControlsX);
			
			updateOutput(timeVector,residualsVector1,residualsVector2,residualsVector3,residualsVector4, residualsVectorMax, 
				testMeshX, testfields2dX, solControlsX, outputX, dynControlsX);
	
			
						# EVALUATE STAGE:
			if (solControlsX.timeStepMethod == 1)
				dynControlsX.flowTime += dynControlsX.tau;  	
			else
				dynControlsX.flowTime += solControlsX.dt;  
			end
			

	

			if (flowTime>= solControlsX.stopTime || dynControlsX.isSolutionConverged == 1)
				dynControlsX.isRunSimulation = 0;
		
				if (dynControlsX.isSolutionConverged == true)
					println("Solution converged! ");
				else
					println("Simultaion flow time reached the set Time!");
				end
			
				if (outputX.saveResiduals == 1)
					#println("Saving Residuals ... ");
					#cd(dynControlsX.localTestPath);
					#saveResiduals(output.fileNameResiduals, timeVector, residualsVector1, residualsVector2, residualsVector3, residualsVector4);
					#cd(dynControlsX.globalPath);
				end
				if (outputX.saveResults == 1)
					#println("Saving Results ... ");
					#cd(dynControlsX.localTestPath);
					#saveSolution(output.fileNameResults, testMeshX.xNodes, testMeshX.yNodes, UphysNodes);
					#cd(dynControlsX.globalPath);
				end
			
			end

			dynControlsX.cpuTime  += CPUtoq(); 
			
			if (dynControlsX.flowTime >= solControls.stopTime)
				dynControlsX.isRunSimulation = 0;
			end
			
		end ## end while
		 
	
	
end



@time godunov2dthreads("testTriMesh2d.bson",4); 


#@time godunov2d("testQuadMesh2d.bson"); 
#@time godunov2d("testMixedMesh2d.bson"); 





