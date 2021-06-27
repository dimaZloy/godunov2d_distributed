

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

include("D:/projects/mesh2d/primeObjects.jl");
include("thermo.jl"); #setup thermodynamics
include("utilsIO.jl");
include("RoeFlux2d.jl")
include("AUSMflux2d.jl"); #AUSM+ inviscid flux calculation 
include("utilsFVM2dp.jl"); #FVM utililities

include("initfields2d.jl");









function godunov2dthreads(pname::String, numThreads::Int64)

	
	#@load "testTriMesh2d.bson" testMesh
	#@load "testQuadMesh2d.bson" testMesh
	#@load "testMixedMesh2d.bson" testMesh
	
	@load pname testMesh ## load mesh
	
	
	cellsThreads = distibuteCellsInThreadsSA(numThreads, testMesh.nCells); ## partition mesh 
	

	include("setupSolver2DoblickShock2d.jl"); #setup FVM and numerical schemes
	
	
	## init primitive variables 
	println("set initial and boundary conditions ...");
	testfields2d = createFields2d_shared(testMesh, thermo);
	
	println("nCells:\t", testMesh.nCells);
	println("nNodes:\t", testMesh.nNodes);
	
	## init conservative variables 
	
	
	UconsCellsOldX = SharedArray{Float64}(testMesh.nCells,4);
	UconsCellsNewX = SharedArray{Float64}(testMesh.nCells,4);
	DeltaX = SharedArray{Float64}(testMesh.nCells,4);
	iFLUX  = SharedArray{Float64}(testMesh.nCells,4);
	iFLUXdist  = SharedArray{Float64}(testMesh.nCells,4);
	
	n = size(testMesh.mesh_connectivity,2);
	mesh_connectivity = SharedArray{Int64}(testMesh.nCells,n);
	
	n = size(testMesh.cell_edges_length,2);
	cell_edges_length = SharedArray{Float64}(testMesh.nCells,n);
	
	n = size(testMesh.cell_edges_Nx,2);
	cell_edges_Nx = SharedArray{Float64}(testMesh.nCells,n);
	cell_edges_Ny = SharedArray{Float64}(testMesh.nCells,n);
	
	n = size(testMesh.cell_stiffness,2);
	cell_stiffness = SharedArray{Int64}(testMesh.nCells,n);
	
	#n = size(testMesh.Z,2);
	Z = SharedArray{Float64}(testMesh.nCells);
	
	for i = 1:testMesh.nCells
	
		mesh_connectivity[i,:] = testMesh.mesh_connectivity[i,:];
		cell_edges_length[i,:] = testMesh.cell_edges_length[i,:];
		cell_edges_Nx[i,:] = testMesh.cell_edges_Nx[i,:];
		cell_edges_Ny[i,:] = testMesh.cell_edges_Ny[i,:];
		cell_stiffness[i,:] = testMesh.cell_stiffness[i,:];
		Z[i] = testMesh.Z[i];
	
	end
	
	testMeshDistr = mesh2d_shared(
		mesh_connectivity,
		Z,
		cell_edges_Nx,
		cell_edges_Ny,
		cell_edges_length,
		cell_stiffness);
	
	# display(mesh_connectivity)
	# display(cell_edges_length)
	# display(cell_edges_Nx)
	# display(cell_edges_Ny)
	# display(Z)
	
	
	phs2dcns2dcellsSA(UconsCellsOldX,testfields2d, thermo.Gamma);	
	phs2dcns2dcellsSA(UconsCellsNewX,testfields2d, thermo.Gamma);	
		
	println("Start calculations ...");
	println(output.header);
	

	@everywhere testMeshDistrX = $testMeshDistr; 
	@everywhere testMeshX = $testMesh; 
	@everywhere thermoX   = $thermo;
	@everywhere cellsThreadsX = $cellsThreads;
	@everywhere testfields2dX  = $testfields2d;
		
	@everywhere dynControlsX = $dynControls;
	@everywhere solControlsX = $solControls;
	@everywhere pControlsX = $pControls;
	@everywhere outputX = $output;
	


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
				
										
				# inviscidFluxMM22(beginCell, endCell, 1.0, 
						# mesh_connectivity,cell_edges_length,cell_edges_Nx,cell_edges_Ny,cell_stiffness, Z, testfields2dX,  
						# thermoX.Gamma, solControlsX, dynControlsX, UconsCellsNewX, UconsCellsOldX, iFLUXdist); 

				inviscidFluxMM44(beginCell, endCell, 1.0,  testMeshDistrX, testfields2dX,  
						thermoX.Gamma, solControlsX, dynControlsX, UconsCellsNewX, UconsCellsOldX, iFLUXdist); 
					
			end
			
			#@everywhere finalize(inviscidFluxMM22);		
			@everywhere finalize(inviscidFluxMM44);				
			
			
			@sync @distributed for p in workers()	
	
				beginCell::Int64 = cellsThreadsX[p-1,1];
				endCell::Int64 = cellsThreadsX[p-1,2];
				#println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);
														
				updateVariablesMM22(beginCell, endCell, thermoX.Gamma,  UconsCellsNewX, UconsCellsOldX, DeltaX, testfields2dX);
		
			end
			
			@everywhere finalize(updateVariablesMM22);	
					
			#inviscidFlux(1, testMeshX.nCells, 1.0, testMeshX, testfields2dX, thermoX, solControlsX, dynControlsX, iFLUX); 			
			#display(iFLUX-iFLUXdist)
			
		
			cells2nodesSolutionReconstructionWithStencilsImplicitSA!(testMeshX, testfields2dX); 
			
	
			(dynControlsX.rhoMax,id) = findmax(testfields2dX.densityCells);
			(dynControlsX.rhoMin,id) = findmin(testfields2dX.densityCells);
			

			push!(timeVector, dynControlsX.flowTime); 
			dynControlsX.curIter += 1; 
			dynControlsX.verIter += 1;
			
			
			updateResidual(DeltaX, 
				residualsVector1,residualsVector2,residualsVector3,residualsVector4, residualsVectorMax,  
				convergenceCriteria, dynControlsX);
			
			updateOutputSA(timeVector,residualsVector1,residualsVector2,residualsVector3,residualsVector4, residualsVectorMax, 
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


#@time godunov2dthreads("testMixedMesh2d.bson",4); 
@time godunov2dthreads("testTriMesh2d.bson",4); 
#@time godunov2dthreads("testQuadMesh2d.bson",4); 


#@time godunov2d("testQuadMesh2d.bson"); 
#@time godunov2d("testMixedMesh2d.bson"); 





