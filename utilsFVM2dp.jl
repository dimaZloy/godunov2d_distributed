
# utilities for FVM 

@everywhere @inline function phs2dcns2dcells(testFields::fields2d, gamma::Float64)::Array{Float64,2}

	N::Int64 = size(testFields.densityCells,1);
	
	ACons = zeros(Float64,N,4);

	for i = 1:N
		ACons[i,1] = testFields.densityCells[i];
		ACons[i,2] = testFields.densityCells[i]*testFields.UxCells[i];
		ACons[i,3] = testFields.densityCells[i]*testFields.UyCells[i];
		ACons[i,4] = testFields.pressureCells[i]/(gamma-1.0) + 0.5*testFields.densityCells[i]*(	testFields.UxCells[i]*testFields.UxCells[i] +  testFields.UyCells[i]*testFields.UyCells[i] );

	end #for
	
	return ACons;
end

@everywhere @inline function phs2dcns2dcellsSA(ACons::SharedArray{Float64,2}, testFields::fields2d, gamma::Float64)

	N::Int64 = size(testFields.densityCells,1);
	
	#ACons = zeros(Float64,N,4);

	for i = 1:N
		ACons[i,1] = testFields.densityCells[i];
		ACons[i,2] = testFields.densityCells[i]*testFields.UxCells[i];
		ACons[i,3] = testFields.densityCells[i]*testFields.UyCells[i];
		ACons[i,4] = testFields.pressureCells[i]/(gamma-1.0) + 0.5*testFields.densityCells[i]*(	testFields.UxCells[i]*testFields.UxCells[i] +  testFields.UyCells[i]*testFields.UyCells[i] );

	end #for
	
	#return ACons;
end

@everywhere @inline function phs2dcns2dcellsSA(
	ACons::SharedArray{Float64,2}, testFields::fields2d_shared, gamma::Float64)

	N::Int64 = size(testFields.densityCells,1);
	
	#ACons = zeros(Float64,N,4);

	for i = 1:N
		ACons[i,1] = testFields.densityCells[i];
		ACons[i,2] = testFields.densityCells[i]*testFields.UxCells[i];
		ACons[i,3] = testFields.densityCells[i]*testFields.UyCells[i];
		ACons[i,4] = testFields.pressureCells[i]/(gamma-1.0) + 0.5*testFields.densityCells[i]*(	testFields.UxCells[i]*testFields.UxCells[i] +  testFields.UyCells[i]*testFields.UyCells[i] );

	end #for
	
	#return ACons;
end



# @everywhere function calculateVariablesAtStage(bettaKJ::Float64, UConsOld::Array{Float64,2}, UphysCells::Array{Float64,2}, UphysNodes::Array{Float64,2})

	# global solver;
	# #global thermo; 

	# #global nThreads;
	# #global nThreadsMax;
	
    # if (solver.SpatialDiscretization == 1)
    	# #return FirstOrderUpwind(bettaKJ,UConsOld);	
		# return FirstOrderUpwindM1(bettaKJ,UconsCellsOld, UphysCells );
    # elseif (solver.SpatialDiscretization == 2)
		# println("SOU is not implemented  ...   ");
		# println("Using default First order upwind scheme  ...   ");
		# return FirstOrderUpwindM1(bettaKJ,UconsCellsOld, UphysCells );
    	# #return SecondOrderUpwind(bettaKJ,UConsOld);				
		# #SecondOrderUpwind(1.0,UconsCellsOld, UphysCells, UphysNodes); 
    # else
    	# println("Spatial discretization scheme is not set ...   ");
    	# println("Using default First order upwind scheme  ...   ");
    	# #return FirstOrderUpwind(bettaKJ,UConsOld);	
		# return FirstOrderUpwindM1( bettaKJ,UconsCellsOld, UphysCells );
    # end	
	

# end



	

@everywhere function ComputeUPhysFromBoundaries(i,k,neib_cell, cur_cell, nx,ny)

	#global TEST;	
	bnd_cell = zeros(Float64,4);

	

		####bc_data = [bctop; bcright; bcbottom; bcleft;];

		if (neib_cell == -1) #top 

            bnd_cell[1] = 1.7;
            bnd_cell[2] = 263.72;
            bnd_cell[3] = -51.62;
            bnd_cell[4] = 15282.0;

		
		elseif (neib_cell == -2) #right 

			bnd_cell = cur_cell;	
			

		elseif (neib_cell == -3) #bottom 

	 	   	#bnd_cell = cur_cell;	
	        #bnd_cell = updateVelocityFromCurvWall(i,k,bnd_cell,nx,ny);
			bnd_cell = updateVelocityFromCurvWall(i,k,cur_cell,nx,ny);

        
		elseif (neib_cell == -4) # left boundary 

            bnd_cell[1] = 1.0;
            bnd_cell[2] = 290.0;
            bnd_cell[3] = 0.0;
            bnd_cell[4] = 7143;

					
		end	

			

	return bnd_cell; 
end


@everywhere function updateVelocityFromCurvWall(i::Int64, k::Int64, U, nx::Float64, ny::Float64)

# High-Order Accurate Implementation of Solid Wall Boundary Conditions in Curved Geometries, 
# Lilia Krivodonova and Marsha Berger, Courant Institute of Mathematical Sciences, New York, NY 10012

# a = U[1]*(ny*ny - nx*nx) - 2.0*nx*ny*U[2];
# b = U[2]*(nx*nx - ny*ny) - 2.0*nx*ny*U[1];

# U[1] = a;
# U[2] = b;

	Un = U; 

        Un[2] = U[2]*(ny*ny - nx*nx) - 2.0*nx*ny*U[3];
        Un[3] = U[3]*(nx*nx - ny*ny) - 2.0*nx*ny*U[2];


	return Un;	
end


@everywhere function cells2nodesSolutionReconstructionWithStencilsImplicit!(testMesh::mesh2d,testFields::fields2d)

	node_solution = zeros(Float64,testMesh.nNodes,4); 
	
	for J=1:testMesh.nNodes
	
		det::Float64 = 0.0;
		
		for j = 1:testMesh.nNeibCells
		
			neibCell::Int64 = testMesh.cell_clusters[J,j]; 
			
			if (neibCell !=0)
				 wi::Float64 = testMesh.node_stencils[J,j];
				 #node_solution[J,:] += cell_solution[neibCell,:];
				 node_solution[J,1] += testFields.densityCells[neibCell]*wi;
				 node_solution[J,2] += testFields.UxCells[neibCell]*wi;
				 node_solution[J,3] += testFields.UyCells[neibCell]*wi;
				 node_solution[J,4] += testFields.pressureCells[neibCell]*wi;
				 
				 det += wi;
			end
		end
		if (det!=0)
			node_solution[J,:] = node_solution[J,:]/det; 
			
		end
	end

	
	for J=1:testMesh.nNodes
	
		testFields.densityNodes[J] = node_solution[J,1]; 
		testFields.UxNodes[J] 	   = node_solution[J,2]; 
		testFields.UyNodes[J] 	   = node_solution[J,3]; 
		testFields.pressureNodes[J] =  node_solution[J,4]; 
		
	end

end

@everywhere function cells2nodesSolutionReconstructionWithStencilsImplicitSA!(
		testMesh::mesh2d,testFields::fields2d_shared)	

	node_solution = zeros(Float64,testMesh.nNodes,4); 
	
	for J=1:testMesh.nNodes
	
		det::Float64 = 0.0;
		
		for j = 1:testMesh.nNeibCells
		
			neibCell::Int64 = testMesh.cell_clusters[J,j]; 
			
			if (neibCell !=0)
				 wi::Float64 = testMesh.node_stencils[J,j];
				 #node_solution[J,:] += cell_solution[neibCell,:];
				 node_solution[J,1] += testFields.densityCells[neibCell]*wi;
				 node_solution[J,2] += testFields.UxCells[neibCell]*wi;
				 node_solution[J,3] += testFields.UyCells[neibCell]*wi;
				 node_solution[J,4] += testFields.pressureCells[neibCell]*wi;
				 
				 det += wi;
			end
		end
		if (det!=0)
			node_solution[J,:] = node_solution[J,:]/det; 
			
		end
	end

	
	for J=1:testMesh.nNodes
	
		testFields.densityNodes[J] = node_solution[J,1]; 
		testFields.UxNodes[J] 	   = node_solution[J,2]; 
		testFields.UyNodes[J] 	   = node_solution[J,3]; 
		testFields.pressureNodes[J] =  node_solution[J,4]; 
		
	end

end




@everywhere  @inline function cells2nodesSolutionReconstructionWithStencils(testMesh::mesh2d,cell_solution::Array{Float64,1})::Array{Float64,1}

node_solution = zeros(Float64,testMesh.nNodes); 

for J=1:testMesh.nNodes
	det::Float64 = 0.0;
	for j = 1:testMesh.nNeibCells
		neibCell::Int64 = testMesh.cell_clusters[J,j]; 
		if (neibCell !=0)
			wi::Float64 = testMesh.node_stencils[J,j];
			node_solution[J] += cell_solution[neibCell]*wi;
			det += wi;
		end
	end
	if (det!=0)
		node_solution[J] = node_solution[J]/det; 
	end
end

return node_solution;	

end


@everywhere  @inline function cells2nodesSolutionReconstructionWithStencilsSA(
		testMesh::mesh2d,cell_solution::SharedArray{Float64,1} ) ::Array{Float64,1}

node_solution = zeros(Float64,testMesh.nNodes); 

for J=1:testMesh.nNodes
	det::Float64 = 0.0;
	for j = 1:testMesh.nNeibCells
		neibCell::Int64 = testMesh.cell_clusters[J,j]; 
		if (neibCell !=0)
			wi::Float64 = testMesh.node_stencils[J,j];
			node_solution[J] += cell_solution[neibCell]*wi;
			det += wi;
		end
	end
	if (det!=0)
		node_solution[J] = node_solution[J]/det; 
	end
end

return node_solution;	

end


# function cells2nodesSolutionReconstructionWithStencils(testMesh::mesh2d,cell_solution::Array{Float64,2})::Array{Float64,2}

# node_solution = zeros(Float64,testMesh.nNodes,4); 

# for J=1:testMesh.nNodes
	# det::Float64 = 0.0;
	# for j = 1:testMesh.nNeibCells
		# neibCell::Int64 = testMesh.cell_clusters[J,j]; 
		# if (neibCell !=0)
			# wi::Float64 = testMesh.node_stencils[J,j];
			# node_solution[J,:] += cell_solution[neibCell,:];
			# det += wi;
		# end
	# end
	# if (det!=0)
		# node_solution[J,:] = node_solution[J,:]/det; 
	# end
# end

# return node_solution;	

# end





@everywhere function inviscidFlux( beginCell::Int64,endCell::Int64, bettaKJ::Float64, testMesh::mesh2d, testFields::fields2d, 
	thermo::THERMOPHYSICS, solControls::CONTROLS, dynControls::DYNAMICCONTROLS,
	UconsCellsNew::SharedArray{Float64,2},UconsCellsOld::SharedArray{Float64,2}, fluxX::SharedArray{Float64,2})


	#UconsCellsNew = SharedArray{Float64}(testMesh.nCells,4);
	
	uLeftp = zeros(Float64,4);
	uRightp = zeros(Float64,4);	

	uConsLeftp = zeros(Float64,4);
	uConsRightp = zeros(Float64,4);	

	
	for i = beginCell:endCell
	
		ck::Int64 = testMesh.mesh_connectivity[i,3]; 

		uLeftp[1] = testFields.densityCells[i];
		uLeftp[2] = testFields.UxCells[i];
		uLeftp[3] = testFields.UyCells[i];
		uLeftp[4] = testFields.pressureCells[i];
		
		uRightp[1] = testFields.densityCells[i];
		uRightp[2] = testFields.UxCells[i];	
		uRightp[3] = testFields.UyCells[i];	
		uRightp[4] = testFields.pressureCells[i];
		
		FLUXES = zeros(Float64,4);
		#FLUXES_TMP = zeros(Float64,4);
		
			for k =1:ck
					
				side::Float64 = testMesh.cell_edges_length[i,k];
				nx::Float64   = testMesh.cell_edges_Nx[i,k];
				ny::Float64   = testMesh.cell_edges_Ny[i,k];

				ek::Int64 = testMesh.cell_stiffness[i,k]; 
				edge_flux = zeros(Float64,4);
				#edge_fluxTmp = zeros(Float64,4);


				if (ek>=1 && ek<=testMesh.nCells ) 
					
					uRightp[1] = testFields.densityCells[ek];
					uRightp[2] = testFields.UxCells[ek];
					uRightp[3] = testFields.UyCells[ek];
					uRightp[4] = testFields.pressureCells[ek];
					
					
				else
					uRightp = ComputeUPhysFromBoundaries(i,k, ek, uRightp, nx,ny);
				
					#uRightp = ComputeUPhysFromBoundaries(i,k, ek, uLeftp, nx,ny);
				end 
				
			
				
				edge_flux = RoeFlux2d(uRightp,uLeftp, nx,ny,side,thermo.Gamma);
				#edge_flux = AUSMplusFlux2d(uRightp,uLeftp,nx,ny,side,thermo.Gamma);			
				#edge_flux = RiemannFlux2d(uRightp,uLeftp,nx,ny,side,thermo.Gamma); ## to be test!!!			
				
			

				FLUXES[1] = FLUXES[1] + edge_flux[1];
				FLUXES[2] = FLUXES[2] + edge_flux[2];
				FLUXES[3] = FLUXES[3] + edge_flux[3];
				FLUXES[4] = FLUXES[4] + edge_flux[4];
				
				
			end # K for neib cells 
			
			#display(FLUXES)
			
		
			if (solControls.timeStepMethod == 1)
			
				fluxX[i,1] = FLUXES[1]*bettaKJ*testMesh.Z[i]*dynControls.tau;
				fluxX[i,2] = FLUXES[2]*bettaKJ*testMesh.Z[i]*dynControls.tau;
				fluxX[i,3] = FLUXES[3]*bettaKJ*testMesh.Z[i]*dynControls.tau;
				fluxX[i,4] = FLUXES[4]*bettaKJ*testMesh.Z[i]*dynControls.tau;
			
			
			else
				fluxX[i,1] = FLUXES[1]*bettaKJ*testMesh.Z[i]*solControls.dt;
				fluxX[i,2] = FLUXES[2]*bettaKJ*testMesh.Z[i]*solControls.dt;
				fluxX[i,3] = FLUXES[3]*bettaKJ*testMesh.Z[i]*solControls.dt;
				fluxX[i,4] = FLUXES[4]*bettaKJ*testMesh.Z[i]*solControls.dt;
			end
			
			UconsCellsNew[i,1] = UconsCellsOld[i,1] - fluxX[i,1];
			UconsCellsNew[i,2] = UconsCellsOld[i,2] - fluxX[i,2];
			UconsCellsNew[i,3] = UconsCellsOld[i,3] - fluxX[i,3];
			UconsCellsNew[i,4] = UconsCellsOld[i,4] - fluxX[i,4];

		
	end	#  i for loop cells 

	
	#display(UconsCellsNew)
	
	#return UconsCellsNew[beginCell:endCell,:];

end



@everywhere function inviscidFluxMM22( beginCell::Int64,endCell::Int64, bettaKJ::Float64, 
			mesh_connectivity::SharedArray{Float64,2},cell_edges_length::SharedArray{Float64,2},cell_edges_Nx::SharedArray{Float64,2},cell_edges_Ny::SharedArray{Float64,2},
			cell_stiffness::SharedArray{Float64,2}, Z::SharedArray{Float64,2}, testFields::fields2d_shared, 
			gamma::Float64, solControls::CONTROLS, dynControls::DYNAMICCONTROLS, 
			UconsCellsNew::SharedArray{Float64,2}, UconsCellsOld::SharedArray{Float64,2},  fluxX::SharedArray{Float64,2})


	
	uLeftp = zeros(Float64,4);
	uRightp = zeros(Float64,4);	

	uConsLeftp = zeros(Float64,4);
	uConsRightp = zeros(Float64,4);	

	N = size(fluxX,1);
	
	for i = beginCell:endCell
	
		ck::Int64 = mesh_connectivity[i,3]; 

		uLeftp[1] = testFields.densityCells[i];
		uLeftp[2] = testFields.UxCells[i];
		uLeftp[3] = testFields.UyCells[i];
		uLeftp[4] = testFields.pressureCells[i];
		
		uRightp[1] = testFields.densityCells[i];
		uRightp[2] = testFields.UxCells[i];	
		uRightp[3] = testFields.UyCells[i];	
		uRightp[4] = testFields.pressureCells[i];
		
		FLUXES = zeros(Float64,4);
		
		
			for k =1:ck
					
				side::Float64 = cell_edges_length[i,k];
				nx::Float64   = cell_edges_Nx[i,k];
				ny::Float64   = cell_edges_Ny[i,k];

				ek::Int64 =  cell_stiffness[i,k]; 
				edge_flux = zeros(Float64,4);
		

				if (ek>=1 && ek<=N ) 
					
					uRightp[1] = testFields.densityCells[ek];
					uRightp[2] = testFields.UxCells[ek];
					uRightp[3] = testFields.UyCells[ek];
					uRightp[4] = testFields.pressureCells[ek];
					
					
				else
					uRightp = ComputeUPhysFromBoundaries(i,k, ek, uRightp, nx,ny);
				
					#uRightp = ComputeUPhysFromBoundaries(i,k, ek, uLeftp, nx,ny);
				end 
				
			
				
				edge_flux = RoeFlux2d(uRightp,uLeftp, nx,ny,side,gamma);
				#edge_flux = AUSMplusFlux2d(uRightp,uLeftp,nx,ny,side,thermo.Gamma);			
				#edge_flux = RiemannFlux2d(uRightp,uLeftp,nx,ny,side,thermo.Gamma); ## to be test!!!			
				
			

				FLUXES[1] = FLUXES[1] + edge_flux[1];
				FLUXES[2] = FLUXES[2] + edge_flux[2];
				FLUXES[3] = FLUXES[3] + edge_flux[3];
				FLUXES[4] = FLUXES[4] + edge_flux[4];
				
				#FLUXES += edge_flux;
				
			end # K for neib cells 
			
			#display(FLUXES)
		
			if (solControls.timeStepMethod == 1)
			
				fluxX[i,1] = FLUXES[1]*bettaKJ*Z[i]*dynControls.tau;
				fluxX[i,2] = FLUXES[2]*bettaKJ*Z[i]*dynControls.tau;
				fluxX[i,3] = FLUXES[3]*bettaKJ*Z[i]*dynControls.tau;
				fluxX[i,4] = FLUXES[4]*bettaKJ*Z[i]*dynControls.tau;
			
			
			else
				fluxX[i,1] = FLUXES[1]*bettaKJ*Z[i]*solControls.dt;
				fluxX[i,2] = FLUXES[2]*bettaKJ*Z[i]*solControls.dt;
				fluxX[i,3] = FLUXES[3]*bettaKJ*Z[i]*solControls.dt;
				fluxX[i,4] = FLUXES[4]*bettaKJ*Z[i]*solControls.dt;
			end
			
			UconsCellsNew[i,1] = UconsCellsOld[i,1] - fluxX[i,1];
			UconsCellsNew[i,2] = UconsCellsOld[i,2] - fluxX[i,2];
			UconsCellsNew[i,3] = UconsCellsOld[i,3] - fluxX[i,3];
			UconsCellsNew[i,4] = UconsCellsOld[i,4] - fluxX[i,4];

		
	end	#  i for loop cells 


end


