

@everywhere function distibuteCellsInThreadsSA(nThreads::Int64, nCells::Int64 )::SharedArray{Int64,2}

	cellsThreads = SharedArray{Int64}(nThreads,2);
 
 	if (nThreads>1)
		
  		#cellsThreads = zeros(Int64,nThreads,2);
		#cellsThreads = SharedArray{Int64}(nThreads,2);

    	#cout << "nThreads: " <<  nThreads << endl;
    	#cout << "nCells: " <<  get_num_cells() << endl;
    	nParts = floor(nCells/nThreads);
    	#cout << "nParts: " <<  nParts << endl;


	    for i=1:nThreads
    		cellsThreads[i,2] =  nCells - nParts*(nThreads-i );
		end
	

    	for i=1:nThreads
      		cellsThreads[i,1] =  cellsThreads[i,2] - nParts + 1;
		end

    	cellsThreads[1,1] = 1;

		#display(cellsThreads);	
		
	end
	
	return cellsThreads;

end



# function distibuteCellsInThreads(nThreads::Int64, nCells::Int64 )

 
 	# if (nThreads>1)
		
  		# cellsThreads = zeros(Int64,nThreads,2);
    	# nParts = floor(nCells/nThreads);


	    # for i=1:nThreads
    		# cellsThreads[i,2] =  nCells - nParts*(nThreads-i );
		# end
	

    	# for i=1:nThreads
      		# cellsThreads[i,1] =  cellsThreads[i,2] - nParts + 1;
		# end

    	# cellsThreads[1,1] = 1;
		
		# return cellsThreads;
		
	# else				
		# return 0;			
	# end

# end


@everywhere function createFields2d(testMesh::mesh2d, thermo::THERMOPHYSICS)


	##display("set initial and boundary conditions ...");

	phsLeftBC = zeros(Float64,4);
	phsTopBC = zeros(Float64,4);

	phsLeftBC[1] = 1.0;
	phsLeftBC[2] = 290.0;
	phsLeftBC[3] = 0.0;
	phsLeftBC[4] = 7143.0;

	phsTopBC[1] = 1.7;
	phsTopBC[2] = 263.72;
	phsTopBC[3] = -51.62;
	phsTopBC[4] = 15282.0;


	densityCells = zeros(Float64, testMesh.nCells); 
	UxCells = zeros(Float64, testMesh.nCells); 
	UyCells = zeros(Float64, testMesh.nCells); 
	pressureCells = zeros(Float64, testMesh.nCells); 
	aSoundCells = zeros(Float64, testMesh.nCells); #speed of sound
	VMAXCells = zeros(Float64, testMesh.nCells); #max speed in domain

	#entropyCell = zeros(Float64, testMesh.nCells); #entropy

	densityNodes = zeros(Float64, testMesh.nNodes); 
	UxNodes = zeros(Float64, testMesh.nNodes); 
	UyNodes = zeros(Float64, testMesh.nNodes); 
	pressureNodes = zeros(Float64, testMesh.nNodes); 

	for i=1:testMesh.nCells

		densityCells[i] = phsLeftBC[1];
		UxCells[i] = phsLeftBC[2];
		UyCells[i] = phsLeftBC[3];
		pressureCells[i] = phsLeftBC[4];
			
		aSoundCells[i] = sqrt( thermo.Gamma * pressureCells[i]/densityCells[i] );
		VMAXCells[i]  = sqrt( UxCells[i]*UxCells[i] + UyCells[i]*UyCells[i] ) + aSoundCells[i];
		#entropyCell[i] = UphysCells[i,1]/(thermo.Gamma-1.0)*log(UphysCells[i,4]/UphysCells[i,1]*thermo.Gamma);
				
	end

	
	densityF = cells2nodesSolutionReconstructionWithStencils(testMesh, densityCells); 

		
	if (output.saveDataToVTK == 1)	
		filename = string("zzz",dynControls.curIter+1000);
		saveResults2VTK(filename, testMesh, densityF, "density");
		
	end

		


	# create fields 
	testFields2d = fields2d(
		densityCells,
		UxCells,
		UyCells,
		pressureCells,
		aSoundCells,
		VMAXCells,
		densityNodes,
		UxNodes,
		UyNodes,
		pressureNodes
		#UconsCellsOld,
		#UconsCellsNew
	);

	return testFields2d; 


end

@everywhere function updateResidual(
	Delta::SharedArray{Float64,2},
	#Delta::Array{Float64,2},
	residualsVector1::Array{Any,1},
	residualsVector2::Array{Any,1},
	residualsVector3::Array{Any,1},
	residualsVector4::Array{Any,1},
	residualsVectorMax::Array{Float64,1},
	convergenceCriteria::Array{Float64,1},
	dynControls::DYNAMICCONTROLS,
	)

	residuals1::Float64   =sum( Delta[:,1].*Delta[:,1] );
	residuals2::Float64  = sum( Delta[:,2].*Delta[:,2] );
	residuals3::Float64  = sum( Delta[:,3].*Delta[:,3] );
	residuals4::Float64  = sum( Delta[:,4].*Delta[:,4] );
	
	push!(residualsVector1, residuals1);
	push!(residualsVector2, residuals2);
	push!(residualsVector3, residuals3);
	push!(residualsVector4, residuals4);
	
	if (dynControls.curIter<6 && dynControls.curIter>1)

   		(residualsVectorMax[1],id1) = findmax(residualsVector1[1:dynControls.curIter]);	
   		(residualsVectorMax[2],id2) = findmax(residualsVector2[1:dynControls.curIter]);		
   		(residualsVectorMax[3],id3) = findmax(residualsVector3[1:dynControls.curIter]);		
   		(residualsVectorMax[4],id4) = findmax(residualsVector4[1:dynControls.curIter]);		

	end

	if ( (dynControls.curIter>5) && 
    	(residualsVector1[dynControls.curIter]./residualsVectorMax[1] <= convergenceCriteria[1]) &&
     	(residualsVector2[dynControls.curIter]./residualsVectorMax[2] <= convergenceCriteria[2]) &&
     	(residualsVector3[dynControls.curIter]./residualsVectorMax[3] <= convergenceCriteria[3]) &&
     	(residualsVector4[dynControls.curIter]./residualsVectorMax[4] <= convergenceCriteria[4]) )

	 	dynControls.isSolutionConverged  = 1; 

	end



end


# DEPRICATED!!!
# @everywhere function updateVariablesSA(
	# UconsCellsNew::SharedArray{Float64,2},
	# UconsCellsOld::SharedArray{Float64,2},
	# Delta::SharedArray{Float64,2},
	# testMesh::mesh2d,
	# testfields2d::fields2d,
	# dynControls::DYNAMICCONTROLS)
	
	# for i=1:testMesh.nCells
	
	
		# testfields2d.densityCells[i] = UconsCellsNew[i,1];
		# testfields2d.UxCells[i] 	 = UconsCellsNew[i,2]/UconsCellsNew[i,1];
		# testfields2d.UyCells[i] 	 = UconsCellsNew[i,3]/UconsCellsNew[i,1];
		# testfields2d.pressureCells[i] = (thermo.Gamma-1.0)*( UconsCellsNew[i,4] - 0.5*( UconsCellsNew[i,2]*UconsCellsNew[i,2] + UconsCellsNew[i,3]*UconsCellsNew[i,3] )/UconsCellsNew[i,1] );

		# testfields2d.aSoundCells[i] = sqrt( thermo.Gamma * testfields2d.pressureCells[i]/testfields2d.densityCells[i] );
		# testfields2d.VMAXCells[i]  = sqrt( testfields2d.UxCells[i]*testfields2d.UxCells[i] + testfields2d.UyCells[i]*testfields2d.UyCells[i] ) + testfields2d.aSoundCells[i];
		
		
	# end
	
	
	# cells2nodesSolutionReconstructionWithStencilsImplicit!(testMesh, testfields2d); 
	
	# (dynControls.rhoMax,id) = findmax(testfields2d.densityCells);
	# (dynControls.rhoMin,id) = findmin(testfields2d.densityCells);

# end



@everywhere function updateVariablesMM22(
	beginCell::Int64,endCell::Int64,Gamma::Float64,
	 UconsCellsNew::SharedArray{Float64,2},
	 UconsCellsOld::SharedArray{Float64,2},
	 Delta::SharedArray{Float64,2},
	 testfields2d::fields2d_shared)
	
	for i=beginCell:endCell
	
		testfields2d.densityCells[i] = UconsCellsNew[i,1];
		testfields2d.UxCells[i] 	  = UconsCellsNew[i,2]/UconsCellsNew[i,1];
		testfields2d.UyCells[i] 	  = UconsCellsNew[i,3]/UconsCellsNew[i,1];
		testfields2d.pressureCells[i] = (Gamma-1.0)*( UconsCellsNew[i,4] - 0.5*( UconsCellsNew[i,2]*UconsCellsNew[i,2] + UconsCellsNew[i,3]*UconsCellsNew[i,3] )/UconsCellsNew[i,1] );

		testfields2d.aSoundCells[i] = sqrt( Gamma * testfields2d.pressureCells[i]/testfields2d.densityCells[i] );
		testfields2d.VMAXCells[i]  = sqrt( testfields2d.UxCells[i]*testfields2d.UxCells[i] + testfields2d.UyCells[i]*testfields2d.UyCells[i] ) + testfields2d.aSoundCells[i];
		
		Delta[i,1] = UconsCellsNew[i,1] - UconsCellsOld[i,1];
		Delta[i,2] = UconsCellsNew[i,2] - UconsCellsOld[i,2];
		Delta[i,3] = UconsCellsNew[i,3] - UconsCellsOld[i,3];
		Delta[i,4] = UconsCellsNew[i,4] - UconsCellsOld[i,4];

		
		UconsCellsOld[i,1] = UconsCellsNew[i,1];
		UconsCellsOld[i,2] = UconsCellsNew[i,2];
		UconsCellsOld[i,3] = UconsCellsNew[i,3];
		UconsCellsOld[i,4] = UconsCellsNew[i,4];
		
	end
	
 end

@everywhere function updateOutput(
	timeVector::Array{Any,1},
	residualsVector1::Array{Any,1},
	residualsVector2::Array{Any,1},
	residualsVector3::Array{Any,1},
	residualsVector4::Array{Any,1},
	residualsVectorMax::Array{Float64,1}, 
	testMesh::mesh2d,
	testFields::fields2d,
	solControls::CONTROLS,
	output::outputCONTROLS,
	dynControls::DYNAMICCONTROLS)

	if (dynControls.verIter == output.verbosity)


		densityWarn = @sprintf("Density Min/Max: %f/%f", dynControls.rhoMin, dynControls.rhoMax);
		out = @sprintf("%0.6f\t %0.6f \t %0.6f \t %0.6f \t %0.6f \t %0.6f \t %0.6f", 
			dynControls.flowTime,
			dynControls.tau,
			residualsVector1[dynControls.curIter]./residualsVectorMax[1],
			residualsVector2[dynControls.curIter]./residualsVectorMax[2],
			residualsVector3[dynControls.curIter]./residualsVectorMax[3],
			residualsVector4[dynControls.curIter]./residualsVectorMax[4],
			dynControls.cpuTime
			 );
		#outputS = string(output, cpuTime);
		#println(outputS); 
		println(out); 
		println(densityWarn);
		
		#densityF = cells2nodesSolutionReconstructionWithStencils(testMesh,  testFields.densityCells  )
		
		if (output.saveDataToVTK == 1)
			filename = string("zzz",dynControls.curIter+1000); 
				
			saveResults2VTK(filename, testMesh, testFields.densityNodes, "density");
			
		end
		 

		 
		# display(testMesh.xNodes)
		# display(testMesh.yNodes)
		# display(testFields.densityNodes);
	
		
	
		if (solControls.plotResidual == 1)	


			
			subplot(2,1,1);
			
			cla();
			
			# display(timeVector);
			# display(residualsVector1./residualsVectorMax[1])
			
			
			tricontourf(testMesh.xNodes,testMesh.yNodes, testFields.densityNodes,pControls.nContours,vmin=pControls.rhoMINcont,vmax=pControls.rhoMAXcont);
			#tricontour(testMesh.xNodes,testMesh.yNodes, testFields.densityNodes,pControls.nContours,vmin=pControls.rhoMINcont,vmax=pControls.rhoMAXcont);
			
			set_cmap("jet");
			xlabel("x");
			ylabel("y");
			title("Contours of density");
			axis("equal");
			
			subplot(2,1,2);
			cla();
			
			if (size(timeVector,1) >1)
				plot(timeVector, residualsVector1./residualsVectorMax[1],"-r",label="continuity"); 
				plot(timeVector, residualsVector2./residualsVectorMax[2],"-g",label="momentum ux"); 
				plot(timeVector, residualsVector3./residualsVectorMax[3],"-b",label="momentum uy"); 
				plot(timeVector, residualsVector4./residualsVectorMax[4],"-c",label="energy"); 
			end
			
			yscale("log");	
			xlabel("flow time [s]");
			ylabel("Res");
			title("Residuals");
			legend();
			
			pause(1.0e-5);
			
			
		end

   

		#pause(1.0e-3);
		dynControls.verIter = 0; 

	end



end


@everywhere function createFields2d_shared(testMesh::mesh2d, thermo::THERMOPHYSICS)


	##display("set initial and boundary conditions ...");

	phsLeftBC = zeros(Float64,4);
	phsTopBC = zeros(Float64,4);

	phsLeftBC[1] = 1.0;
	phsLeftBC[2] = 290.0;
	phsLeftBC[3] = 0.0;
	phsLeftBC[4] = 7143.0;

	phsTopBC[1] = 1.7;
	phsTopBC[2] = 263.72;
	phsTopBC[3] = -51.62;
	phsTopBC[4] = 15282.0;


	densityCells = SharedArray{Float64}(testMesh.nCells); 
	UxCells = SharedArray{Float64}(testMesh.nCells); 
	UyCells = SharedArray{Float64}(testMesh.nCells); 
	pressureCells = SharedArray{Float64}(testMesh.nCells); 
	aSoundCells = SharedArray{Float64}(testMesh.nCells); #speed of sound
	VMAXCells = SharedArray{Float64}(testMesh.nCells); #max speed in domain
	
	densityNodes = SharedArray{Float64}(testMesh.nNodes); 
	UxNodes = SharedArray{Float64}(testMesh.nNodes); 
	UyNodes = SharedArray{Float64}(testMesh.nNodes); 
	pressureNodes = SharedArray{Float64}(testMesh.nNodes); 

	for i=1:testMesh.nCells

		densityCells[i] = phsLeftBC[1];
		UxCells[i] = phsLeftBC[2];
		UyCells[i] = phsLeftBC[3];
		pressureCells[i] = phsLeftBC[4];
			
		aSoundCells[i] = sqrt( thermo.Gamma * pressureCells[i]/densityCells[i] );
		VMAXCells[i]  = sqrt( UxCells[i]*UxCells[i] + UyCells[i]*UyCells[i] ) + aSoundCells[i];
		#entropyCell[i] = UphysCells[i,1]/(thermo.Gamma-1.0)*log(UphysCells[i,4]/UphysCells[i,1]*thermo.Gamma);
				
	end

	
	densityF = cells2nodesSolutionReconstructionWithStencilsSA(testMesh, densityCells); 

		
	if (output.saveDataToVTK == 1)	
		filename = string("zzz",dynControls.curIter+1000);
		saveResults2VTK(filename, testMesh, densityF, "density");
		
	end

		


	# create fields 
	testFields2d = fields2d_shared(
		densityCells,
		UxCells,
		UyCells,
		pressureCells,
		aSoundCells,
		VMAXCells,
		densityNodes,
		UxNodes,
		UyNodes,
		pressureNodes
		#UconsCellsOld,
		#UconsCellsNew
	);

	return testFields2d; 


end



@everywhere function updateOutputSA(
	timeVector::Array{Any,1},
	residualsVector1::Array{Any,1},
	residualsVector2::Array{Any,1},
	residualsVector3::Array{Any,1},
	residualsVector4::Array{Any,1},
	residualsVectorMax::Array{Float64,1}, 
	testMesh::mesh2d,
	testFields::fields2d_shared,
	solControls::CONTROLS,
	output::outputCONTROLS,
	dynControls::DYNAMICCONTROLS)

	if (dynControls.verIter == output.verbosity)


		densityWarn = @sprintf("Density Min/Max: %f/%f", dynControls.rhoMin, dynControls.rhoMax);
		out = @sprintf("%0.6f\t %0.6f \t %0.6f \t %0.6f \t %0.6f \t %0.6f \t %0.6f", 
			dynControls.flowTime,
			dynControls.tau,
			residualsVector1[dynControls.curIter]./residualsVectorMax[1],
			residualsVector2[dynControls.curIter]./residualsVectorMax[2],
			residualsVector3[dynControls.curIter]./residualsVectorMax[3],
			residualsVector4[dynControls.curIter]./residualsVectorMax[4],
			dynControls.cpuTime
			 );
		#outputS = string(output, cpuTime);
		#println(outputS); 
		println(out); 
		println(densityWarn);
		
		densityF = cells2nodesSolutionReconstructionWithStencilsSA(testMesh,  testFields.densityCells  )
		
		
		if (output.saveDataToVTK == 1)
			filename = string("zzz",dynControls.curIter+1000); 
				
			saveResults2VTK(filename, testMesh, densityF, "density");
			
		end
		 

		 
		# display(testMesh.xNodes)
		# display(testMesh.yNodes)
		# display(testFields.densityNodes);
	
		
	
		if (solControls.plotResidual == 1)	


			
			subplot(2,1,1);
			
			cla();
			
			# display(timeVector);
			# display(residualsVector1./residualsVectorMax[1])
			
			
			tricontourf(testMesh.xNodes,testMesh.yNodes, densityF,pControls.nContours,vmin=pControls.rhoMINcont,vmax=pControls.rhoMAXcont);
			#tricontour(testMesh.xNodes,testMesh.yNodes, testFields.densityNodes,pControls.nContours,vmin=pControls.rhoMINcont,vmax=pControls.rhoMAXcont);
			
			set_cmap("jet");
			xlabel("x");
			ylabel("y");
			title("Contours of density");
			axis("equal");
			
			subplot(2,1,2);
			cla();
			
			if (size(timeVector,1) >1)
				plot(timeVector, residualsVector1./residualsVectorMax[1],"-r",label="continuity"); 
				plot(timeVector, residualsVector2./residualsVectorMax[2],"-g",label="momentum ux"); 
				plot(timeVector, residualsVector3./residualsVectorMax[3],"-b",label="momentum uy"); 
				plot(timeVector, residualsVector4./residualsVectorMax[4],"-c",label="energy"); 
			end
			
			yscale("log");	
			xlabel("flow time [s]");
			ylabel("Res");
			title("Residuals");
			legend();
			
			pause(1.0e-5);
			
			
		end

   

		#pause(1.0e-3);
		dynControls.verIter = 0; 

	end



end

