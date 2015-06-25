function array= genPointMassAnnlSheet(mass, iRadius, oRadius, thickness, ...
					nxpoints, nypoints)

	xgrid = thickness     / nxpoints;
	ygrid = oRadius * 2.0 / nypoints;
	zgrid = ygrid;
	nzpoints= nypoints;

	density = mass / ( pi * ( oRadius^2 - iRadius^2 ) * thickness );
	
	pointMass = density * xgrid * ygrid * zgrid;

	array = [];

	for i = 1:nxpoints
		for j = 1:nypoints
			for k = 1:nzpoints
				newMass=[ pointMass, ...
					( i - (nxpoints + 1) / 2) * xgrid,...
					( j - (nypoints + 1) / 2) * ygrid,...
					( k - (nzpoints + 1) / 2) * zgrid];

				pRadius = sqrt( newMass(3)^2 + newMass(4)^2 );

				if(pRadius <= oRadius && pRadius >= iRadius)
					array = [ array ; newMass ];
				end
			end
		end
	end

	%Correcting any mass discrepancy
	MassDiscrepancyRatio = sum( array(:,1) ) / mass;
	array(:,1) = array(:,1) / MassDiscrepancyRatio;
	MassDiscrepancyRatio = sum( array(:,1) ) / mass;
end
