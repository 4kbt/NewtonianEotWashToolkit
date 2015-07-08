function array= genPointMassRect(mass, thickness, width, height, ...
					nxpoints,nypoints,nzpoints)

	xgrid   = thickness / nxpoints;
	ygrid   = width     / nypoints;
	zgrid   = height    / nzpoints;

	npoints = nxpoints * nypoints * nzpoints;
	
	if( mod( npoints , 1 ) != 0 )
		error('npoints IS NOT AN INTEGER!');
	end

	pointMass = mass / npoints;

	array = [];

	for i = 1:nxpoints
		for j = 1:nypoints
			for k = 1:nzpoints
				newMass=[ pointMass, ...
					( i - (nxpoints+1) / 2) * xgrid,...
					( j - (nypoints+1) / 2) * ygrid,...
					( k - (nzpoints+1) / 2) * zgrid];

					array = [ array ; newMass ];
			end
		end
	end

	if( rows( array ) != npoints )
		## counting problem in XSheet
		array = 0;
	end
end
