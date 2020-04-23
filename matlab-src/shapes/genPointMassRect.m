#    PointGravity: a simple point-wise Newtonian gravitation calculator.
#    Copyright (C) 2017  Charles A. Hagedorn
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 3.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

	array = zeros(npoints,4);
	ctr = 1;

	for i = 1:nxpoints
		for j = 1:nypoints
			for k = 1:nzpoints
				newMass=[ pointMass, ...
					( i - (nxpoints+1) / 2) * xgrid,...
					( j - (nypoints+1) / 2) * ygrid,...
					( k - (nzpoints+1) / 2) * zgrid];

					array(ctr,:) =  newMass;
					ctr++;
			end
		end
	end

	if( rows( array ) != npoints )
		## counting problem in XSheet
		array = 0;
	end
end
