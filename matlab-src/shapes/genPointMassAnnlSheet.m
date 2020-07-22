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

function array= genPointMassAnnlSheet(mass, iRadius, oRadius, thickness, ...
					nxpoints, nypoints)

	xgrid = thickness     / nxpoints;
	ygrid = oRadius * 2.0 / nypoints;
	zgrid = ygrid;
	nzpoints= nypoints;

	density = mass / ( pi * ( oRadius^2 - iRadius^2 ) * thickness );
	
	pointMass = density * xgrid * ygrid * zgrid;

	array = zeros(nxpoints*nypoints*nzpoints, 4);

	loopcounter = 1;

	for i = 1:nxpoints
		for j = 1:nypoints
			for k = 1:nzpoints
				newMass=[ pointMass, ...
					( i - (nxpoints + 1) / 2) * xgrid,...
					( j - (nypoints + 1) / 2) * ygrid,...
					( k - (nzpoints + 1) / 2) * zgrid];

				pRadius = sqrt( newMass(3)^2 + newMass(4)^2 );

				if(pRadius <= oRadius && pRadius >= iRadius)
					array(loopcounter,:) = newMass;
					loopcounter = loopcounter + 1;
				end
			end
		end
	end

	%decrement loopcounter to undo final increment;
	loopcounter = loopcounter - 1;

	%trim array to appropriate size
	array = array(1:loopcounter,:);

	%Correcting any mass discrepancy
	MassDiscrepancyRatio = sum( array(:,1) ) / mass;
	array(:,1) = array(:,1) / MassDiscrepancyRatio;
	MassDiscrepancyRatio = sum( array(:,1) ) / mass;
end

%!test
%!
%! M = genPointMassAnnlSheet( 1, 0, 1, 1, 2, 2);
%!
%! %Are there the expected eight points?
%! assert( rows(M) == 8 );
%! 
%! %Are all the points the same distance from the origin?
%! r2 = M(1,2)^2 + M(1,3)^2 + M(1,4)^2;
%! assert(  ones(rows(M),1) * r2 == M(:,2).^2 + M(:,3).^2 + M(:,4).^2);
%! 
%! %Is mass properly normalized?
%! assert( sum(M(:,1)) == 1);
%! assert( sum(abs( diff( M(:,1) ) ) )  < rows(M) * eps);
%! 
%! %Are points properly distributed?
%! assert( genPointMassAnnlSheet( 1, 0 , 1, 1, 1,1) == [ 1 0 0 0 ])
%!
%! assert( M == [ 0.125 -0.25 -0.5 -0.5;
%!                0.125 -0.25 -0.5  0.5;
%!                0.125 -0.25  0.5 -0.5;
%!                0.125 -0.25  0.5  0.5;
%!                0.125  0.25 -0.5 -0.5;
%!                0.125  0.25 -0.5  0.5;
%!                0.125  0.25  0.5 -0.5;
%!                0.125  0.25  0.5  0.5;] )
