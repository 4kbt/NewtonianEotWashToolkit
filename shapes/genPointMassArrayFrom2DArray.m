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

%Array is in the x-y plane
function o = genPointMassArrayFrom2DArray( A, xSpacing , ySpacing, zSpacing, density)

	m = xSpacing*ySpacing*zSpacing*density;

	%gridding has already happened prior to this step, so this makes sense
	xSize = xSpacing * columns(A);
	ySize = ySpacing * rows(A);

	%define positions
	xpos = (xSpacing - xSize)/2.0 + xSpacing * ( 0 : (columns(A) - 1) );
	ypos = (ySpacing - ySize)/2.0 + ySpacing * ( 0 : (rows(A)    - 1) );
	zstart = zSpacing/2.0;

	tic

	%Simple z-grid	
	A = floor(A./zSpacing);

	%preallocation of memory. The two is an overcompensation, corrected at the end.
	o = zeros( sum(sum(A(~isnan(A))))*2 , 4);

	%Printing these may or may not be important, they're illuminating for prealloc errors
	sum(sum(A(~isnan(A))))
	size(o)

	%Define zsteps
	zSteps =( 0:(max(max(A)))  )*zSpacing + zstart;
	zSteps = zSteps';

	%Apparently, I need a lookup table for z-steps
	lib = ones( rows(zSteps), 4); 
	lib(:,4) = zSteps;
	lib(:,1) = lib(:,1)*m;

	tem = [];

	octr  = 1; 
	for xctr = 1:columns(A)
		for yctr = 1:rows(A)

			if(A(yctr,xctr) > 0 )
				%Create points from this column at xpos,ypos.
				tem = lib(1:A(yctr,xctr),:);
				tem(:,2) = tem(:,2)*xpos(xctr);
				tem(:,3) = tem(:,3)*ypos(yctr);

				%The price you pay for preallocation, bookkeeping
				o( octr: ( octr+ A(yctr,xctr) - 1 ), :) = tem;

				%increment counter
				octr = octr+rows(tem);
			end

		end

		fraction = xctr/columns(A)*100.0;

		if( mod(xctr,30) < 1 )
			['Assembly is ' num2str(fraction) ' percent complete']
			toc
		end
	end

	%Compensates for over-allocating memory; fixes any trivialities, too.
	o = o( o(:,1) > 0, :);
end

