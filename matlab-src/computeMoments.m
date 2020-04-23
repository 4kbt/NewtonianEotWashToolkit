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

%m is a point-mass array
%moments are computed about (0,0,0), with the primary axis oriented vertically, as usual.
%phi increases counterclockwise from the x axis

function moment = computeMoments( l, m, mArray )

	moment = 0;

	for ctr = 1:rows(mArray)

		theta = 0;
		phi   = 0;

		x = mArray(ctr, 2);
		y = mArray(ctr, 3);
		z = mArray(ctr, 4);

		r     = sqrt( sum( mArray( ctr, 2:4).^2 ) );
		if( r > 0)

			%compute in-plane radius
			rxy   = sqrt( sum( mArray( ctr, 2:3).^2 ) );
			
			%Theta computation
			%edge cases
			if( rxy == 0 )
				if( z > 0)
					theta =  0;
				end
				if( z < 0)
					theta = pi;
				end
			%main case
			else
				theta = atan( z / rxy ) + pi / 2.0;
			end

			%Phi computation
			%edge cases
			if( x == 0 )
				if( y > 0 )
					phi = pi/2.0;
				end
				if( y < 0 )
					phi = -pi/2.0;
				end
			%main case
			else
				phi   = atan( y / x );
			end 

			%handle the fact that phi runs from 0-2pi
			if ( sign ( x )  == -1 )
				phi = phi + pi;
			end
		end

		moment = moment + mArray(ctr,1) * r^l * ...
			   conj( YLM( l, m, theta, phi) );
	end
end


%!test
%! %From D'Urso and Adelberger
%! assert( abs(computeMoments(3,3, [1 1 0 0; 1 1 0 0]) - -sqrt(35/16/pi) ) ...
%!                                                       < 10*eps) 
%! assert( abs(computeMoments(3,3, [1 2 0 0; 1 2 0 0]) - -sqrt(35/16/pi) * 8 ) ...
%!							 < 10*eps) 
%! assert( abs(computeMoments(1,1, [1 1 0 0; 1 1 0 0]) - -sqrt(3/2/pi) ) ...
%!                                                       < 10*eps) 
%! assert( abs(computeMoments(1,1, [1 2 0 0; 1 2 0 0]) - -sqrt(3/2/pi) * 2 ) ...
%!                                                       < 10*eps) 

