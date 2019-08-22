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

%Returns the three axis force on mass1 by mass2 (vector)
function force=Gmmr2Array(mass1, mass2)

	fundamentalConstants

	%Which way does the force act?
	rvec = mass2( : , 2:4 ) - ones( rows( mass2 ) , 1 ) * mass1( 2:4 );

	%Pythagorean theorem to determine |r|
	r = sqrt( sum( (rvec .* rvec) , 2 ) );
	
	%The inverse square law!
	force = rvec' * ( G * mass1( 1 ) * mass2( : , 1 ) ./ r .^ 3 );

end
