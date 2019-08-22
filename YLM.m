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

function o = YLM(l,m, theta, phi)

 	o = legendre_sphPlm( l, m, cos(theta) ) .*  exp( i * m * phi);

end

%!test
%!assert( YLM(0,0,0,0) == 1/sqrt(4*pi) );

%!test
%! N = 50;
%! angles = rand( N , 2); 
%! angles(:,1) = angles(:,1) * pi;
%! angles(:,2) = angles(:,2) * pi * 2; 
%! for l = 0:10
%! l
%! for m = 0:l
%! for ctr = 1:N
%!	y =  YLM( l, m, angles(ctr,1), angles(ctr,2) );
%!	expected = sqrt( ( 2 * l + 1 ) / 4 / pi * factorial( l - m ) / ...
%!	 factorial( l + m) ) * ...
%!	 legendre(l, cos( angles(ctr , 1) )  ) (m + 1) * exp(i* m * angles(ctr, 2) );
%!	assert ( abs( y - expected) / y < 1e-6 ) 
%! end
%! end
%! end 
