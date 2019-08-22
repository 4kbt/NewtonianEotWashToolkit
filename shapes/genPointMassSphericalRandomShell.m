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

%Generates a spherical shell with mass M, radius R, and approximately N points.
function sph = genPointMassSphericalRandomShell( M, R, N)

        %From Marsaglia (1972)
        %http://mathworld.wolfram.com/SpherePointPicking.html

	%Generate 2
	x = 2*rand(floor(N*4/pi), 2) - 1;

	%Cut into circle
	x2 = sum(x.^2, 2);
	x = x( x2 < 1, :);
	x2 = x2( x2 < 1 );

	%speed
	sx2 = sqrt(1-x2);

	%conformally map
	sph = [ ones(rows(x),1), ...
		2 * x(:,1) .* sx2, ...
		2 * x(:,2) .* sx2, ...
		1 - 2 * x2];	

	%set radius, mass
	sph(:,2:4) = sph(:,2:4) * R;
	sph(:,1) = sph(:,1) * M / sum( sph(:,1));

end 

%!test
%! more off
%! m = 1; R = 10; N = 100000;
%! s = genPointMassSphericalRandomShell( m, R, N);
%! thresh = 100*sqrt(rows(s))*eps;
%! %Radius Check
%! assert(abs(sqrt(sum(s(:,2:4).^2,2)) - R) < thresh)
%! %Mass Check
%! assert( abs( sum(s(:,1)) - m ) < thresh)
%! %Quadrupole check
%! zq = s( abs(s(:,4)) > R/2 , :);
%! yq = s( abs(s(:,3)) > R/2 , :);
%! xq = s( abs(s(:,2)) > R/2 , :);
%! q = [sum(xq(:,1)) sum(yq(:,1))  sum(zq(:,1)) ];
%! assert( (max(q) - min(q))/mean(q) < 2*sqrt(N) ) 
