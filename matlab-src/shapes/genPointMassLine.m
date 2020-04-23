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

function  o = genPointMassLine( eachMass, x, y, zstart, zspace, N)

        o = ones(N,4);

	o = [  o(:,1) * eachMass, o(:,2) * x, o(:,3) * y, (0: (N- 1) )' * zspace];


end

