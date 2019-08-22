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

function m= genPointMassRandomRectangle(pointsMass, yWidth, zWidth, NPts)
	if(zWidth < 0 | yWidth < 0)
		error('negative widths!');
	end

	r = rand( NPts, 2);
	r = r*2-1;

	r(:,1) = r(:,1) * yWidth/2.0;
	r(:,2) = r(:,2) * zWidth/2.0;

	m = [pointsMass*ones(rows(r),1), zeros(rows(r), 1), r];
end
