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

function d =  scanCenterAndRemoveLinear(d)

	% Center it
	d(:,1) = d(:,1) - mean(d(:,1));
	d(:,2) = d(:,2) - mean(d(:,2));

	%fit it
	[b s r] = ols(d(:,3), d(:,1:2));

	%remove linear fit
	d(:,3) =  d(:,3) - d(:,1:2)*b;

	%Null it
	d(:,3) = d(:,3) - mean(d(:,3));

	%SI units
	d = d*1e-3;

end
