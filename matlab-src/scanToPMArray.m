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

function [o]= scanToPMArray( scanData, xspacing, yspacing, zspacing, density)

	xmin = min(scanData(:,1));
	xmax = max(scanData(:,1));

	ymin = min(scanData(:,2));
	ymax = max(scanData(:,2));

	xSpan = xmin:xspacing:xmax;
	ySpan = ymin:yspacing:ymax;

	xs = repmat(xSpan, columns(ySpan), 1); 
	ys = repmat(ySpan', 1,  columns(xSpan));

	tic

	'gridding data'
	[xi yi zi] = griddata(scanData(:,1), scanData(:,2), scanData(:,3), xs,ys);
	toc

	'building point mass array'
	o = genPointMassArrayFrom2DArray( zi, xspacing, yspacing, zspacing, density);
end

	
