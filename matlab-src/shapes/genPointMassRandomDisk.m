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

function m= genPointMassRandomAnnl(pointsMass, iRadius, oRadius, NPts)
	if(iRadius < 0 | oRadius < 0)
		error('negative radii!');
	end

	if (iRadius >= oRadius)
		error ('inner radius cannot be larger than outer radius')
	end

	r = [];

	while ( rows(r) < NPts)

		tem = rand(1,2)*2-1.0;

		rad = sqrt(tem(:,1).^2 + tem(:,2).^2);

		if( rad >= iRadius/oRadius & rad <=1)
			r = [r;tem];
		end
	end

	r = r.*oRadius;

	m = [ pointsMass * ones(rows(r),1) zeros(rows(r),1) r];

end
