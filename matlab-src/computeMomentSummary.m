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

%Computes a summary of the moments of a mass distribution
%
% This is a wrapper for computeMoments to do the most-common task
%
% Inputs: maxL - the largest l you'd like to compute
%       : massDistribution - the mass distribution for which moments are desired
%
% Output: [l m moment], one row for each moment.

function output = computeMomentSummary(maxL, massDistribution)

	output = [];

	for l = 1:maxL
		for m = 0:l
			output = [ output; l m computeMoments(l,m, massDistribution)];
		end
	end
end
