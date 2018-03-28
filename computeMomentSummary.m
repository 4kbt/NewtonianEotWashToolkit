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
