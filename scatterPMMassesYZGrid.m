%randomly scatters point mass arrays in the y and z directions with standard deviation of the scan gridsize.
%Gridsize is hardcoded for scope.
function M = scatterPMMassesYZGrid(M);

	run3147PendulumParameters;

	R = pendulumPMScanGridSize * randn(rows(M), 2);

	M(:,2:3) = M(:,2:3) + R;

end
