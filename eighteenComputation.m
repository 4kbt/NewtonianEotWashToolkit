more off;

clear;

%import the shapes directory
addpath('shapes/');

%Set up parameters
wInnerRadius = 23.5e-3; % ted's thesis
wOuterRadius = 26e-3;   % ted's thesis

wThick = 54e-6;	% ted's thesis
wWidth = wInnerRadius * 2*pi/36;
wLength = wOuterRadius-wInnerRadius;

nPts = 16 %number of points passed to each axis of genPMRect.
nzPts = 1;
nAngles = 100; %number of angles to sample

rho = 21000;

%Construct a single wedge
wedgeMass = rho * wThick * wWidth * wLength;

wedgeMass * 18

Wedge = genPointMassRect(wedgeMass,  wLength, wWidth, wThick, nPts, nPts, nzPts);

Wedge = translatePMArray(Wedge, [wInnerRadius - wLength/2 0 0]);

%Build up mass array
Ring =[];

for ctr = 0:17

	Ring = [Ring; rotatePMArray(Wedge, 2*pi/18*ctr,[0 0 1])];
end

Pendulum = Ring;

Attractor = translatePMArray(Ring, [0 0 -wThick/2.0]);

%initialize result-accumulating arrays
ResultsArray =[];
fit = [];

for HorizOffset = [0 1e-3 2e-3 3.5e-3 5e-3 10e-3];
for height = logspace(-4,-2,20)

	HorizOffset
	height

	out = [];
	for angle = (2*pi*rand(1,nAngles))
		angle

		A = translatePMArray(Attractor, [0 0 -height]);
		A = rotatePMArray(A,angle,[0 0 1]);
		A = translatePMArray(A,[HorizOffset 0 0]);

		[f t] = pointMatrixGravity(Pendulum,A);

		out = [out; HorizOffset height angle f t];

	end

	%Fit it!
	[b s r] = sineFitter(out(:,3), out(:,9), 18/2/pi);

	%Save it all.
	fit = [fit; HorizOffset height b' s' ];
	ResultsArray = [ResultsArray; out];
save 'HorizontalOffsetResultsArray.dat' ResultsArray

end %height
end %HorizOffset

bsFitter

save 'HorizontalOffsetFitOutput.dat' BootOut

save 'HorizontalOffsetResultsArray.dat' ResultsArray

loglogerr(BootOut(:,2), sqrt(BootOut(:,3).^2 + BootOut(:,4).^2), sqrt(BootOut(:,6).^2+BootOut(:,7).^2))

