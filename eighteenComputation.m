more off;

clear;

%import the shapes directory
addpath('shapes/');

%Set up parameters
wInnerRadius = 23.5e-3; % ted's thesis
wOuterRadius = 26e-3;   % ted's thesis
wSubtendedAngle = 2*pi/36; %18-omega;
wThick = 54e-6;	% ted's thesis

nPts = 8 %number of points across the wedge at its widest point
nCirPoints = ceil(nPts/sin(wSubtendedAngle/2));  %this works up to angle of pi
nzPts = 1;
nAngles = 100; %number of angles to sample

rho = 21000;

%Construct a single wedge
wedgeMass = rho * pi *( wOuterRadius.^2 - wInnerRadius.^2) * ...
		wSubtendedAngle/2/pi *wThick;

wedgeMass * 18

Wedge = genPointMassWedge(wedgeMass,  wInnerRadius, wOuterRadius, wThick,
			  wSubtendedAngle, nzPts, nCirPoints);

Wedge = rotatePMArray(Wedge, pi/2, [0 1 0]); 

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

for XOffset = linspace(-5e-3 , 5e-3, 20);
for YOffset = linspace(-5e-3 , 5e-3, 20);
for height = 160e-6 %logspace(-4,-2,20)

	XOffset
	YOffset
	height

	out = [];
	for angle = (2*pi*rand(1,nAngles))
		angle

		A = translatePMArray(Attractor, [0 0 -height]);
		A = rotatePMArray(A,angle,[0 0 1]);
		A = translatePMArray(A,[XOffset YOffset 0]);

		[f t] = pointMatrixGravity(Pendulum,A);

		out = [out; XOffset YOffset height angle f t];

	end

	%Fit it!
	[b s r] = sineFitter(out(:,3), out(:,9), 18/2/pi);

	%Save it all.
	fit = [fit; XOffset YOffset height b' s' ];
	ResultsArray = [ResultsArray; out];
save 'HorizontalXYResultsArray160um8pts.dat' ResultsArray

bsFitter

save 'HorizontalXYFitOutputHeight160um8pts.dat' BootOut

end %height
end %YOffset
end %XOffset


save 'HorizontalXYResultsArray160um8pts.dat' ResultsArray

loglogerr(BootOut(:,2), sqrt(BootOut(:,3).^2 + BootOut(:,4).^2), sqrt(BootOut(:,6).^2+BootOut(:,7).^2))

