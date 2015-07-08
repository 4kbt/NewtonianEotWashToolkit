function array=genPointMassXRing(mass, radius, npoints)

	pointMass=mass/npoints;

	for i=1:npoints
		angle=2*pi/npoints*(i-1);
		array(i,:)=[pointMass, 0, sin(angle)*radius, cos(angle)*radius];
	end

end
