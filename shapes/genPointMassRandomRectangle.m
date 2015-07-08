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
