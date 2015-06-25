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
