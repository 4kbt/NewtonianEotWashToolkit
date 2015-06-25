function d =  scanCenterAndRemoveLinear(d)

	% Center it
	d(:,1) = d(:,1) - mean(d(:,1));
	d(:,2) = d(:,2) - mean(d(:,2));

	%fit it
	[b s r] = ols(d(:,3), d(:,1:2));

	%remove linear fit
	d(:,3) =  d(:,3) - d(:,1:2)*b;

	%Null it
	d(:,3) = d(:,3) - mean(d(:,3));

	%SI units
	d = d*1e-3;

end
