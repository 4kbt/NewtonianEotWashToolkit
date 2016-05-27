BootOut = [];

for ctr = 1:nAngles:(rows(ResultsArray) )

	RA = ResultsArray(ctr:(ctr+nAngles - 1),:);

	x = RA(1,1);
	y = RA(1,2);
	h = RA(1,2);

	ib = [];
	for ictr = 1:49 %49 is 7*7 = # of bootstraps.
	 	B= bootstrapData(RA);
		[b s r] = sineFitter(B(:,4), B(:,10), 18/2/pi);

		ib = [ib; b'];
	end

	BootOut = [BootOut; x y h mean(ib) std(ib)];

end
