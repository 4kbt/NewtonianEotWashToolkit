BootOut = [];

for ctr = 1:nAngles:(rows(ResultsArray) )

	RA = ResultsArray(ctr:(ctr+nAngles - 1),:);

	o = RA(1,1);
	h = RA(1,2);

	ib = [];
	for ictr = 1:21
	 	B= bootstrapData(RA);
		[b s r] = sineFitter(B(:,3), B(:,9), 18/2/pi);

		ib = [ib; b'];
	end

	BootOut = [BootOut; o h mean(ib) std(ib)];

end
