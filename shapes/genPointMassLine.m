function  o = genPointMassLine( eachMass, x, y, zstart, zspace, N)

        o = ones(N,4);

	o = [  o(:,1) * eachMass, o(:,2) * x, o(:,3) * y, (0: (N- 1) )' * zspace];


end

