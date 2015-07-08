function o = YLM(l,m, theta, phi)

 	o = legendre_sphPlm( l, m, cos(theta) ) .*  exp( i * m * phi);

end

%!test
%!assert( YLM(0,0,0,0) == 1/sqrt(4*pi) );

%!test
%! N = 50;
%! angles = rand( N , 2); 
%! angles(:,1) = angles(:,1) * pi;
%! angles(:,2) = angles(:,2) * pi * 2; 
%! for l = 0:10
%! l
%! for m = 0:l
%! for ctr = 1:N
%!	y =  YLM( l, m, angles(ctr,1), angles(ctr,2) );
%!	expected = sqrt( ( 2 * l + 1 ) / 4 / pi * factorial( l - m ) / ...
%!	 factorial( l + m) ) * ...
%!	 legendre(l, cos( angles(ctr , 1) )  ) (m + 1) * exp(i* m * angles(ctr, 2) );
%!	assert ( abs( y - expected) / y < 1e-6 ) 
%! end
%! end
%! end 
