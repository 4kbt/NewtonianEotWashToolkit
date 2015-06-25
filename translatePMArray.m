%Translates a point mass array by transVec (a three vector)

function transArray = translatePMArray( array, transVec )
	
	array( : , 2:4 ) = array(:, 2:4) + ones( rows( array ) ,1) * transVec;

	transArray=array;
end

%!test
%! %The simplest possible test
%! o = translatePMArray( [1 0 0 0] , [1 1 1] );
%! assert(o == [1 1 1 1]);


%!test
%! %Randomly generate mass distributions, translate by random amounts, verify.
%! numMasses = 6;
%! for ctr = 1:100
%! 	v = randn( 1 , 3 );
%!	m = randn( numMasses , 4 );
%!	o = translatePMArray( m, v );
%!	assert( m( : , 1 ) == o( : , 1 ) );
%!	newDiff      = o( : , 2:4 ) - m( : , 2:4 );
%!	expectedDiff = repmat( v , numMasses , 1 );
%!	totalError   = sum( sum( newDiff - expectedDiff ) );
%! 	assert( totalError < 3 * numMasses * eps );
%! end
