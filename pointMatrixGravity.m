%Returns the three axis force and torque on array1 by array2
%array entries of the form [m, x, y, z] 
function [force, torque]=pointMatrixGravity(array1,array2)

	force  = [0 0 0];

	torque = [0 0 0];

	for i = 1:rows(array1)

			iforce  = Gmmr2Array( array1( i , : ) , array2 );
	
			itorque = cross( array1( i , 2:4 ) , iforce' , 2 );

			torque += itorque;
			force  += iforce';
	end
end

%!test
%! 'pointMatrix force test'
%! fundamentalConstants
%! m1 = [ 1 0 0 0 ];
%! m2 = [ 1 1 0 0 ];
%! [ F T ] = pointMatrixGravity( m1 , m2 );
%! Fg = G;
%! assert( abs( F( 1 ) - Fg ) < 2 * eps )
%! assert( abs( F( 2:3 )    ) < 2 * eps )
%! assert( abs( T )           < 2 * eps )

%!test
%! 'pointMatrix torque test'
%! fundamentalConstants
%! m1 = [1 0 1 0];
%! m2 = [1 1 1 0];
%! [ F T ] = pointMatrixGravity( m1 , m2 );
%! Fg = G;
%! assert( abs( F( 1) - Fg ) < 2 * eps )
%! assert( abs( F( 2:3 )   ) < 2 * eps )
%! assert( abs( T( 1:2 )   ) < 2 * eps )
%! assert( abs( T( 3 )+ Fg ) < 2 * eps )

%!test 
%! 'pointMatrix ISL convergence test'
%! fundamentalConstants
%! m1 = genPointMassAnnlSheet( 1, 0, 1, 1, 5, 10 );
%! for ctr = 1:100
%! 	exp = rand * 5 + 2;
%!	d = 10^exp;
%!	r = randn * 0.01;
%!	md = translatePMArray( m1 , [ d, r, 0 ] );
%!	[ F T ] = pointMatrixGravity( m1, md );
%!	assert( sum( abs( T ) ) < 6 * eps )
%!	assert( abs( F( 1 ) - G / d^2 ) / ( G / d^2 ) < 0.001 )
%!  end

%!test
%! 'pointMatrix sheet uniformity test'
%! fundamentalConstants
%! tm = [ 1 0 0 0 ];
%! r = 140; t = 0.01; xspacing = 0.005; rspacing = 0.5; m = 1 * pi * r * r * t;
%! sheet = genPointMassAnnlSheet( m, 0, r, t, t / xspacing, r / rspacing );
%! v = [];
%! for ctr = 1:100
%!	d = 2.0 * rand + rspacing; %yes, rand
%!	y = randn * rspacing;
%!	z = randn * rspacing;
%!	s = translatePMArray( sheet, [ d, y, z ] );
%!	v = [v; d y z pointMatrixGravity( tm, s ) ];
%!	clear s
%! end
%! save( [ "testOutput/flatSheet.dat" ], "v" );
%! expectedForce = 2 * pi * G * t;
%! longerRange = v( v(:,1) > 1, : );
%! assert( abs( longerRange(:,4) / expectedForce - 1 ) < 0.02);

%!test 
%! "newton's shell theorem"
%! shell = genPointMassSphericalRandomShell( 1, 10, 100000 );
%! v = [];
%! for ctr = 1:300
%! 	p = randn( 1, 3 );
%!	m = [ 1 0 0 0 ];
%!	s = translatePMArray( shell, p );
%!	v = [ v; p, pointMatrixGravity( m, s ) ];
%! end
%! save( [ "testOutput/sphereShell.dat" ], "v" );
%! scatter = sum( v( :, 4:6 ).^2 , 2 );
%! [ fullF fullT ] = pointMatrixGravity( m, [ 1, 10, 0, 0 ] );
%! 'fractional error'
%! fe = sqrt( max( scatter ) ) / fullF( 1 )
%! assert( fe < 0.01 );

%!test
%! "quadrupole torques"
%! for iterations = 1:100
%! fundamentalConstants
%! d = 1; R = rand * 100 + 1.1; m = 1; M = 1
%! p = [ m d 0 0; m -d  0 0 ];
%! q = [ M R 0 0; M -R 0 0  ]; 
%! v = [];
%! for a = 0:( 2 * pi / 60 ):( 2 * pi )
%!	Q = rotatePMArray(q, a, [0 1 0]);
%!	% Long-range quadrupole
%!	K = 2 * 6.0 * G * M * m * d^2 / R^3 * (-sin( a ) ) * cos( a );
%!	% Krishna's exact quadrupole torque
%!     Tau = 2 * G * M * m * d / R^2 * -sin(a) * ( 1. / (1 - 2 *d/R * cos(a) ...
%!		+ (d/R)^2)^(3/2) - 1./(1 + 2 * d /R * cos(a) + (d/R)^2)^(3/2) );
%!	[ f, t ] = pointMatrixGravity( p, Q );
%!	v = [ v; a K f t Q(1,:) Tau ];
%! end
%! save( "testOutput/quadrupoleTest.dat", "v" );
%! assert( abs( v( :, 13 ) -  v( :, 7 ) ) < (mean( abs( v( :, 13) )) / 1e9 ) )
%! assert( v( :, 3:5 ) < 10 * eps );
%! end %iterations 
