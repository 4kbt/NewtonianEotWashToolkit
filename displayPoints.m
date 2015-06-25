%Companion function to pointMassGravity
%Displays test mass (array1) in blue, field mass (array2) in red
function displayPoints(array1, array2)

	plot3(0,0,0,'x4', ...
		array1(:,2) , array1(:,3) , array1(:,4) , '.3' ,...
	      	array2(:,2) , array2(:,3) , array2(:,4) , '.1' );
	xlabel( 'x (m)' );
	ylabel( 'y (m)' );
	zlabel( 'z (m)' );
	axis( "equal" );

end
