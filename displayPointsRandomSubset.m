%Companiou function to pointMassGravity
%Displays test mass (array1) in blue, field mass (array2) in red
function displayPointsRandomSubset(array1, array2, fraction)

	array1 = array1( (rand(rows(array1),1) < fraction ) , :);
	array2 = array2( (rand(rows(array2),1) < fraction ) , :);

	plot3(0,0,0,'x4',array1(:,2), array1(:,3), array1(:,4),'.3',...
	      array2(:,2), array2(:,3), array2(:,4),'.1');

end
