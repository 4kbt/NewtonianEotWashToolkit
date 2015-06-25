function [o]= scanToPMArray( scanData, xspacing, yspacing, zspacing, density)

	xmin = min(scanData(:,1));
	xmax = max(scanData(:,1));

	ymin = min(scanData(:,2));
	ymax = max(scanData(:,2));

	xSpan = xmin:xspacing:xmax;
	ySpan = ymin:yspacing:ymax;

	xs = repmat(xSpan, columns(ySpan), 1); 
	ys = repmat(ySpan', 1,  columns(xSpan));

	tic

	'gridding data'
	[xi yi zi] = griddata(scanData(:,1), scanData(:,2), scanData(:,3), xs,ys);
	toc

	'building point mass array'
	o = genPointMassArrayFrom2DArray( zi, xspacing, yspacing, zspacing, density);
end

	
