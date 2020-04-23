#    PointGravity: a simple point-wise Newtonian gravitation calculator.
#    Copyright (C) 2017  Charles A. Hagedorn
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 3.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.



%Create a simple pendulum
Pendulum = [0.1 5e-2 0 5e-2;
	    0.1 -5e-2 0 -5e-2];

%Create a simple attractor
Attractor = Pendulum;


%Pre-allocate memory (fully-allocate for a massive speedup)
results = [];


%Main loop
for angle = 0:0.1:(2*pi)

	%This is the attractor we will move
	TempAttractor = Attractor;

	%Rotate attractor by "angle"
	TempAttractor  = rotatePMArray(Attractor, angle,  [0 0 1]);

	%Translate attractor to some desired position
	TempAttractor = translatePMArray(TempAttractor, [-30e-2 0 0] );

	%Uncomment this (and page through the pauses) to verify correct behavior
	%displayPoints(Pendulum, TempAttractor); pause

	%Compute forces and torques
	[Force Torque] = pointMatrixGravity(Pendulum, TempAttractor);


	%Aggregate forces and torques
	results = [results; angle Force Torque];


end

%Plot torque on pendulum
 plot(results(:,1), results(:,7))
xlabel('angle (rad)');
ylabel('torque (N-m)');

