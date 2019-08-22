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

OCT:= octave --no-init-file \
	--eval 'graphics_toolkit("gnuplot");'\
	--eval 'ignore_function_time_stamp("all");'\
	-p shapes/ \
	-q 

GNU    := gnuplot -e 'set term dumb' -e 'HOMEDIR = "$(HOMEDIR)"' -d

.INTERACT: 
	$(OCT) --persist --eval 'ignore_function_time_stamp("system");'

#The dash is meaningful.
.GINTERACT:
	$(GNU) -

.PHONY : 
	@echo $(OCT)

PARALLEL := -j 8 

AGGREGATE := testOutput/aggregate.test

all : $(AGGREGATE)

$(AGGREGATE): $(shell ls *.m | sed 's/\.m/\.test/')
	-rm testOutput/aggregate.test
	#The below is a dangerous, but apparently functional, usage of the minus-sign operator.
	#The line below is not a failure of grep, rather a grep for failure
	-grep failed testOutput/* > $(AGGREGATE)
	#The following line also doesn't denote any error.
	$(if $(shell cat $(AGGREGATE)),	./assert.sh  "`wc $(AGGREGATE) | awk '{print $$1}'` -lt 1" "assert error")

%.test : %.m 
	$(OCT) --eval "test $*" > testOutput/$@

clean :
	-rm testOutput/*
