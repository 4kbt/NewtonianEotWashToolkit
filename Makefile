OCT:= octave --no-init-file \
	--eval 'graphics_toolkit("gnuplot");'\
	--eval 'ignore_function_time_stamp("all");'\
	-q 

GNU    := gnuplot -e 'set term dumb' -e 'HOMEDIR = "$(HOMEDIR)"' -d

LYXINTERACT := lyx -userdir $(HOMEDIR)/.lyx
.LYXINTERACT:
	$(LYXINTERACT)

LYX	:= $(LYXINTERACT) -batch --export "pdf2" -dbg info,latex

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
