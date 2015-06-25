include ../../Makefile.inc

AGGREGATE := testOutput/aggregate.test

all : $(AGGREGATE)

$(AGGREGATE): $(shell ls *.m | sed 's/\.m/\.test/')
	-rm testOutput/aggregate.test
	#The below is a dangerous, but apparently functional, usage of the minus-sign operator.
	#The line below is not a failure of grep, rather a grep for failure
	-grep failed testOutput/* > $(AGGREGATE)
	#The following line also doesn't denote any error.
	$(if $(shell cat $(AGGREGATE)),	$(HOMEDIR)/bin/assert.sh  "`wc $(AGGREGATE) | awk '{print $$1}'` -lt 1" "assert error")

%.test : %.m 
	$(OCT) --eval "test $*" > testOutput/$@

clean :
	-rm testOutput/*
