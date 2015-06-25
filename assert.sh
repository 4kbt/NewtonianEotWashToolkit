#!/bin/bash

#inspired by http://stackoverflow.com/questions/5382768/testing-a-bash-shell-script
#inspired by http://comments.gmane.org/gmane.org.user-groups.linux.aklug.general/9954
#######################################################################
                 #  If condition false,
                         #+ exit from script
                          #+ with appropriate error message.
  E_PARAM_ERR=98
  E_ASSERT_FAILED=99


  if [ -z "$2" ]          #  Not enough parameters passed
  then                    #+ to assert() function.
    return $E_PARAM_ERR   #  No damage done.
  fi

  lineno=$2

  if [ ! $1 ] 
  then
    echo "Assertion failed:  \"$1\""
    echo "File \"$0\", line $lineno"    # Give name of file and line number.
    exit $E_ASSERT_FAILED
  # else
  #   return
  #   and continue executing the script.
  fi  
#######################################################################
