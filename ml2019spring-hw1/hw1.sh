#!/bin/bash
# Program:
#	User inputs his first name and last name.  Program shows his full name.
# History:
# 2015/07/16	VBird	First release
PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH

python model.py $1 $2

exit 0
