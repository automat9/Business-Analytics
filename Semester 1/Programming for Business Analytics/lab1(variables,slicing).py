### Q1
# Assign the following variables in python (use variable names ‘a’,’b’,’c’,’d’).
# Which data types are produced? (check with 'type')”
#	1
#	2.0
#	‘Exeter’
#	‘3.0’

# Enter your code below here:
a = 1
b = 2.0
c = 'Exeter'
d = '3.0'

### Q2
# Assign a variable: my_string = 'Today we are learning about variables and data-types'. 
# Extract  (using slicing syntax) the following strings from 'my_string': 
#	'Today'
#	'data-types'
#	'learning about variables'
#	'sepyt-atad dna selbairav tuoba gninrael era ew yadoT' (backwards!)
#	'yadoT'
#	 'Tdyw r erigaotvralsaddt-ye' (?!?)

# Enter your code below here:
# Key = [start:end:step]
my_string = 'Today we are learning about variables and data-types'
print(my_string[:5])
print(my_string[-10:])
print(my_string[13:37])
print(my_string[::-1])
print(my_string[5::-1])
print(my_string[::2])
