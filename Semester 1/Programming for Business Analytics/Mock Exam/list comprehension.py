numbers = [1,2,3,4,5]
new_numbers = [num ** 2 for num in numbers]
print(new_numbers) # changes to squared

numbers = [10,15,20,25,30]
even = [num for num in numbers if num % 2 == 0]
print(even) # even only
