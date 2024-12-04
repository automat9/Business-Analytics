#1
numbers = [1,2,3,4,5]
new_numbers = [num ** 2 for num in numbers]
print(new_numbers) # changes to squared

#2
numbers = [10,15,20,25,30]
even = [num for num in numbers if num % 2 == 0]
print(even) # even only

#3
words = ['apple', ' banana', 'cherry', 'date']
length = [len(word) for word in words]
print(length)

#4
numbers = [5,10,15,20,25]
multiplied = [num * index for index, num in enumerate(numbers)]
print(multiplied) # multiplied elements by their respective indices
