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

# printing every second element
# range(start,end,step)
numbers = [10, 20, 30, 40, 50, 60, 70]
every_second = []
for num in range(0, len(numbers), 2):
    every_second.append(numbers[num])
print(every_second)
