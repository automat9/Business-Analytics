# add new key-value pair
student = {'name': 'Alice', 'age': 25, 'grade': 'A'}
student['major'] = 'Computer Science'

# print fruits with value > 2
fruits = {'apple': 3, 'banana': 5, 'orange': 2, 'cherry': 3}
for fruit, quantity in fruits.items():
    if quantity > 2:
        print(fruit)
