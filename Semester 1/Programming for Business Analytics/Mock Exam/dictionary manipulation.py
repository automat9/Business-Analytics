# add new key-value pair
student = {'name': 'Alice', 'age': 25, 'grade': 'A'}
student['major'] = 'Computer Science'

# print fruits with value > 2
fruits = {'apple': 3, 'banana': 5, 'orange': 2, 'cherry': 3}
for fruit, quantity in fruits.items():
    if quantity > 2:
        print(fruit)

# update value
grades = {'Alice': 85, 'Bob': 90, 'Carol': 88}
grades['Bob'] = 95
print(grades)

# remove a key
inventory = {'apple': 10, 'banana': 15, 'cherry': 20}
del inventory['banana']
print(inventory)
