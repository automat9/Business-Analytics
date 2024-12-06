


# printing every second element
# range(start,end,step)
numbers = [10, 20, 30, 40, 50, 60, 70]
every_second = []
for num in range(0, len(numbers), 2):
    every_second.append(numbers[num])
print(every_second)
