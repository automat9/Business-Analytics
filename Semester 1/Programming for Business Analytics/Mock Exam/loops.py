# for loop
#Q1
for i in range(1,21):
    print(i)

#Q2
text = 'Hello World'
for character in text:
    print(character)

#Q3
list = [3, 5, 7, 9, 11]
total = 0
for i in list:
    total += i
print(total)

# while loop
#Q1
i = 1
while i <= 10:
    print(i)
    i += 1

#Q2
while True:
    user_input = int(input("Input a number"))
    if user_input == 5:
        break
    print('Continue')

#3
x = 10
while x >= 1:
    print(x)
    x = x/2

# sum of multiples of 3 between 1 and 50
total = 0
for i in range(1,51):
    if i % 3 == 0:
        total +=i
print(total)

# vertical output
words = ["hello", "world", "python", "rocks"]
for word in words:
    print(word[::-1])
    
# [... ... ...] output
words = ["hello", "world", "python", "rocks"]
reversed = [word[::-1] for word in words]
print(reversed)


# print total in range 10
total = 0
for i in range(1,11):
    total += i
print(total)

# multiple of 5
i = 1
while i <= 50:
    if i % 5 == 0:
        print(i)
    i += 1
#or 
for i in range(1,51):
    if i % 5 ==0:
        i
print(i)
