# [start:end:step]
string = 'This is a short sentence'
string[:4] # This
string[4:] #  is a short sentence
string[::2] # Ti sasotsnec
string[::-1] # ecnetnes trohs a si sihT
string[3::-1] # sihT

# Count vowels
text2 = "programming is fun"
vowels = 'aeiou'
count = 0
for vowel in vowels:
    count += text2.count(vowel)
print(count)

# Replace
text3 = "I love Python because Python is powerful."
replaced = text3.replace('Python', 'Javascript')
print(replaced)

# Find coordinates of word, if multiple instances use re
string = "Hello, World!"
word = 'World'
start_index = string.find(word)
end_index = start_index + len(word)
result = string[start_index:end_index]
print('The coordinates of your word are:', start_index, end_index)
