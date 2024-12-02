# Backwards
text = 'Hello'
print(text[::-1])

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
