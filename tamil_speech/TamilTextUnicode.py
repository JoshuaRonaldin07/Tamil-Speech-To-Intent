import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
print("hello for Tamil")
word = "வணக்கம்" 
print("tamil word:", word)
print("Number of code points:", len(word))
print("lesson 1.1")
print('='*40)
print(f"Word:{word}")
print(f"total code points:{len(word)}")
print("="*40)

for i, char in enumerate(word):
    print(f"Position {i}:'{char}'")

print("lesson 1.2")
name = "Tamil"
count = 7
score = 95.5

# Without f-string (messy)
print("Language: " + name)

# With f-string (clean)
print(f"Language: {name}")
print(f"Code points: {count}")
print(f"Score: {score}")

# You can even do math inside {}
print(f"Double the count: {count * 2}")
print(f"Is count 7? {count == 7}")