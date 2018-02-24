from PIL import Image


img = Image.open('test/test.jpg')
img = img.convert('L')
# img.show()
print(list(img.getdata()))
print(img)