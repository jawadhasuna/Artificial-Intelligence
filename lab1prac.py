lab1
import time
print(time.ctime())
#time.sleep(5)
print(time.ctime())
#variable
name='ali'
x=10
y=4.6
print(name)
print(x)
print(y)
print(type(name))
print(type(x))
print(type(y))
#casting
x=str(y)
print(x)
type(x)
#bool
print(10<9)
print(10==10)
a=20
b=30
if a>b:
  print("a>b")
else:
  print("a<b")
#list
color=['red','green','blue','white']
color1=list(('red','orange',"purple"))
print(color)
print(color1)
numb=list(range(1,10))
print(numb)
print(len(color))
print(len(numb))
print(color[0])
print(numb[2:9])
#list
color=['red','green','blue','white']
color1=list(('red','orange',"purple"))
print(color[1])
print(color[0:4])
if 'green' in color:
  print("yes")
else:
  print('no')
for i in color:
  print(i)
for j in range(len(color1)):
  print(color1[j])
k=0
while k<len(color):
  print(color[k])
  k+=1
#add
color[0]='pink'
color.append('skin')
print(color)
#remove
print(color.pop())
print(color)
color.remove('blue')
print(color)
del color[1]
print(color)
color.clear()
print(color)
#task1
i=1500
while i<=2700:
  if(i%7==0 and i%5==0):
    print(i)
  i+=1
result=[]
i=100
while i<=400:
  s=str(i)
  if int(s[0])%2==0 and int(s[1])%2==0 and int(s[2])%2==0:
    result.append(i)
  i+=1
for j in result:
  print(j)
  #Task2
result=[]
i=100
while i<=400:
  s=str(i)
  if int(s[0])%2==0 and int(s[1])%2==0 and int(s[2])%2==0:
    result.append(i)
  i+=1
print(result)
#Task3:
sum=0
k=0
while sum>-1:
  i=int(input("enter numbers"))
  if i==0:
    print("finish")
    print("sum:",sum)
    print("average:",sum/k)
    break
  sum=sum+i
  k+=1
  #task4:
classes=int(input("Number of classes held:"))
attended=int(input("Number of classes attended:"))
perc=(attended/classes)*100
print ("percentage of class attended:",perc)
if perc<75:
  print("student not allowed to sit in exam.")
else:
  print("allowed")

#task5
numbers=list(range(1,6))
print(numbers)
numbers.append(6)
numbers.insert(0,0)
print(numbers)
numbers.remove(3)
print(numbers)
