
# implement and test some of the regression methods

## Regression methods list

1. least-squares (LS)
2. regularized LS (RLS)
3. L1-regularized LS (LASSO)
4. robust regression (RR)
5. Bayesian regression (BR)

## Implement Discussion

### Environments setting

this program is wrote by Python2.7, using package numpy and scipy as the utils of matrix calculating and data import. 

### Calculating Implements

1. Least-Squares Regression

<html>
<div align="center">

![](2018-10-06-14-58-37.png)

![](2018-10-06-14-58-54.png)
![](2018-10-06-14-59-08.png)

</div>
</html>
According to the LSR algorithm, we should figure out the vector A = [a0...ak], which can be calculated by solving the XA = Y. Here is the implements

> first we need fill the three matrix,called matX and matY

```python
matX = []

for k in range(0,order):
  # vector = [sigma x^k, sigma x^k+1 ....  sigma x^k+order]
  vectX = []
  for column in range(k,k+order):
    result = 0
    for x in sampx:
      result = x**column + result
    vectX.append(result)
  matX.append(vectX)

matX = np.array(matX)
print matX
```

```python
# Y = [sigma x^0 * y, sigma x^1 * y ....  sigma x^k * y]
matY = []
for k in range(0,order):
  result = 0
  for index in range(0,len(sampx)):
    result = result + (sampx[index]**k) * sampy[index]
  matY.append(result)
```

> than we use matX and matY to figure out what is matA

```python
matA = np.linalg.solve(matX,matY)
```

> and then draw the lines and points

```python
fig = plt.figure()
ax = fig.add_subplot(111)
# sample points
ax.plot(sampx,sampy,color='r',linestyle='',marker='.')
# regression line
ax.plot(polyx,targety,color='g',linestyle='-',marker='')
# poly points
ax.plot(polyx,polyy,color='b',linestyle='',marker='*')

ax.legend()
plt.show()
```

> and here is the resuilt figure, the red dots are the samp data, the green line is the regression function polt and the blue stars are the poly data.

![](2018-10-06-15-25-59.png)

2. RLS

according to the material, we need figure out the following evalution

![](2018-10-06-15-37-45.png)

while 

![](2018-10-06-15-38-35.png) and ![](2018-10-06-15-38-50.png)

3. LASSO

![](2018-10-06-16-57-50.png)