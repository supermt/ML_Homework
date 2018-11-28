import numpy as np
import itertools

def verify(polyx,matA,order):
  targety=[]
  for targetx in polyx:
    result = 0
    for k in range(0,order):
      # y = a0 + a1 * x + ... ak * x^k
      result = matA[k] * (targetx**k) + result
    targety.append(result[0])
  return targety

def RLS(matFi,vectY,lambda_arg=0):
  fifiT = np.array(np.dot(matFi,matFi.T))
  origin = fifiT + lambda_arg * np.eye(len(fifiT[0]))
  firstPart = np.linalg.inv(origin)
  midpart = np.dot(firstPart,matFi)
  result = midpart.dot(vectY)
  return result


def LASSO(X, y, lambd=0.2, threshold=0.1):
  rss = lambda X, y, w: (y - X*w).T*(y - X*w)
  m, n = X.shape
  w = np.matrix(np.zeros((n, 1)))
  r = rss(X, y, w)
  niter = itertools.count(1)
  for it in niter:
      for k in range(n):
          z_k = (X[:, k].T*X[:, k])[0, 0]
          p_k = 0
          for i in range(m):
              p_k += X[i, k]*(y[i, 0] - sum([X[i, j]*w[j, 0] for j in range(n) if j != k]))
          if p_k < -lambd/2:
              w_k = (p_k + lambd/2)/z_k
          elif p_k > lambd/2:
              w_k = (p_k - lambd/2)/z_k
          else:
              w_k = 0
          w[k, 0] = w_k
      r_prime = rss(X, y, w)
      delta = abs(r_prime - r)[0, 0]
      r = r_prime
      if delta < threshold:
          break
  return w

def RR(X, y, lambd=0.2):
  XTX = X.T*X
  m, _ = XTX.shape
  I = np.matrix(np.eye(m))
  w = (XTX + lambd*I).I*X.T*y
  return w