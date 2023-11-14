from copy import deepcopy
from scipy import special
from scipy.sparse.linalg import cg
from polynomials import *
import numpy as np
import pandas as pd
from copy import deepcopy

class Solve(object):
    def __init__(self,d):
        self.deg = d['dimensions']
        self.filename_input = d['input_file']
        self.p = list(map(lambda x:x+1,d['degrees'])) 
        self.weights = d['weights']
        self.poly_type = d['poly_type']
        self.splitted_lambdas = d['lambda_multiblock']
        self.eps = 1e-6
        self.norm_error = 0.0
        self.error = 0.0

    def define_data(self):
        self.datas = np.fromstring(self.filename_input, sep='\t').reshape(-1, sum(self.deg))
        self.n = len(self.datas)
        self.degf = [sum(self.deg[:i + 1]) for i in range(len(self.deg))]

    def _minimize_equation(self, A, b):
        return coordinate_descent_method(A.T @ A, A.T @ b, eps=self.eps)

    def norm_data(self):
        n,m = self.datas.shape
        vec = np.ndarray(shape=(n,m),dtype=float)
        for j in range(m):
            minv = np.min(self.datas[:,j])
            maxv = np.max(self.datas[:,j])
            for i in range(n):
                vec[i,j] = (self.datas[i,j] - minv)/(maxv - minv)
        self.data = np.array(vec)

    def define_norm_vectors(self):
        X1 = self.data[:, :self.degf[0]]
        X2 = self.data[:, self.degf[0]:self.degf[1]]
        X3 = self.data[:, self.degf[1]:self.degf[2]]
      
        self.X = [X1, X2, X3]
        self.mX = self.degf[2]
        self.Y = self.data[:, self.degf[2]:self.degf[3]]
        self.Y_ = self.datas[:, self.degf[2]:self.degf[3]]
        self.X_ = [self.datas[:, :self.degf[0]], self.datas[:,self.degf[0]:self.degf[1]],
                   self.datas[:, self.degf[1]:self.degf[2]]]

    def built_B(self):
        def B_average():         
            return np.tile((self.Y.max(axis=1) + self.Y.min(axis=1))/2, (self.deg[3], 1)).T

        def B_scaled():
            return deepcopy(self.Y)

        if self.weights == 'Середнє арифметичне':
            self.B = B_average()
        elif self.weights =='Нормоване значення':
            self.B = B_scaled()
        else:
            exit('B not definded')

    def poly_func(self):
        if self.poly_type =='Чебишова':
            self.poly_f = special.eval_sh_chebyt
        elif self.poly_type == 'Лежандра':
            self.poly_f = special.eval_sh_legendre
        elif self.poly_type == 'Лаґерра':
            self.poly_f = special.eval_laguerre
        elif self.poly_type == 'Ерміта':
            self.poly_f = special.eval_hermite

    def built_A(self):
        def coordinate(v,deg):
            c = np.ndarray(shape=(self.n,1), dtype = float)
            for i in range(self.n):
                c[i,0] = self.poly_f(deg, v[i])
            return c

        def vector(vec, p):
            n, m = vec.shape
            a = np.ndarray(shape=(n,0),dtype = float)
            for j in range(m):
                for i in range(p):
                    ch = coordinate(vec[:,j],i)
                    a = np.append(a,ch,1)
            return a

        A = np.ndarray(shape = (self.n,0),dtype =float)
        for i in range(len(self.X)):
            vec = vector(self.X[i],self.p[i])
            A = np.append(A, vec,1)
        self.A = np.array(A)

    def lamb(self):
        lamb = np.ndarray(shape = (self.A.shape[1],0), dtype = float)
        for i in range(self.deg[3]):
            if self.splitted_lambdas:
                boundary_1 = self.p[0] * self.deg[0]
                boundary_2 = self.p[1] * self.deg[1] + boundary_1
                lamb1 = self._minimize_equation(self.A[:, :boundary_1], self.B[:, i])
                lamb2 = self._minimize_equation(self.A[:, boundary_1:boundary_2], self.B[:, i])
                lamb3 = self._minimize_equation(self.A[:, boundary_2:], self.B[:, i])
                lamb = np.append(lamb, np.concatenate((lamb1, lamb2, lamb3)), axis=1)
            else:
                lamb = np.append(lamb, self._minimize_equation(self.A, self.B[:, i]), axis=1)
        self.Lamb = np.array(lamb)

    def psi(self):
        def built_psi(lamb):
            psi = np.ndarray(shape=(self.n, self.mX), dtype = float)
            q = 0
            l = 0 
            for k in range(len(self.X)): 
                for s in range(self.X[k].shape[1]):
                    for i in range(self.X[k].shape[0]):
                        psi[i,l] = self.A[i,q:q+self.p[k]] @ lamb[q:q+self.p[k]]
                    q += self.p[k]
                    l += 1
            return np.array(psi)

        self.Psi = [] 
        for i in range(self.deg[3]):
            self.Psi.append(built_psi(self.Lamb[:,i]))

    def built_a(self):
        self.a = np.ndarray(shape=(self.mX,0), dtype=float)
        for i in range(self.deg[3]):
            a1 = self._minimize_equation(self.Psi[i][:, :self.degf[0]], self.Y[:, i])
            a2 = self._minimize_equation(self.Psi[i][:, self.degf[0]:self.degf[1]], self.Y[:, i])
            a3 = self._minimize_equation(self.Psi[i][:, self.degf[1]:], self.Y[:, i])
            self.a = np.append(self.a, np.vstack((a1, a2, a3)),axis = 1)

    def built_F1i(self, psi, a):
            m = len(self.X) 
            F1i = np.ndarray(shape = (self.n,m),dtype = float)
            k = 0 
            for j in range(m): 
                for i in range(self.n): 
                    F1i[i,j] = psi[i,k:self.degf[j]] @ a[k:self.degf[j]]
                k = self.degf[j]
            return np.array(F1i)

    def built_Fi(self):
        self.Fi = []
        for i in range(self.deg[3]):
            self.Fi.append(self.built_F1i(self.Psi[i],self.a[:,i]))

    def built_c(self):
        self.c = np.ndarray(shape = (len(self.X),0),dtype = float)
        for i in range(self.deg[3]):
            self.c = np.append(self.c, coordinate_descent_method(self.Fi[i].T @ self.Fi[i], self.Fi[i].T @ self.Y[:,i], eps=self.eps), axis = 1)

    def built_F(self):
        F = np.ndarray(self.Y.shape, dtype = float)
        for j in range(F.shape[1]):
            for i in range(F.shape[0]): 
                F[i,j] = self.Fi[j][i,:] @ self.c[:,j]
        self.F = np.array(F)
        self.norm_error = np.abs(self.Y - self.F).max(axis=0).tolist()

    def built_F_(self):
        minY = self.Y_.min(axis=0)
        maxY = self.Y_.max(axis=0)
        self.F_ = np.multiply(self.F,maxY - minY) + minY
        self.error = np.abs(self.Y_ - self.F_).max(axis=0).tolist()

    def show_streamlit(self):
        res = []
        res.append(('Вихідні дані',
            pd.DataFrame(self.datas, 
            columns = [f'X{i+1}{j+1}' for i in range(3) for j in range(self.deg[i])] + [f'Y{i+1}' for i in range(self.deg[-1])],
            index = np.arange(1, self.n+1))
        ))
        res.append(('Нормовані вихідні дані',
            pd.DataFrame(self.data, 
            columns = [f'X{i+1}{j+1}' for i in range(3) for j in range(self.deg[i])] + [f'Y{i+1}' for i in range(self.deg[-1])],
            index = np.arange(1, self.n+1))
        ))

        res.append((r'Матриця $\|\lambda\|$',
            pd.DataFrame(self.Lamb)
        ))
        res.append((r'Матриця $\|a\|$',
            pd.DataFrame(self.a)
        ))
        res.append((r'Матриця $\|c\|$',
            pd.DataFrame(self.c)
        ))

        for j in range(len(self.Psi)):
            res.append((r'Матриця $\|\Psi_{}\|$'.format(j+1),
            pd.DataFrame(self.Psi[j])
        ))
        for j in range(len(self.Fi)):
            res.append((r'Матриця $\|\Phi_{}\|$'.format(j+1),
            pd.DataFrame(self.Fi[j])
        ))
    
        df = pd.DataFrame(self.norm_error).T
        df.columns = np.arange(1, len(self.norm_error)+1)
        res.append((r'Нормалізована похибка',
            df
        ))
        df = pd.DataFrame(self.error).T
        df.columns = np.arange(1, len(self.error)+1)
        res.append((r'Похибка',
            df
        ))
        return res

    def prepare(self):
        func_runtimes = {}
        self.define_data()
        self.norm_data()
        self.define_norm_vectors()
        self.built_B()
        self.poly_func()
        self.built_A()
        self.lamb()
        self.psi()
        self.built_a()
        self.built_Fi()
        self.built_c()
        self.built_F()
        self.built_F_()
        return func_runtimes

def coordinate_descent_method(A, b, eps, max_iter=1000):
    if np.abs(np.linalg.det(A)) < 1e-15:
        return cg(A, b, tol=eps)[0].reshape(-1, 1)

    n = len(b)
    x = np.random.randn(n)
    gradient_i = [0] * n
    
    for _ in range(max_iter):
        for i in range(n):
            gradient_i[i] = A[i] @ x - b[i]
            x[i] = (b[i] - A[i] @ x + A[i, i] * x[i]) / A[i, i]
        if np.linalg.norm(gradient_i) < eps:
            break

    return x.reshape(-1, 1)