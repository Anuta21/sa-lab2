import numpy as np
from solve import Solve
from polynomials import *

class PolynomialBuilder(object):
    def __init__(self, solution):
        assert isinstance(solution, Solve)
        self._solution = solution
        max_degree = max(solution.p) - 1
        if solution.poly_type == 'Чебишова':
            self.symbol = 'T'
            self.basis = chebyshev(max_degree)
        elif solution.poly_type == 'Лежандра':
            self.symbol = 'P'
            self.basis = legendre(max_degree)
        elif solution.poly_type == 'Лаґерра':
            self.symbol = 'L'
            self.basis = laguerre(max_degree)
        elif solution.poly_type == 'Ерміта':
            self.symbol = 'H'
            self.basis = hermite(max_degree)
        self.a = solution.a.T.tolist()
        self.c = solution.c.T.tolist()
        self.minX = [X.min(axis=0).ravel() for X in solution.X_]
        self.maxX = [X.max(axis=0).ravel() for X in solution.X_]
        self.minY = solution.Y_.min(axis=0).ravel()
        self.maxY = solution.Y_.max(axis=0).ravel()

    def _form_lamb_lists(self):
        self.psi = list()
        for i in range(self._solution.Y.shape[1]):  
            psi_i = list()
            shift = 0
            for j in range(3):  
                psi_i_j = list()
                for k in range(self._solution.deg[j]): 
                    psi_i_j_k = self._solution.Lamb[shift:shift + self._solution.p[j], i].ravel()
                    shift += self._solution.p[j]
                    psi_i_j.append(psi_i_j_k)
                psi_i.append(psi_i_j)
            self.psi.append(psi_i)

    def _transform_to_standard(self, coeffs):
        std_coeffs = np.zeros(coeffs.shape)
        for index in range(coeffs.shape[0]):
            cp = self.basis[index].coef.copy()
            cp.resize(coeffs.shape)
            if type(coeffs) is np.matrix:
                std_coeffs += coeffs[index].getA1() * cp[0]
            else:
                std_coeffs += coeffs[index] * cp
        return std_coeffs.squeeze()

    def _print_psi_i_j_k(self, i, j, k):
        strings = list()
        for n in range(len(self.psi[i][j][k])):
            strings.append(r'{0:.6f}\cdot {symbol}_{{{deg}}}(x_{{{1}{2}}})'.format(
                self.psi[i][j][k][n], 
                j+1, k+1,symbol=self.symbol, deg=n
            ))
        res = ' + '.join(strings)
        return res.replace('+ -', '- ')

    def _print_phi_i_j(self, i, j):
        strings = list()
        for k in range(len(self.psi[i][j])):
            shift = sum(self._solution.deg[:j]) + k
            for n in range(len(self.psi[i][j][k])):
                strings.append(r'{0:.6f}\cdot {symbol}_{{{deg}}}(x_{{{1}{2}}})'.format(
                    self.a[i][shift] * self.psi[i][j][k][n],
                    j+1, k+1, symbol=self.symbol, deg=n
                ))
        res = ' + '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i(self, i):
        strings = list()
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                for n in range(len(self.psi[i][j][k])):
                    strings.append(r'{0:.6f}\cdot {symbol}_{{{deg}}}(x_{{{1}{2}}})'.format(
                        self.c[i][j] * self.a[i][shift] * self.psi[i][j][k][n],
                        j + 1, k + 1, symbol=self.symbol, deg=n
                    ))
        res = ' + '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i_transformed_denormed(self, i):
        strings = list()
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                raw_coeffs = self._transform_to_standard(self.c[i][j] * self.a[i][shift] * self.psi[i][j][k])
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = np.poly1d([1 / diff, - self.minX[j][k]] / diff)
                add_poly = np.poly1d([1])
                current_poly = np.poly1d([0])
                for n in range(len(raw_coeffs)):
                    current_poly += add_poly * raw_coeffs[n]
                    add_poly *= mult_poly
                current_poly = current_poly * (self.maxY[i] - self.minY[i]) + self.minY[i]
                current_poly = np.poly1d(current_poly.coeffs, variable='(x_{0}{1})'.format(j+1, k+1))
                strings.append(str(PolynomialPrint(
                    current_poly, 
                    symbol='(x_{0}{1})'.format(j+1, k+1),
                    subscr='{0}{1}'.format(j+1, k+1))))
        res = ' + '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i_transformed(self, i):
        strings = list()
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                current_poly = np.poly1d(self._transform_to_standard(self.c[i][j] * self.a[i][shift] *
                                                                     self.psi[i][j][k])[::-1],
                                         variable='(x_{0}{1})'.format(j+1, k+1))
                strings.append(str(PolynomialPrint(
                    current_poly, 
                    symbol='(x_{0}{1})'.format(j+1, k+1),
                    subscr='{0}{1}'.format(j+1, k+1))))
        res = ' + '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i_F_ij(self, i):
        res = ''
        for j in range(3):
            coef = self.c[i][j]
            if coef >= 0:
                res += f'+ {coef:.6f} \\cdot \\Phi_{{{i+1}{j+1}}} (x_{j+1}) '
            else:
                res += f'- {-coef:.6f} \\cdot \\Phi_{{{i+1}{j+1}}} (x_{j+1})'
        if self.c[i][0] >= 0:
            return res[2:-1]
        else:
            return res[:-1]

    def get_results(self):
        self._form_lamb_lists()
        psi_strings = [r'$\Psi_{{{1}{2}}}^{{[{0}]}}(x_{{{1}{2}}}) = {result}$'.format(i+1, j+1, k+1, result=self._print_psi_i_j_k(i, j, k)) + '\n'
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)
                       for k in range(self._solution.deg[j])]
        phi_strings = [r'$\Phi_{{{0}{1}}}(x_{{{1}}}) = {result}$'.format(i+1, j+1, result=self._print_phi_i_j(i, j)) + '\n'
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_strings = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(i + 1, result=self._print_F_i(i)) + '\n'
                     for i in range(self._solution.Y.shape[1])]
        f_strings_transformed = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(i + 1, result=self._print_F_i_transformed(i)) + '\n'
                                 for i in range(self._solution.Y.shape[1])]
        f_strings_transformed_denormed = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(
                                            i+1, result=self._print_F_i_transformed_denormed(i)) + '\n'
                                            for i in range(self._solution.Y.shape[1])]
        f_strings_from_f_ij = [r'$\Phi_{i}(x_1, x_2, x_3) = {result}$'.format(i=i+1, result=self._print_F_i_F_ij(i)) + '\n' 
                                for i in range(self._solution.Y.shape[1])]
        
        return '\n'.join(
            [r'$\Phi_i$ через $\Phi_{i1}(x_1)$, $\Phi_{i2}(x_2)$, $\Phi_{i3}(x_3)$:' + '\n'] + f_strings_from_f_ij +
            [r'$\Phi_i$' + f'через поліноми {self._solution.poly_type}:' + '\n'] + f_strings + 
            [r'$\Phi_i$ у звичайному вигляді (нормовані):' + '\n'] + f_strings_transformed + 
            [r'$\Phi_i$ у звичайному вигляді (відновлені):' + '\n'] + f_strings_transformed_denormed + 
            [r'Проміжні функції $\Phi$:' + '\n'] + phi_strings +
            [r'Проміжні функції $\Psi$:' + '\n'] + psi_strings)
    
class PolynomialPrint(object):
    def __init__(self, ar, symbol='x', subscr=None, eps=1e-6):
        self.ar = ar
        self.symbol = symbol
        self.subscr = subscr
        self.eps = eps

    def __repr__(self):
        joiner = {
            (True, True): '-',
            (True, False): '',
            (False, True): ' - ',
            (False, False): ' + '
        }

        result = []
        for deg, coef in reversed(list(enumerate(self.ar))):
            sign = joiner[not result, coef < 0]
            coef  = abs(coef)
            if coef == 1 and deg != 0:
                coef = ''
            if coef < self.eps:
                continue
            f = {0: '{}{:f}', 1: '{}{:f}'+self.symbol}.get(deg, '{}{:f}' + self.symbol + '^{{{}}}')
            res = f.format(sign, coef, deg)
            res = res.replace(self.symbol, r' x_{{{}}}'.format(self.subscr))
            result.append(res)
        return ''.join(result) or '0'