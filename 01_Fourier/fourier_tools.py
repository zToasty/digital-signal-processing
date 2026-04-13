import numpy as np
from scipy import integrate

class FourierSeries:
    def __init__(self, function, period, num_harmonics):
        self.function = function
        self.T = period
        self.N = num_harmonics
        self.omega = 2 * np.pi / self.T  #  w = 2*pi/T

        # Списки для хранения коэффициентов
        self.a0 = 0
        self.a_n = []
        self.b_n = []

    def calculate_coefficients(self):
        t_start = -self.T / 2
        t_end = self.T / 2

        # 1. Считаем a0
        # Формула: (2/T) * интеграл( x(t) dt )
        result_a0, error = integrate.quad(self.function, t_start, t_end)
        self.a0 = (2 / self.T) * result_a0

        self.a_n = []
        self.b_n = []

        # 2. Считаем an и bn для каждого n от 1 до N
        for n in range(1, self.N + 1):
            # Вспомогательная функция для an: x(t) * cos(n*w*t)
            def integrand_a(t):
                return self.function(t) * np.cos(n * self.omega * t)
            
            # Вспомогательная функция для bn: x(t) * sin(n*w*t)
            def integrand_b(t):
                return self.function(t) * np.sin(n * self.omega * t)

            res_a, err_a = integrate.quad(integrand_a, t_start, t_end)
            res_b, err_b = integrate.quad(integrand_b, t_start, t_end)

            self.a_n.append((2 / self.T) * res_a)
            self.b_n.append((2 / self.T) * res_b)

    def reconstruct(self, t_values):
        
        # Начальное значение: a0 / 2
        # np.full_like создает массив той же длины, что t_values, заполненный числом
        result_signal = np.full_like(t_values, self.a0 / 2)

        # Добавляем гармоники в цикле
        for n in range(1, self.N + 1):
            idx = n - 1  # Индекс списка (т.к. n начинается с 1, а списки с 0)
            
            # Слагаемое: an * cos(nw t) + bn * sin(nw t)
            term = (self.a_n[idx] * np.cos(n * self.omega * t_values) +
                    self.b_n[idx] * np.sin(n * self.omega * t_values))
            
            result_signal += term
            
        return result_signal
    


