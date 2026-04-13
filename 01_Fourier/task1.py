import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fourier_tools import FourierSeries

AMPLITUDE = 2.0
PERIOD = 2.0
N_HARMONICS = 10

def rect_pulse(t):
    return AMPLITUDE * signal.square(2 * np.pi * t / PERIOD)

def main():
    # Создаем объект разложения Фурье
    fs = FourierSeries(rect_pulse, PERIOD, N_HARMONICS)
    
    # Считаем коэффициенты (a0, an, bn)
    fs.calculate_coefficients()
    print(f"a0 = {fs.a0:.4f}")

    t_values = np.linspace(-4, 4, 1000)

    original_signal = rect_pulse(t_values)

    approx_signal = fs.reconstruct(t_values)

    error_signal = original_signal - approx_signal

    plt.figure(figsize=(10, 8))

    # Верхний график
    plt.subplot(2, 1, 1)
    plt.plot(t_values, original_signal, label='Исходный x(t)', color='gray', linewidth=2, alpha=0.6)
    plt.plot(t_values, approx_signal, label=f'Аппроксимация x*(t), N={N_HARMONICS}', color='orange', linewidth=2)
    plt.title(f'Разложение прямоугольного импульса (T={PERIOD}, A={AMPLITUDE})')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend(loc='upper right')

    # Нижний график
    plt.subplot(2, 1, 2)
    plt.plot(t_values, error_signal, label='Ошибка ε = x(t) - x*(t)', color='tab:blue')
    plt.title('Ошибка приближения')
    plt.xlabel('Время, с')
    plt.ylabel('Ошибка')
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()