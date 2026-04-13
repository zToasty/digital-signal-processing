import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy import signal
from fourier_tools import FourierSeries

def main():
    # --- 1. Параметры сигнала ---
    A = 2.0       # Амплитуда
    T = 2.0       # Период (2 секунды -> частота основной гармоники f = 0.5 Гц)
    f_base = 1.0 / T # 0.5 Гц
    
    # Функция прямоугольного импульса (как в Задании 1)
    def rect_func(t):
        return A * signal.square(2 * np.pi * t / T)

    # --- 2. Генерация данных для FFT ---
    fs = 1000     # Частота дискретизации (точек в секунду)
    duration = 40.0 # Длительность (возьмем побольше периодов для четкого спектра)
    
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    x = rect_func(t)
    
    # --- ЗАДАНИЕ 4: Добавление ШУМА ---
    # Раскомментируй строку ниже, чтобы добавить шум
    x = x + np.random.normal(0, 0.5, size=len(x)) 
    
    # --- 3. Вычисление FFT (Быстрое преобразование Фурье) ---
    N_points = len(t)
    yf = rfft(x)
    xf = rfftfreq(N_points, 1 / fs)
    
    # Нормированный спектр амплитуд
    fft_spectrum = 2.0 / N_points * np.abs(np.asarray(yf))

    # --- 4. Вычисление через НАШ класс ---
    # Мы хотим проверить первые 15 гармоник
    N_harmonics = 15
    fs_class = FourierSeries(rect_func, T, N_harmonics)
    
    print("Считаю коэффициенты через наш класс...")
    fs_class.calculate_coefficients()
    
    # Для сравнения нам нужна полная амплитуда гармоники: C_n = sqrt(an^2 + bn^2)
    our_freqs = [n * f_base for n in range(1, N_harmonics + 1)]
    our_amps = []
    for n in range(N_harmonics):
        an = fs_class.a_n[n]
        bn = fs_class.b_n[n]
        cn = np.sqrt(an**2 + bn**2) # Полная амплитуда
        our_amps.append(cn)

    # --- 5. Построение графиков ---
    plt.figure(figsize=(12, 8))
    
    # График 1: Сам сигнал (кусочек)
    plt.subplot(2, 1, 1)
    plt.plot(t[:3000], x[:3000]) # Показываем только начало, чтобы было видно форму
    plt.title('Сигнал (первые 0.5 сек)')
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    # График 2: Спектр
    plt.subplot(2, 1, 2)
    
    # Рисуем FFT (черная линия)
    plt.plot(xf, fft_spectrum, color='black', label='Спектр (FFT)', linewidth=1)
    
    # Рисуем наши точки (красные кружки)
    plt.stem(our_freqs, our_amps, linefmt='--r', markerfmt='ro', basefmt=" ", label='Наш алгоритм')
    
    # Настройки отображения спектра
    plt.xlim(0, 8) # Покажем частоты от 0 до 8 Гц (гармоники будут на 0.5, 1.5, 2.5...)
    plt.title('Спектр прямоугольного сигнала')
    plt.xlabel('Частота, Гц')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()