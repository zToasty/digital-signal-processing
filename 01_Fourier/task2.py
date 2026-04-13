import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from fourier_tools import FourierSeries  # Импортируем твой класс

def main():
    # --- 1. Параметры сигнала ---
    # По заданию: x(t) = A * cos(wt), частота f = 100 Гц
    A = 5.0          # Зададим амплитуду (например, 5)
    f_signal = 100.0 # Частота сигнала (Гц)
    T = 1.0 / f_signal # Период сигнала (0.01 сек)
    
    # Функция сигнала: A * cos(2 * pi * f * t)
    def cos_func(t):
        w = 2 * np.pi * f_signal
        return A * np.cos(w * t)

    # --- 2. Прямое преобразование Фурье (FFT) - Эталон ---
    # Чтобы получить красивый спектр, нужно взять много точек
    fs = 10000       # Частота дискретизации (точек в секунду)
    duration = 0.5   # Длительность сигнала для FFT (0.5 сек = 50 периодов)
    
    # Временная ось для FFT
    t_fft = np.linspace(0, duration, int(duration * fs), endpoint=False)
    x_fft = cos_func(t_fft)
    
    # Вычисляем FFT
    N_points = len(t_fft)
    
    # rfft - для вещественных сигналов (возвращает половину спектра, что нам и надо)
    yf = rfft(x_fft)
    xf = rfftfreq(N_points, 1 / fs)
    
    # Нормируем амплитуду: умножаем на 2/N, чтобы получить реальные Вольты/Амплитуду
    spectrum_fft = 2.0 / N_points * np.abs(np.asarray(yf))

    # --- 3. Расчет через НАШ класс (FourierSeries) ---
    # Мы хотим проверить, найдет ли он нашу частоту 100 Гц.
    # Посчитаем первые 5 гармоник (100, 200, 300, 400, 500 Гц)
    N_harmonics = 5
    
    # Создаем объект класса (передаем функцию, Период, число гармоник)
    fs_class = FourierSeries(cos_func, T, N_harmonics)
    fs_class.calculate_coefficients()
    
    # Подготовим данные для графика (наш алгоритм)
    # Частоты гармоник: 1*f, 2*f, 3*f...
    our_freqs = [n * f_signal for n in range(1, N_harmonics + 1)]
    # Амплитуды an (косинусные коэффициенты)
    our_amps = [abs(val) for val in fs_class.a_n]

    # Вывод в консоль для проверки
    print(f"Амплитуда сигнала A = {A}")
    print(f"Рассчитанный a1 (должен быть равен A): {fs_class.a_n[0]:.4f}")
    print(f"Остальные коэффициенты должны быть ~0")

    # --- 4. Построение графика (как на Рис. 1 в методичке) ---
    plt.figure(figsize=(10, 6))
    
    # а) Спектр FFT (сплошная линия)
    plt.plot(xf, spectrum_fft, color='black', label='Спектр (Numpy FFT)', linewidth=1.5)
    
    # б) Наши коэффициенты an (пунктирная оранжевая линия/точки)
    # Используем stem для красивого отображения палочками
    plt.stem(our_freqs, our_amps, linefmt='--r', markerfmt='ro', basefmt=" ", label='Наш алгоритм (an)')

    # Настройки графика
    plt.xlim(0, 500)  # Покажем область до 500 Гц
    plt.ylim(0, A + 1) # По оси Y чуть выше амплитуды
    plt.title('Спектр сигнала A*cos(wt), f=100 Гц')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()