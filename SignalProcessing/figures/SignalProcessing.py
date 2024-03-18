import numpy as np
from scipy import signal
from scipy import signal, fft
import matplotlib.pyplot as plt

# Генерація випадкового сигналу
a = 0  # середнє значення розподілу
b = 10  # стандартне відхилення розподілу
n = 1000  # кількість згенерованих елементів
random_signal = np.random.normal(a, b, n)

# Визначення значень часу
Fs = 1000  # частота дискретизації (відліків за секунду)
time_values = np.arange(n) / Fs

# Розрахунок параметрів фільтру
F_max = 20e3  # максимальна частота сигналу (20 кГц)
w = F_max / (Fs / 2)  # нормована частота

# Розрахунок фільтру низьких частот (ФНЧ)
order = 3  # порядок фільтру
w_normalized = w / Fs  # нормована частота
sos = signal.butter(order, w_normalized, 'low', output='sos')

# Фільтрація сигналу
filtered_signal = signal.sosfiltfilt(sos, random_signal)

# Відображення результатів
def plot_signal(time, signal, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
    ax.plot(time, signal, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    plt.savefig('./figures/' + title + '.png', dpi=600)

plot_signal(time_values, filtered_signal, 'Сигнал з максимально частотою Fmax=20 Гц', 'Час', 'Aмплітуда')
plt.show()

# Розрахунок спектру сигналу
spectrum = fft.fft(filtered_signal)
spectrum_shifted = fft.fftshift(spectrum)
frequency_bins = fft.fftfreq(len(filtered_signal), 1/Fs)
frequency_bins_shifted = fft.fftshift(frequency_bins)

# Відображення результатів
def plot_spectrum(frequency, spectrum, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
    ax.plot(frequency, np.abs(spectrum), linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    plt.savefig('./figures/' + title + '.png', dpi=600)

plot_spectrum(frequency_bins_shifted, spectrum_shifted, 'Спектор сигналу з максимально частотою Fmax=20Гц', 'Час', 'Амплітуда')
plt.show()






