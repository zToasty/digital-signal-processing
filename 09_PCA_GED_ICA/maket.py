#!/usr/bin/env python
# coding: utf-8
"""
ЛАБОРАТОРНАЯ РАБОТА: МЕТОДЫ РАЗДЕЛЕНИЯ ИСТОЧНИКОВ
Цель: Реализовать PCA, GED и ICA для разделения пространственно-временных сигналов.
Заполните все блоки с пометкой # <-- и запустите код.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.decomposition import FastICA

# ======================== НАСТРОЙКИ ========================
np.random.seed(42)
N_CHANNELS = 10
N_TIMEPOINTS = 500
TIME_AXIS = np.arange(N_TIMEPOINTS)
AMP_1, FREQ_1 = 2.0, 0.05
AMP_2, FREQ_2 = 1.0, 0.10
PLOT_LIMIT = 100
VERT_OFFSET = 2.5

# Истинные паттерны (не менять)
spatial_1 = np.sin(np.linspace(0, 2*np.pi, N_CHANNELS))
spatial_2 = np.cos(np.linspace(0, 2*np.pi, N_CHANNELS))
temporal_1 = AMP_1 * np.sin(2*np.pi*FREQ_1*TIME_AXIS)
temporal_2 = AMP_2 * np.sin(2*np.pi*FREQ_2*TIME_AXIS)

def generate_data(noise_level):
    clean = np.outer(spatial_1, temporal_1) + np.outer(spatial_2, temporal_2)
    noise = np.random.randn(N_CHANNELS, N_TIMEPOINTS) * noise_level
    return clean + noise, clean, noise

def normalize_component(weights, temporal_pc, ref_signal=None, target_amp=None):
    scale = np.max(np.abs(weights))
    w_norm, t_norm = weights/scale, temporal_pc*scale
    if ref_signal is not None:
        if np.corrcoef(t_norm, ref_signal)[0,1] < 0:
            w_norm, t_norm = -w_norm, -t_norm
    if target_amp is not None and np.max(np.abs(t_norm)) > 0:
        t_norm = t_norm / np.max(np.abs(t_norm)) * target_amp
    return w_norm, t_norm

def plot_results(noisy_data, components, spatial_weights, titles, plot_limit=100):
    n_chan = noisy_data.shape[0]
    fig, ax = plt.subplots(figsize=(8,3))
    for ch in range(min(5, n_chan)):
        ax.plot(TIME_AXIS[:plot_limit], noisy_data[ch,:plot_limit] + ch*VERT_OFFSET, lw=0.8, alpha=0.7)
    ax.set_xlabel('Время'); ax.set_ylabel('Канал'); ax.set_title('Исходные данные'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    
    fig, ax = plt.subplots(figsize=(6,4))
    for w, lbl in zip(spatial_weights[:3], titles[:3]):
        ax.plot(np.arange(len(w)), w, marker='o', ms=3, label=lbl)
    ax.set_xlabel('Канал'); ax.set_ylabel('Вес'); ax.set_title('Пространственные паттерны'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    
    fig, ax = plt.subplots(figsize=(8,4))
    for t, lbl in zip(components[:3], titles[:3]):
        ax.plot(TIME_AXIS[:plot_limit], t[:plot_limit], label=lbl, lw=1.5)
    ax.plot(TIME_AXIS[:plot_limit], temporal_1[:plot_limit], 'k--', label='Источник 1', alpha=0.7)
    ax.plot(TIME_AXIS[:plot_limit], temporal_2[:plot_limit], 'r--', label='Источник 2', alpha=0.7)
    ax.set_xlabel('Время'); ax.set_ylabel('Амплитуда'); ax.set_title('Временные компоненты'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# ======================== ЗАДАНИЕ 1: PCA ========================
print("\n=== ЗАДАНИЕ 1: PCA ===")
NOISE_1 = 0.3
noisy_1, clean_1, noise_1 = generate_data(NOISE_1)

# <-- Центрирование: вычесть среднее по времени (ось=1) для каждого канала
data_centered_1 = None

# <-- Ковариационная матрица (каналы × каналы)
cov_matrix_1 = None

# <-- Собственные значения и векторы (np.linalg.eigh)
eigenvalues_1, eigenvectors_1 = None, None

# <-- Отсортируйте по убыванию eigenvalues и переставьте eigenvectors
sorted_idx_1 = None

# <-- Проекция на главные компоненты (eigenvectors.T @ data_centered)
principal_components_1 = None

# Визуализация
w1,t1 = normalize_component(eigenvectors_1[:,0], principal_components_1[0], temporal_1, AMP_1)
w2,t2 = normalize_component(eigenvectors_1[:,1], principal_components_1[1], temporal_2, AMP_2)
w3,t3 = normalize_component(eigenvectors_1[:,2], principal_components_1[2])
plot_results(noisy_1, [t1,t2,t3], [w1,w2,w3], ['PCA1','PCA2','PCA3'])

# <-- Вычислите корреляции t1 с temporal_1 и t2 с temporal_2
corr1_pca = None
corr2_pca = None
print(f"PCA: corr1 = {corr1_pca:.4f}, corr2 = {corr2_pca:.4f}")


# ======================== ЗАДАНИЕ 2: GED ========================
print("\n=== ЗАДАНИЕ 2: GED ===")
NOISE_2 = 0.8
noisy_2, clean_2, noise_2 = generate_data(NOISE_2)

data_c_2 = noisy_2 - noisy_2.mean(axis=1, keepdims=True)
noise_c_2 = noise_2 - noise_2.mean(axis=1, keepdims=True)

# <-- Ковариация сигнал+шум и ковариация шума
cov_sig_2 = None
cov_noise_2 = None

# <-- Регуляризация шумовой ковариации: cov_noise_reg = cov_noise + 1e-8 * trace/n_chan * I
reg = 1e-8 * np.trace(cov_noise_2) / N_CHANNELS
cov_noise_reg_2 = None

# <-- Обобщённое собственное разложение: scipy.linalg.eigh(cov_sig, cov_noise_reg)
evals_2, evecs_2 = None, None

# <-- Отсортируйте evals по убыванию, переставьте evecs и вычислите временные компоненты comp_ts = evecs.T @ data_c_2
comp_ts_2 = None

# Форвард-модель: a = (C_sig @ w) / (w.T @ C_noise @ w)
def compute_forward(w, C_sig, C_noise):
    denom = w.T @ C_noise @ w
    if np.abs(denom) < 1e-12:
        return np.zeros_like(w)
    return (C_sig @ w) / denom

C_sig_diff_2 = cov_sig_2 - cov_noise_2
f1 = compute_forward(evecs_2[:,0], C_sig_diff_2, cov_noise_reg_2)
f2 = compute_forward(evecs_2[:,1], C_sig_diff_2, cov_noise_reg_2)

w1g,t1g = normalize_component(f1, comp_ts_2[0], temporal_1, AMP_1)
w2g,t2g = normalize_component(f2, comp_ts_2[1], temporal_2, AMP_2)
plot_results(noisy_2, [t1g,t2g,comp_ts_2[2]], [w1g,w2g,evecs_2[:,2]], ['GED1','GED2','GED3'])

# <-- Корреляции
corr1_ged = None
corr2_ged = None
print(f"GED: corr1 = {corr1_ged:.4f}, corr2 = {corr2_ged:.4f}")


# ======================== ЗАДАНИЕ 3: ICA ========================
print("\n=== ЗАДАНИЕ 3: ICA ===")
X_ica_input = noisy_2 - noisy_2.mean(axis=1, keepdims=True)

# <-- Подготовка: X_ica = X_ica_input.T (время, каналы)
X_ica = None

# <-- FastICA с n_components=N_CHANNELS, random_state=42, whiten='arbitrary-variance'
ica = None
sources = None  # результат fit_transform (время, компоненты)
mixing = None   # ica.mixing_

# <-- Преобразуйте sources.T -> (компоненты, время), spatial_maps = mixing.T
sources = None
spatial_maps = None

# <-- Сортировка по энергии spatial_maps (сумма квадратов по каналам)
energy = None
sorted_idx = None
spatial_maps = spatial_maps[sorted_idx]
sources = sources[sorted_idx]

# <-- Найдите индексы компонент, максимально коррелирующих с temporal_1 и temporal_2 (независимо)
idx1 = None
idx2 = None

t1i,s1i = normalize_component(spatial_maps[idx1], sources[idx1], temporal_1, AMP_1)
t2i,s2i = normalize_component(spatial_maps[idx2], sources[idx2], temporal_2, AMP_2)
plot_results(noisy_2, [sources[idx1], sources[idx2], sources[0]], 
             [s1i, s2i, spatial_maps[0]], [f'ICA{idx1+1}', f'ICA{idx2+1}', 'ICA-шум'])

# <-- Корреляции
corr1_ica = None
corr2_ica = None
print(f"ICA: corr1 = {corr1_ica:.4f}, corr2 = {corr2_ica:.4f}")


# ======================== ЗАДАНИЕ 4*: ЭЭГ (дополнительно) ========================
print("\n=== ЗАДАНИЕ 4*: ЭЭГ (дополнительно) ===")
try:
    import scipy.io as sio
    # <-- Загрузка данных, моделирование, GED, визуализация
    print("Выполнено (при наличии emptyEEG.mat и pytopo)")
except Exception as e:
    print(f"Пропущено: {e}")


# =============================================================================
# 0. ГЛОБАЛЬНЫЕ НАСТРОЙКИ (параметры моделирования)
# =============================================================================
MATFILE_NAME = 'emptyEEG.mat'         # файл с leadfield и структурой EEG
DIPOLE_LOC1 = 108                     # номер первого диполя в leadfield
DIPOLE_LOC2 = 134                     # номер второго диполя
FREQ1 = 15                            # частота первого источника (Гц)
FREQ2 = 10                            # частота второго источника (Гц)
AMP1 = 2.0                            # амплитуда первого диполя
AMP2 = 1.0                            # амплитуда второго диполя
N_COMPONENTS = 2                      # количество извлекаемых GED-компонент
REG_PARAM = 1e-8                      # параметр регуляризации ковариации шума

# =============================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =============================================================================
matfile = sio.loadmat(MATFILE_NAME)
lf = matfile['lf'][0, 0]              # leadfield структура
EEG = matfile['EEG'][0, 0]            # структура EEG

# Убедимся, что все необходимые поля присутствуют
assert 'Gain' in lf.dtype.names, "Leadfield не содержит поля Gain"
assert 'chanlocs' in EEG.dtype.names, "EEG не содержит chanlocs"

# Принудительно задаём параметры моделирования (можно брать из файла, но для надёжности задаём)
EEG['srate'] = 500                    # частота дискретизации (Гц)
EEG['trials'] = 200                   # количество эпох
EEG['pnts'] = 1000                    # отсчётов на эпоху
EEG['times'] = np.arange(EEG['pnts']) / EEG['srate']

# Индекс "момента стимула" – середина временной оси
tidx = np.argmin(np.abs(EEG['times'] - np.mean(EEG['times'])))
n_pre = tidx                          # длина пре-стимульного интервала
n_pst = EEG['pnts'] - tidx            # длина пост-стимульного интервала

# Размеры leadfield: (каналы, ориентации?, диполи)
# Обычно lf['Gain'] имеет форму (каналы, 3, диполи). Берём только радиальную компоненту (индекс 0)
Gain = lf['Gain'][:, 0, :]            # матрица размером (chan x dipoles)

n_chan, n_dip = Gain.shape

# Инициализация массива данных ЭЭГ: (каналы × время × эпохи)
EEG['data'] = np.zeros((n_chan, EEG['pnts'], EEG['trials']))

# Массив для хранения истинной активности диполей во всех эпохах (для последующего сравнения)
true_dipoles = np.zeros((EEG['trials'], EEG['pnts'], n_dip))

# =============================================================================
# 2. ГЕНЕРАЦИЯ АКТИВНОСТИ ИСТОЧНИКОВ И ПРОЕКЦИЯ НА СКАЛЬП
# =============================================================================
# Временная ось после стимула
t_post = EEG['times'][tidx:]
omega1 = 2 * np.pi * FREQ1 * t_post    # угловая частота диполя 1
omega2 = 2 * np.pi * FREQ2 * t_post    # угловая частота диполя 2

# ======================== ИТОГ ========================
print("\n=== РЕЗУЛЬТАТЫ ===")
print("Метод | corr1 | corr2")
print(f"PCA   |  {corr1_pca:.4f}  |  {corr2_pca:.4f}")
print(f"GED   |  {corr1_ged:.4f}  |  {corr2_ged:.4f}")
print(f"ICA   |  {corr1_ica:.4f}  |  {corr2_ica:.4f}")

print("\nВопросы для отчёта:")
print("1. Какой метод лучше при высоком шуме и почему?")
print("2. Когда использовать PCA, GED или ICA?"

plt.show()