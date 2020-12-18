from matplotlib import font_manager
font_manager._rebuild()
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


class chart:
    def __init__(self):
        ScalarFormatter().set_scientific(False)
        font = 'NanumSquareRound, AppleGothic, Malgun Gothic, DejaVu Sans'
        plt.style.use('seaborn')
        plt.rcParams['font.family'] = font
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.grid'] = True
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.7
        plt.rcParams['lines.antialiased'] = True
        plt.rcParams['figure.figsize'] = [10.0, 5.0]
        plt.rcParams['savefig.dpi'] = 96
        plt.rcParams['font.size'] = 12
        plt.rcParams['legend.fontsize'] = 'medium'
        plt.rcParams['figure.titlesize'] = 'medium'

