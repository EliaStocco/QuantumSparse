import matplotlib.pyplot as plt
from importlib import resources

def use_default_style(style:str="settings"):
    with resources.path("quantumsparse", f"{style}.mplstyle") as style_path:
        plt.style.use(style_path)