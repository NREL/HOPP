import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_energy_profile(show_plots, save_plots):
    colors = ["#0079C2", "#F7A11A", "#5D9732", "#8CC63F", "#5E6A71", "#D1D5D8", "#933C06", "#D9531E"]
    paths = ["../ProFAST_financial_summary_results/Energy_Profile_TX_2030_6MW_Centralized_no policy_grid-only-retail-flat.csv",
                "../ProFAST_financial_summary_results/Energy_Profile_TX_2030_6MW_Centralized_no policy_hybrid-grid-retail-flat.csv",
                "../ProFAST_financial_summary_results/Energy_Profile_TX_2030_6MW_Centralized_no policy_off-grid.csv"]
    

    fig, ax = plt.subplots(3,1, figsize=(12,6))
    for path, axi in zip(paths, ax):
        df = pd.read_csv(path)
        df.drop(columns=["Unnamed: 0", "total_energy"], inplace=True)
        df = df.multiply(1E-3)
        df.plot(ax=axi, color=colors, ylabel="Power (MW)", xlim=[0,12000], ylim=[-100,1100], style=["-","-","-","--"])
        axi.legend(frameon=False)
    if show_plots:
        plt.show()
    if save_plots:
        plt.savefig("energy_profiles.pdf", transparent=True)

    return 0

if __name__ == "__main__":
    show_plots = True
    save_plots = False 
    plot_energy_profile(show_plots, save_plots)