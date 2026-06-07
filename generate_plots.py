import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a professional plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 14
})

# 1. Generate penalty_chart.png
def make_penalty_chart():
    labels = ['Physics Baseline', 'PI-PML (XGBoost)']
    penalties = [2483254, 129282]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#e74c3c', '#2ecc71'] # Red for baseline, Green for ML
    bars = ax.bar(labels, penalties, color=colors, width=0.5, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('CERC Penalty (INR ₹)', fontweight='bold')
    ax.set_title('CERC DSM Penalty Comparison (2-Month Test Set)\nRSOPL Koppal Wind Plant (75 MW)', pad=15)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50000,
                f'₹ {height:,.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    ax.set_ylim(0, 3000000)
    # Format y-axis with commas
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    plt.tight_layout()
    plt.savefig('penalty_chart.png', dpi=300)
    plt.close()
    print("Generated penalty_chart.png successfully!")

# 2. Generate log_profile.png
def make_log_profile():
    # Parameters
    z_0 = 0.05  # Roughness length for Koppal hill terrain
    z_r = 10.0  # Reference height
    z_h = 84.0  # Vestas V112 Hub height
    u_r = 6.0   # Reference wind speed (m/s) at 10m
    
    # Wind shear log profile formula: u(z) = u_r * ln(z/z_0) / ln(z_r/z_0)
    z = np.linspace(z_0 + 0.1, 120, 500)
    u = u_r * np.log(z / z_0) / np.log(z_r / z_0)
    
    # Compute specific values
    u_hub = u_r * np.log(z_h / z_0) / np.log(z_r / z_0)
    
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(u, z, color='#1f77b4', linewidth=2.5, label='Logarithmic Wind Profile')
    
    # Highlight reference height
    ax.scatter([u_r], [z_r], color='#d35400', s=100, zorder=5, edgecolor='black')
    ax.annotate(f'GFS Reference: 10m\n({u_r:.1f} m/s)', 
                xy=(u_r, z_r), 
                xytext=(u_r + 0.8, z_r - 2),
                arrowprops=dict(arrowstyle="->", color='black'),
                fontweight='bold')
                
    # Highlight hub height
    ax.scatter([u_hub], [z_h], color='#2c3e50', s=100, zorder=5, edgecolor='black')
    ax.annotate(f'Turbine Hub Height: 84m\n({u_hub:.1f} m/s, +{((u_hub-u_r)/u_r)*100:.1f}%)', 
                xy=(u_hub, z_h), 
                xytext=(u_hub - 4.5, z_h + 4),
                arrowprops=dict(arrowstyle="->", color='black'),
                fontweight='bold')
                
    ax.axhline(y=z_h, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=z_r, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Wind Speed (m/s)', fontweight='bold')
    ax.set_ylabel('Height above ground z (m)', fontweight='bold')
    ax.set_title('Logarithmic Wind Profile Correction\n(Deccan Plateau Hill Terrain, Koppal $z_0=0.05$m)', pad=15)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 120)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('log_profile.png', dpi=300)
    plt.close()
    print("Generated log_profile.png successfully!")

if __name__ == "__main__":
    make_penalty_chart()
    make_log_profile()
