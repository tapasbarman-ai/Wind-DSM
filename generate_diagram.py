import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    # Color palette matching LaTeX
    physblue = '#1f77b4'
    mlgreen = '#2ca02c'
    gray = '#7f7f7f'
    light_blue = '#e3f2fd'
    light_green = '#e8f5e9'
    light_gray = '#f5f5f5'
    
    # Draw boxes
    # 1. Weather Inputs
    ax.add_patch(patches.FancyBboxPatch((0.5, 2.0), 1.8, 1.6, boxstyle="round,pad=0.1", 
                                        facecolor=light_gray, edgecolor=gray, linewidth=2))
    ax.text(1.4, 2.8, "NOAA GFS\nWeather Forecast\n(Wind, Temp, Pres)", 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 2. Physics Engine (Top path)
    ax.add_patch(patches.FancyBboxPatch((3.5, 3.6), 2.2, 1.2, boxstyle="round,pad=0.1", 
                                        facecolor=light_blue, edgecolor=physblue, linewidth=2))
    ax.text(4.6, 4.2, "Physics Engine\n(Vestas V112 Model)\n$P_{physics}(t)$", 
            ha='center', va='center', fontsize=10, fontweight='bold', color=physblue)
    
    # 3. XGBoost Residual Learner (Bottom path)
    ax.add_patch(patches.FancyBboxPatch((3.5, 0.8), 2.2, 1.2, boxstyle="round,pad=0.1", 
                                        facecolor=light_green, edgecolor=mlgreen, linewidth=2))
    ax.text(4.6, 1.4, "XGBoost Regressor\n(Residual Learner)\n$\\hat{\\epsilon}_{XGB}(t)$", 
            ha='center', va='center', fontsize=10, fontweight='bold', color=mlgreen)
    
    # 4. Summing Junction
    sum_circle = plt.Circle((7.0, 2.8), 0.3, facecolor='#fff', edgecolor='#2c3e50', linewidth=2, zorder=5)
    ax.add_patch(sum_circle)
    ax.text(7.0, 2.8, "+", ha='center', va='center', fontsize=18, fontweight='bold')
    
    # 5. Output schedule
    ax.add_patch(patches.FancyBboxPatch((8.0, 2.2), 1.6, 1.2, boxstyle="round,pad=0.1", 
                                        facecolor='#fffde7', edgecolor='#f57f17', linewidth=2))
    ax.text(8.8, 2.8, "Final CERC\nSchedule\n$\\hat{P}_{final}(t)$", 
            ha='center', va='center', fontsize=10, fontweight='bold', color='#f57f17')
    
    # Draw arrows
    # GFS to Physics Engine
    ax.annotate("", xy=(3.4, 4.2), xytext=(2.4, 3.2), 
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, connectionstyle="angle,angleA=0,angleB=90,rad=5"))
    # GFS to ML Engine
    ax.annotate("", xy=(3.4, 1.4), xytext=(2.4, 2.4), 
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, connectionstyle="angle,angleA=0,angleB=90,rad=5"))
    
    # Physics to Sum
    ax.annotate("", xy=(6.65, 2.8), xytext=(5.8, 4.2),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, connectionstyle="angle,angleA=0,angleB=-90,rad=5"))
    ax.text(6.4, 3.8, "$P_{physics}$", ha='right', va='bottom', fontsize=9)
    
    # ML to Sum
    ax.annotate("", xy=(6.65, 2.8), xytext=(5.8, 1.4),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, connectionstyle="angle,angleA=0,angleB=90,rad=5"))
    ax.text(6.4, 1.8, "$\\hat{\\epsilon}_{XGB}$", ha='right', va='top', fontsize=9)
    
    # Sum to Output
    ax.annotate("", xy=(7.9, 2.8), xytext=(7.3, 2.8),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=2))
    
    # Physics constraints callout (Clip to [0, Capacity])
    ax.text(8.8, 1.8, "Clipped to $[0, P_{max}]$", ha='center', va='top', fontsize=8, style='italic', color='#7f8c8d')
    
    # Add title and captions
    plt.title("Double-Engine PI-PML Pipeline Architecture", fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig('double_engine_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated double_engine_diagram.png successfully!")

if __name__ == "__main__":
    generate_diagram()
