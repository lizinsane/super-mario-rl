"""
Skript zur Visualisierung der Training-Logs aus training_log.csv

Verwendung:
    python plot_training.py

Erstellt Plots für:
- Average Return über Zeit
- Max Stage Fortschritt
- Evaluations-Metriken (x_pos, score, coins, etc.)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_progress(csv_file="training_log.csv"):
    """
    Liest CSV-Datei und erstellt Trainings-Visualisierungen
    """
    
    if not os.path.exists(csv_file):
        print(f"❌ Datei nicht gefunden: {csv_file}")
        print(f"ℹ️  Starte zuerst das Training, um Daten zu sammeln!")
        print(f"📁 Aktuelles Verzeichnis: {os.getcwd()}")
        return
    
    # Lade Daten
    df = pd.read_csv(csv_file)
    print(f"📊 Geladene Daten: {len(df)} Einträge")
    print(f"📅 Von {df['timestamp'].iloc[0]} bis {df['timestamp'].iloc[-1]}")
    
    # Konvertiere timestamp zu datetime für bessere Plots
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Erstelle Figure mit Subplots - 3 Zeilen x 3 Spalten
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('PPO Training Progress - Super Mario Bros', fontsize=16, fontweight='bold')
    
    # Plot 1: Average Return über Updates
    ax1 = axes[0, 0]
    ax1.plot(df['update'], df['avg_return'], 'b-', alpha=0.6, linewidth=1)
    ax1.set_xlabel('Update')
    ax1.set_ylabel('Average Return')
    ax1.set_title('Training Return (alle Updates)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Erreichte Stages (NEUER HAUPTPLOT für Stages)
    ax2 = axes[0, 1]
    # Zeige max_stage als Punkte (nicht Linien)
    ax2.scatter(df['update'], df['max_stage'], c='green', alpha=0.7, s=30, edgecolors='darkgreen')
    # Füge horizontale Linien für alle 4 Stages hinzu
    ax2.axhline(y=1, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Stage 1')
    ax2.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Stage 2')
    ax2.axhline(y=3, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Stage 3')
    ax2.axhline(y=4, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label='Stage 4')
    ax2.set_xlabel('Update')
    ax2.set_ylabel('Stage / Level')
    ax2.set_title('Erreichte Stages')
    ax2.set_ylim(0.5, 4.5)
    ax2.set_yticks([1, 2, 3, 4])
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Evaluation Return
    df_eval = df[df['eval_avg_return'].notna()]
    ax3 = axes[0, 2]
    if len(df_eval) > 0:
        ax3.plot(df_eval['update'], df_eval['eval_avg_return'].astype(float), 
                'r-', marker='o', linewidth=2, markersize=4)
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Evaluation Avg Return')
        ax3.set_title('Evaluation Performance (alle 10 Updates)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Noch keine\nEvaluations-Daten', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: X Position (Fortschritt im Level)
    ax4 = axes[1, 0]
    if len(df_eval) > 0 and df_eval['eval_x_pos'].notna().any():
        ax4.plot(df_eval['update'], df_eval['eval_x_pos'].astype(float), 
                'purple', marker='o', linewidth=2, markersize=4)
        ax4.axhline(y=3161, color='g', linestyle='--', label='Level-Ende (~3161)')
        ax4.set_xlabel('Update')
        ax4.set_ylabel('X Position (Pixel)')
        ax4.set_title('Fortschritt im Level (x_pos)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Noch keine\nx_pos Daten', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # Plot 5: Score & Coins
    ax5 = axes[1, 1]
    if len(df_eval) > 0:
        ax5_twin = ax5.twinx()
        if df_eval['eval_score'].notna().any():
            ax5.plot(df_eval['update'], df_eval['eval_score'].astype(float), 
                    'orange', marker='s', linewidth=2, markersize=4, label='Score')
            ax5.set_ylabel('Score', color='orange')
            ax5.tick_params(axis='y', labelcolor='orange')
        
        if df_eval['eval_coins'].notna().any():
            ax5_twin.plot(df_eval['update'], df_eval['eval_coins'].astype(float), 
                         'gold', marker='^', linewidth=2, markersize=4, label='Coins')
            ax5_twin.set_ylabel('Coins', color='gold')
            ax5_twin.tick_params(axis='y', labelcolor='gold')
        
        ax5.set_xlabel('Update')
        ax5.set_title('Score & Münzen')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Noch keine\nScore/Coins Daten', 
                ha='center', va='center', transform=ax5.transAxes)
    
    # Plot 6: Evaluation Score (NEUER PLOT statt Leben)
    ax6 = axes[1, 2]
    if len(df_eval) > 0 and df_eval['eval_score'].notna().any():
        ax6.scatter(df_eval['update'], df_eval['eval_score'].astype(float), 
                   c='purple', alpha=0.7, s=50, edgecolors='darkviolet')
        ax6.set_xlabel('Update')
        ax6.set_ylabel('Evaluation Score')
        ax6.set_title('Score bei Evaluation')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Noch keine\nScore Daten', 
                ha='center', va='center', transform=ax6.transAxes)
    
    # Plot 7: X Position Übersicht (ALLE Evaluationen)
    ax7 = axes[2, 0]
    if len(df_eval) > 0 and df_eval['eval_x_pos'].notna().any():
        x_pos_data = df_eval['eval_x_pos'].astype(float)
        ax7.plot(df_eval['update'], x_pos_data, 
                'teal', marker='D', linewidth=2, markersize=5, alpha=0.7)
        ax7.axhline(y=3161, color='lime', linestyle='--', linewidth=2, label='Ziel: 3161 (Level-Ende)')
        ax7.fill_between(df_eval['update'], 0, x_pos_data, alpha=0.2, color='teal')
        ax7.set_xlabel('Update')
        ax7.set_ylabel('X Position (Pixel)')
        ax7.set_title('Level-Fortschritt: X Position bei Evaluation')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        # Füge Annotations für maximale x_pos hinzu
        max_x = x_pos_data.max()
        max_update = df_eval.loc[x_pos_data.idxmax(), 'update']
        ax7.annotate(f'Max: {max_x:.0f}', 
                    xy=(max_update, max_x), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    else:
        ax7.text(0.5, 0.5, 'Noch keine\nx_pos Daten', 
                ha='center', va='center', transform=ax7.transAxes)
    
    # Plot 8: X Position Fortschritt (Balkendiagramm)
    ax8 = axes[2, 1]
    if len(df_eval) > 0 and df_eval['eval_x_pos'].notna().any():
        x_pos_data = df_eval['eval_x_pos'].astype(float)
        # Gruppiere in Bereiche
        recent_evals = df_eval.tail(20)  # Letzte 20 Evaluationen
        if len(recent_evals) > 0:
            x_pos_recent = recent_evals['eval_x_pos'].astype(float)
            colors = ['red' if x < 1000 else 'orange' if x < 2000 else 'green' for x in x_pos_recent]
            ax8.bar(range(len(x_pos_recent)), x_pos_recent, color=colors, alpha=0.7)
            ax8.axhline(y=3161, color='lime', linestyle='--', linewidth=2, label='Level-Ende')
            ax8.set_xlabel('Letzte 20 Evaluationen')
            ax8.set_ylabel('X Position (Pixel)')
            ax8.set_title('Fortschritt: Letzte 20 Evaluationen')
            ax8.legend()
            ax8.grid(True, alpha=0.3, axis='y')
    else:
        ax8.text(0.5, 0.5, 'Noch keine\nx_pos Daten', 
                ha='center', va='center', transform=ax8.transAxes)
    
    # Plot 9: Statistik-Übersicht
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    # Erstelle Text-Statistiken
    stats_text = "📊 TRAINING STATISTIK\n\n"
    stats_text += f"Updates: {df['update'].max()}\n"
    stats_text += f"Aktueller Return: {df['avg_return'].iloc[-1]:.2f}\n"
    stats_text += f"Bester Return: {df['avg_return'].max():.2f}\n\n"
    
    if len(df_eval) > 0:
        stats_text += "🎯 EVALUATION\n\n"
        if df_eval['eval_x_pos'].notna().any():
            max_x = df_eval['eval_x_pos'].astype(float).max()
            stats_text += f"Max x_pos: {max_x:.0f}\n"
            progress = (max_x / 3161) * 100
            stats_text += f"Progress: {progress:.1f}%\n\n"
        if df_eval['eval_score'].notna().any():
            stats_text += f"Max Score: {df_eval['eval_score'].astype(float).max():.0f}\n"
        if df_eval['eval_coins'].notna().any():
            stats_text += f"Max Coins: {df_eval['eval_coins'].astype(float).max():.0f}\n"
        if df_eval['eval_flag_get'].notna().any():
            flags = df_eval['eval_flag_get'].sum()
            if flags > 0:
                stats_text += f"\n🎉 Flagge: {flags}x erreicht!"
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')

    
    plt.tight_layout()
    
    # Speichere Plot
    output_file = "training_progress.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot gespeichert: {output_file}")
    
    # Zeige Plot
    plt.show()
    
    # Statistiken ausgeben
    print("\n📈 TRAININGS-STATISTIKEN:")
    print(f"   Gesamt Updates: {df['update'].max()}")
    print(f"   Aktueller avg_return: {df['avg_return'].iloc[-1]:.2f}")
    print(f"   Bester avg_return: {df['avg_return'].max():.2f}")
    print(f"   Max erreichte Stage: {df['max_stage'].max()}")
    
    if len(df_eval) > 0:
        print(f"\n🎯 EVALUATIONS-STATISTIKEN:")
        print(f"   Anzahl Evaluationen: {len(df_eval)}")
        if df_eval['eval_x_pos'].notna().any():
            print(f"   Weiteste x_pos: {df_eval['eval_x_pos'].astype(float).max():.0f} Pixel")
        if df_eval['eval_score'].notna().any():
            print(f"   Höchster Score: {df_eval['eval_score'].astype(float).max():.0f}")
        if df_eval['eval_coins'].notna().any():
            print(f"   Meiste Münzen: {df_eval['eval_coins'].astype(float).max():.0f}")
        if df_eval['eval_flag_get'].notna().any():
            flag_success = df_eval['eval_flag_get'].sum()
            if flag_success > 0:
                print(f"   🎉 Flagge erreicht: {flag_success} mal!")


if __name__ == "__main__":
    print("🎨 Erstelle Training-Visualisierungen...\n")
    plot_training_progress()
