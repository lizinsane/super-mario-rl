"""
Visualisierung der Episode-Trainingsdaten für Super Mario Bros PPO
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# PARAMETER
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
EPISODE_LOG = SCRIPT_DIR / "episode_log.csv"
OUTPUT_FILE = SCRIPT_DIR / "training_plots.png"

# Plot-Einstellungen
WINDOW_SIZE = 100           # Glättungsfenster für Moving Average
ALPHA_RAW = 0.15            # Transparenz für rohe Daten
ALPHA_SMOOTH = 0.8          # Transparenz für geglättete Linie
LINEWIDTH_RAW = 0.5         # Linienbreite für rohe Daten
LINEWIDTH_SMOOTH = 1.5      # Linienbreite für geglättete Linie
DPI = 150                   # Auflösung des Bildes
FIGSIZE = (16, 10)          # Größe der gesamten Figure

# Maximale X-Position für jedes Level (aus ppo2.py)
LEVEL_MAX_X = {
    (1, 1): 3266,
    (1, 2): 3266,
    (1, 3): 2514,
    (1, 4): 2430,
    (2, 1): 3266,
    (2, 2): 3266,
    (2, 3): 2514,
    (2, 4): 2430,
}


# ============================================================================
# DATEN LADEN UND VERARBEITEN
# ============================================================================

def load_data():
    """Lädt und verarbeitet die Episode-Daten"""
    print(f"[INFO] Lade Daten aus {EPISODE_LOG}...")
    
    try:
        df = pd.read_csv(EPISODE_LOG)
        print(f"[SUCCESS] {len(df)} Episoden geladen")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Datei nicht gefunden: {EPISODE_LOG}")
        return None
    except Exception as e:
        print(f"[ERROR] Fehler beim Laden: {e}")
        return None


def compute_moving_average(data, window):
    """Berechnet gleitenden Durchschnitt"""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean()


# ============================================================================
# PLOTTING FUNKTIONEN
# ============================================================================

def plot_metric(ax, x, y, title, ylabel, color, window_size=WINDOW_SIZE, add_level_lines=False):
    """
    Plottet eine Metrik mit rohen Daten und gleitendem Durchschnitt.
    
    Args:
        ax: Matplotlib Axes Objekt
        x: X-Achsen Daten (Episode-Nummern)
        y: Y-Achsen Daten (Metrik-Werte)
        title: Titel des Subplots
        ylabel: Label für Y-Achse
        color: Farbe für den Plot
        window_size: Fenstergröße für Moving Average
        add_level_lines: Ob horizontale Linien für Level-Enden hinzugefügt werden sollen
    """
    # Rohe Daten (sehr transparent, dünn)
    ax.plot(x, y, color=color, alpha=ALPHA_RAW, linewidth=LINEWIDTH_RAW, 
            label='Rohe Daten')
    
    # Gleitender Durchschnitt (sichtbar, dicker)
    if len(y) >= window_size:
        y_smooth = compute_moving_average(y, window_size)
        ax.plot(x, y_smooth, color=color, alpha=ALPHA_SMOOTH, 
                linewidth=LINEWIDTH_SMOOTH, label=f'MA {window_size}')
    
    # Horizontale Linien für Level-Maxima
    if add_level_lines:
        # Berechne kumulative X-Positionen
        cumulative_positions = []
        cumulative = 0
        for world in range(1, 3):  # World 1 und 2
            for stage in range(1, 5):  # Stage 1-4
                if (world, stage) in LEVEL_MAX_X:
                    cumulative += LEVEL_MAX_X[(world, stage)]
                    cumulative_positions.append((world, stage, cumulative))
        
        # Zeichne gestrichelte Linien
        for world, stage, cum_x in cumulative_positions:
            ax.axhline(y=cum_x, color='gray', linestyle='--', linewidth=0.8, 
                      alpha=0.5, zorder=1)
            # Label am rechten Rand
            ax.text(ax.get_xlim()[1] * 0.98, cum_x, f'{world}-{stage}', 
                   fontsize=7, va='bottom', ha='right', color='gray', alpha=0.7)
    
    # Styling
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    # Formatierung
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)


def create_plots(df):
    """Erstellt alle Plots"""
    print("\nErstelle Plots...")
    
    # Figure Setup
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, dpi=DPI)
    fig.suptitle('Super Mario Bros PPO - Training Progress', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Episode-Nummern für X-Achse
    episodes = np.arange(1, len(df) + 1)
    
    # Plot 1: Score
    plot_metric(
        axes[0, 0], 
        episodes, 
        df['score'], 
        'Score pro Episode',
        'Score',
        color='#2E86AB'  # Blau
    )
    
    # Plot 2: X-Position
    plot_metric(
        axes[0, 1], 
        episodes, 
        df['x_pos'], 
        'Erreichte X-Position',
        'X-Position (Pixel)',
        color='#A23B72',  # Pink
        add_level_lines=True  # Level-Linien hinzufügen
    )
    
    # Plot 3: Max Stage
    plot_metric(
        axes[1, 0], 
        episodes, 
        df['max_stage'], 
        'Maximales Stage erreicht',
        'Stage (1-32)',
        color='#F18F01'  # Orange
    )
    
    # Plot 4: Coins
    plot_metric(
        axes[1, 1], 
        episodes, 
        df['coins'], 
        'Gesammelte Münzen',
        'Anzahl Münzen',
        color='#C73E1D'  # Rot
    )
    
    # Layout optimieren
    plt.tight_layout()
    
    return fig


# ============================================================================
# STATISTIKEN
# ============================================================================

def print_statistics(df):
    """Gibt Trainingsstatistiken aus"""
    print("\n" + "=" * 60)
    print("TRAININGS-STATISTIKEN")
    print("=" * 60)
    
    total_episodes = len(df)
    
    # Score
    print("\nScore:")
    print(f"   Durchschnitt: {df['score'].mean():.1f}")
    print(f"   Maximum: {df['score'].max():.0f}")
    print(f"   Median: {df['score'].median():.1f}")
    
    # X-Position
    print("\nX-Position:")
    print(f"   Durchschnitt: {df['x_pos'].mean():.1f} Pixel")
    print(f"   Maximum: {df['x_pos'].max():.0f} Pixel")
    print(f"   Median: {df['x_pos'].median():.1f} Pixel")
    
    # Max Stage
    print("\nMax Stage:")
    print(f"   Durchschnitt: {df['max_stage'].mean():.2f}")
    print(f"   Maximum: {df['max_stage'].max():.0f}")
    print(f"   Häufigste Stage: {df['max_stage'].mode()[0] if not df['max_stage'].mode().empty else 'N/A'}")
    
    # Coins
    print("\nMünzen:")
    print(f"   Durchschnitt: {df['coins'].mean():.2f}")
    print(f"   Maximum: {df['coins'].max():.0f}")
    print(f"   Total gesammelt: {df['coins'].sum():.0f}")
    
    # Allgemein
    print("\nAllgemein:")
    print(f"   Gesamt Episoden: {total_episodes}")
    
    # Letzte 100 Episoden vs. Erste 100
    if total_episodes >= 200:
        first_100_score = df['score'].iloc[:100].mean()
        last_100_score = df['score'].iloc[-100:].mean()
        improvement = ((last_100_score - first_100_score) / first_100_score * 100) if first_100_score > 0 else 0
        
        print(f"\nFortschritt (Erste 100 vs. Letzte 100):")
        print(f"   Score: {first_100_score:.1f} → {last_100_score:.1f} ({improvement:+.1f}%)")
        
        first_100_xpos = df['x_pos'].iloc[:100].mean()
        last_100_xpos = df['x_pos'].iloc[-100:].mean()
        xpos_improvement = ((last_100_xpos - first_100_xpos) / first_100_xpos * 100) if first_100_xpos > 0 else 0
        print(f"   X-Pos: {first_100_xpos:.0f} → {last_100_xpos:.0f} ({xpos_improvement:+.1f}%)")
    
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Hauptfunktion"""
    print("=" * 60)
    print("Super Mario Bros PPO - Episode Visualisierung")
    print("=" * 60)
    
    # Daten laden
    df = load_data()
    if df is None:
        return
    
    # Statistiken ausgeben
    print_statistics(df)
    
    # Plots erstellen
    fig = create_plots(df)
    
    # Speichern
    print(f"\nSpeichere Plots nach {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Plot gespeichert!")
    
    # Anzeigen
    print("\nZeige Plots...")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Fertig!")
    print("=" * 60)


if __name__ == "__main__":
    main()
