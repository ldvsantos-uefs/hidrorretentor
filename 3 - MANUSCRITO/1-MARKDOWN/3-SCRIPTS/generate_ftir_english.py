"""
Script to generate comparative FTIR figure between Typha domingensis and Syagrus coronata
Based on real data extracted from articles - English Version
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style settings
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Poppins", "Arial", "DejaVu Sans"],
        "axes.linewidth": 1.2,
    }
)

def load_taboa_data(csv_path):
    """Loads Taboa FTIR data"""
    data = pd.read_csv(
        csv_path,
        sep=';',
        skiprows=1,  # Skip header row
        decimal=','
    )
    # Rename columns
    data.columns = ['wavenumber', 'transmittance']
    # Remove NaN and invalid rows
    data = data.dropna()
    # Ensure they are numbers
    data['wavenumber'] = pd.to_numeric(data['wavenumber'], errors='coerce')
    data['transmittance'] = pd.to_numeric(data['transmittance'], errors='coerce')
    data = data.dropna()
    return data

def load_palm_data(csv_path):
    """Loads Palm/Ouricuri FTIR data"""
    data = pd.read_csv(csv_path)
    # Rename columns if necessary
    if 'Wavenumber' in data.columns:
        data.rename(columns={'Wavenumber': 'wavenumber', 'Transmittance': 'transmittance'}, inplace=True)
    return data

def create_comparative_figure(taboa_df, palm_df, output_path):
    """Creates comparative figure with two overlapping spectra"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot Taboa spectrum
    ax.plot(taboa_df['wavenumber'], taboa_df['transmittance'], 
            color='#1B998B', linewidth=1.5, label='Typha domingensis', alpha=0.85)
    
    # Plot Ouricuri spectrum (with vertical offset for better visualization)
    offset = -15  # Vertical offset
    ax.plot(palm_df['wavenumber'], palm_df['transmittance'] + offset, 
            color='#C44536', linewidth=1.5, label='Syagrus coronata', alpha=0.85)
    
    # Main annotations - Typha (upper part)
    typha_annotations = [
        (3300, None, '3300\nO-H'),
        (2920, None, '2920\nC-H'),
        (1735, None, '1735\nC=O'),
        (1600, None, '1600\nAromatic'),
        (1035, None, '1035\nC-O'),
    ]
    
    # Main annotations - Ouricuri (lower part)
    ouricuri_annotations = [
        (3340, None, '3340\nO-H'),
        (2852, None, '2852\nC-H'),
        (1732, None, '1732\nC=O↓'),
        (1590, None, '1590\nSyringyl'),
        (1234, None, '1234\nLignin'),
        (896, None, '896\nβ-glyc'),
    ]
    
    # Add annotations for Typha
    for wave, _, label in typha_annotations:
        # Find approximate transmittance
        idx = (taboa_df['wavenumber'] - wave).abs().idxmin()
        trans = taboa_df.loc[idx, 'transmittance']
        ax.annotate(label, 
                   xy=(wave, trans),
                   xytext=(wave, trans + 8),
                   fontsize=8,
                   ha='center',
                   color='#1B998B',
                   weight='bold',
                   arrowprops=dict(arrowstyle='->', lw=0.8, color='#1B998B'))
    
    # Add annotations for Ouricuri
    for wave, _, label in ouricuri_annotations:
        # Find approximate transmittance in palm data
        idx = (palm_df['wavenumber'] - wave).abs().idxmin()
        trans = palm_df.loc[idx, 'transmittance'] + offset
        ax.annotate(label, 
                   xy=(wave, trans),
                   xytext=(wave, trans - 8),
                   fontsize=8,
                   ha='center',
                   color='#C44536',
                   weight='bold',
                   arrowprops=dict(arrowstyle='->', lw=0.8, color='#C44536'))

    # Settings
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Transmittance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparative FTIR: Typha domingensis vs Syagrus coronata', fontsize=14, fontweight='bold', pad=15)
    
    ax.invert_xaxis()
    ax.set_ylim(20, 110)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved at: {output_path}")
    
    return fig

def main():
    """Main function"""
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / '5-DADOS'
    output_dir = base_dir / '3-IMAGENS'
    
    taboa_file = data_dir / 'FTIR_Taboa.csv'
    palm_file = data_dir / 'FTIR_Ouricuri_Palm.csv'
    output_file = output_dir / 'figura_ftir_comparativa_en.png'
    
    # Check files
    if not taboa_file.exists():
        print(f"ERROR: {taboa_file} not found!")
        return
    if not palm_file.exists():
        print(f"ERROR: {palm_file} not found!")
        return
    
    # Load data
    print("Loading FTIR data...")
    taboa_data = load_taboa_data(taboa_file)
    palm_data = load_palm_data(palm_file)
    
    print(f"✓ Typha: {len(taboa_data)} spectral points")
    print(f"✓ Ouricuri: {len(palm_data)} spectral points")
    
    # Create figures
    print("\nGenerating comparative figures...")
    create_comparative_figure(taboa_data, palm_data, output_file)
    
    print("\n" + "="*60)
    print("✓ Processing completed!")
    print("✓ Figures ready for manuscript")
    print("="*60)

if __name__ == "__main__":
    main()
