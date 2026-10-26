# Wafer defect pattern classes
PATTERN_CLASSES = [
    'Center',
    'Donut',
    'Edge-Loc',
    'Edge-Ring',
    'Local',
    'Near-Full',
    'Normal',
    'Random',
    'Scratch'
]

# Root-cause failure mechanisms associated with each class (1-to-1 proxy mapping)
PATTERN_TO_MECHANISM = {
    'Center': 'Center-slow/fast process non-uniformity (e.g. CMP, spin-coat)',
    'Donut': 'Edge-exclusion / implant-angle or deposition non-uniformity',
    'Edge-Loc': 'Localized edge handling damage or edge-bead residue',
    'Edge-Ring': 'Edge-focused process non-uniformity (etch, CMP, thermal gradient)',
    'Local': 'Localized particle contamination or tool defect',
    'Near-Full': 'Gross process failure (equipment fault, wrong recipe)',
    'Normal': 'No defect mechanism',
    'Random': 'Particle contamination, no systematic root cause',
    'Scratch': 'Mechanical handling damage (transport, chuck)'
}

# Unique mechanism classes in standard order
MECHANISM_CLASSES = [PATTERN_TO_MECHANISM[c] for c in PATTERN_CLASSES]

# Map pattern index to mechanism index
PATTERN_IDX_TO_MECHANISM_IDX = {i: i for i in range(len(PATTERN_CLASSES))}
