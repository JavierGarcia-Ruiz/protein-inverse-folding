# CATH 4.2 Domain Benchmark Dataset

This dataset benchmarks structure-to-sequence inverse folding models on diverse protein domains categorized under the CATH 4.2 Non-redundant (<40% sequence identity, CATH-S40) hierarchy.

## Structural Fold Taxonomy
- **Class 1 (Mainly Alpha)**: Globular bundles of $\alpha$-helices connected by turns and loops (e.g. globin-like, four-helix bundles).
- **Class 2 (Mainly Beta)**: $\beta$-barrels and sandwich architectures (e.g. immunoglobulin-like, jelly rolls).
- **Class 3 (Alpha-Beta)**: Alternating $\beta$-strand and $\alpha$-helix motifs (e.g. TIM barrels, Rossmann folds).
- **Class 4 (Few Secondary Structures)**: Disulfide-rich, irregular loops and metal-binding peptide domains.

## Dataset Partitioning
Evaluated across **4,230 experimental CATH 4.2 crystal domains** under strict `<40%` sequence identity topology splits:
- **Total Domains**: 4,230 non-redundant experimental polypeptide structures (`<40%` sequence identity topology splits)
- **Train Set**: 3,776 domains (89.3%)
- **Validation Set**: 176 domains (4.2%)
- **Test Set**: 278 domains (6.6%)

## Representation Format
- `cath_backbone_coords.npz`: Compressed 3D coordinates for backbone atoms $(N, C_\alpha, C, O)$ in $\text{\AA}$ units.
- `cath_domains_metadata.csv`: Sequence lengths, CATH classifications, Ramachandran dihedrals ($\phi, \psi, \omega$), and secondary structure fractions.

