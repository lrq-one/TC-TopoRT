# Reviewer 2 manuscript edits

The text below is written for direct insertion after the new analyses have been run. Bracketed result fields must be replaced with generated values before submission.

## R2.1 - System dependence and applicability domain

### Add at the end of Section 2.1

Retention time is not an intrinsic molecular property. The SMRT benchmark represents a single reversed-phase LC method using a Zorbax Extend-C18 column and acidic water/acetonitrile mobile phases. TC-TopoRT therefore predicts RT within a defined chromatographic system and does not perform zero-shot conversion between unrelated LC methods. In the external experiments, the SMRT-pretrained representation was adapted using labelled data from each target system. Column chemistry, mobile-phase composition and pH, gradient program, flow rate, and temperature were not explicit model inputs. Target-system validation is consequently required, particularly for HILIC systems or methods with markedly different ionization and retention regimes. The available chromatographic conditions and unreported fields are summarized in Table S34.

### Replace the first paragraph of Section 3.5 with a narrower applicability statement

The candidate-filtering experiments demonstrate analytical utility within the evaluated candidate sets and chromatographic systems. They do not establish a system-independent RT model. TC-TopoRT should be treated as a method-specific predictor or as a pretrained representation that requires labelled adaptation to a new LC system. The presence of two HILIC datasets in the external panel is useful for testing adaptation, but it does not imply direct transfer from the reversed-phase SMRT method without target-system training and validation.

## R2.2 - Tautomer equilibrium and chromatographic resolution

### Add at the end of Section 2.3

The paired branch is a representation-consistency device rather than a physicochemical equilibrium model. Both graphs are assigned the same measured compound-level RT, and the framework does not estimate solution-phase tautomer populations, interconversion kinetics, pH-dependent microstates, or ESI-induced proton rearrangements. If experimentally resolvable tautomers produce distinct chromatographic peaks, they must be represented as separate experimentally labelled analytes. The present one-compound/one-RT protocol cannot assign distinct RT values to such resolved species.

### Add to Section 3.5

The observed dual-view gain should therefore be interpreted as robustness to alternative graph encodings of the same labelled compound, not as evidence that TC-TopoRT reconstructs the dynamic tautomer distribution present during LC or electrospray ionization.

## R2.3 - Isomer discrimination

### Add a new paragraph to Section 3.4 after the structural ablation

To evaluate whether the topology-aware representation helped distinguish closely related structures, we constructed formula-matched pairs from the independent SMRT test set. We evaluated constitutional isomers, stereoisomers, ring-topology challenges, and an operational positional-like subset defined by a shared Murcko scaffold, different constitutional graphs, matched ring signatures, and Morgan similarity of at least 0.45. After requiring an experimental RT gap of at least 10 s, the analysis contained [N_ALL] pairs, including [N_POSITIONAL] positional-like pairs and [N_STEREO] stereoisomer pairs. Full TC-TopoRT achieved a pairwise RT-ordering accuracy of [FULL_ORDERING]% for [CATEGORY], compared with [NO2_ORDERING]% after removing explicit ring 2-cells and [GNN_ORDERING]% for the atom-bond GNN reference. The corresponding RT-difference MAEs were [FULL_DELTA_MAE], [NO2_DELTA_MAE], and [GNN_DELTA_MAE] s. These results provide pair-level evidence on structural discrimination while avoiding the stronger claim that every chromatographic isomer class is resolved.

### Required interpretive sentence

The atom-bond comparison is an architecture-level reference rather than a pure topology-only ablation because it uses a different conventional graph featurization; the no-ring-2-cell model is the direct topology-specific control.

## R2.4 - Exclusion of RT <= 300 s

### Replace the threshold justification in Section 2.1

The `RT > 300 s` threshold was retained to reproduce the established retained-compound SMRT benchmark used by prior studies. It should not be interpreted as evidence that early-eluting compounds are analytically unimportant. Under the SMRT reversed-phase method, this exclusion removes the low-retention region, limits coverage of highly polar metabolites, and prevents the retained benchmark from supporting claims about that region. We therefore treated early-retention performance as a separate full-range sensitivity analysis rather than changing the locked benchmark split.

### Add a new Results paragraph after Section 3.1

In the extended full-range sensitivity analysis, the retained-compound assignments were preserved and the previously excluded `RT <= 300 s` compounds were allocated by a deterministic stratified split. The full-range test set contained [N_FULL_TEST] molecules, including [N_EARLY_TEST] early-retention molecules. TC-TopoRT achieved an MAE of [FULL_MAE] s over the full range, [EARLY_MAE] s for `RT <= 300 s`, and [RETAINED_MAE] s for `RT > 300 s`. The early-region result is reported as a sensitivity analysis because its split is newly constructed rather than part of the official retained benchmark.

## R2.5 - Conditional candidate filtering and denominators

### Add at the beginning of Section 2.7

Candidate filtering was evaluated only for queries whose true structure was present in the initial MS-FINDER list. It is therefore a conditional post-generation evaluation rather than an end-to-end identification rate. Candidate-space reduction was calculated over candidate records, whereas Top-k accuracy, true-candidate retention, and false-negative counts were calculated over queries.

### Add to the Table 2 note

All percentages are conditional on the 45 MetaboBase and 85 RIKEN-PlaSMA evaluable queries. Reduction uses candidate records as the denominator; Top-k and false-negative metrics use queries as the denominator. The analysis does not measure failures in which the true structure was absent from the initial candidate list.

## R2.6 - Concrete cell-complex example

### Replace the abstract-only Figure 2 with a two-part figure

Panel A: the existing generic CWN update.

Panel B: the generated caffeine example showing 14 atom 0-cells, 15 bond 1-cells, and two ring 2-cells.

### Caption addition

(B) Concrete lifting of caffeine into a cell complex. Atoms define 0-cells, covalent bonds define 1-cells, and the two molecular rings define 2-cells attached through bond-ring incidence. This panel illustrates the representation used by TC-TopoRT and is not an independent performance result.

## R2.7 - Chromatographic conditions

### Add Table S34

Use `configs/external_chromatography_metadata.csv` to create Table S34. Include LC mode, column, dimensions, particle size, temperature, flow rate, mobile phases, gradient, source, and an explicit `NR` entry for information not reported. Do not infer missing conditions.

### Add to Section 2.1

Detailed chromatographic conditions for the SMRT source method and all external target systems are provided in Table S34. Missing fields are marked as not reported rather than imputed.
