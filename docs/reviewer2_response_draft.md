# Draft response to Reviewer 2

## Comment 1: Chromatographic-system dependence

**Response.** We agree that RT is determined jointly by molecular structure and the chromatographic method. We have narrowed the applicability claim throughout the manuscript. The revised Methods now states that TC-TopoRT is method-specific and that the external experiments use labelled target-system adaptation rather than zero-shot conversion. We also added Table S34, which reports the available column, mobile-phase, gradient, flow-rate, temperature, and LC-mode information for SMRT and the external datasets, with unavailable fields marked as not reported. The Discussion now explicitly distinguishes reversed-phase and HILIC deployment and requires target-system validation.

## Comment 2: Dynamic tautomer equilibria

**Response.** We agree that the dual-view construction should not be interpreted as a model of solution-phase tautomer populations. We have added an explicit clarification that the paired views are a representation-consistency device. They share one compound-level RT label and do not model tautomer populations, exchange kinetics, pH-dependent microstates, or ESI-induced proton rearrangements. We further state that chromatographically resolved tautomers would require separate experimentally labelled records and cannot be assigned distinct RT values by the present one-compound/one-RT protocol.

## Comment 3: Positional and stereoisomer discrimination

**Response.** We added a controlled, formula-matched pair analysis on the independent SMRT test set. The analysis reports pairwise RT-ordering accuracy and RT-difference error for constitutional isomers, stereoisomers, ring-topology challenges, and an explicitly defined positional-like subset. It compares full TC-TopoRT with the direct no-ring-2-cell control and an atom-bond GNN reference. The generated analysis contained [N_ALL] eligible pairs and the full model achieved [INSERT GENERATED RESULTS]. We have avoided claiming universal chromatographic resolution and now describe the atom-bond GNN as an architecture-level reference because its conventional feature dimensions differ from those of TC-TopoRT.

## Comment 4: Exclusion of RT <= 300 s

**Response.** We agree that the retained-compound benchmark does not characterize highly polar, early-eluting molecules. We retained the threshold only to preserve comparability with the established SMRT benchmark and added this limitation explicitly. We also prepared a separate full-range sensitivity experiment that preserves all retained-compound assignments and adds a deterministic stratified split for the previously excluded molecules. The revised manuscript will report the full-range, early-region, and retained-region errors as [INSERT GENERATED RESULTS], while clearly labelling the extended split as a sensitivity analysis rather than the official benchmark.

## Comment 5: Candidate filtering and parameter selection

**Response.** We clarified that the candidate-filtering evaluation is conditional on the true structure being present in the initial MS-FINDER list and is not an end-to-end identification rate. Candidate-space reduction is calculated over candidate records, whereas Top-k accuracy, true-candidate retention, and false-negative counts are calculated over queries. The revised text also retains the complete four-parameter sensitivity audit over T, g, tau, and alpha. The main operating points remain on the five-metric non-dominated front and were applied unchanged to every RT-based method.

## Comment 6: Concrete 0/1/2-cell example

**Response.** We added a concrete molecular lifting example using caffeine. The new panel identifies the 14 atom 0-cells, 15 bond 1-cells, and two ring 2-cells and shows how ring cells attach through bond-ring incidence. The caption states that this is an explanatory representation figure rather than an additional performance experiment.

## Comment 7: Chromatographic conditions

**Response.** We added a dedicated supplementary table containing the available chromatographic conditions for the SMRT source method and all ten external target systems. The table includes LC mode, column chemistry and dimensions, particle size, mobile phases, gradient program, flow rate, temperature, and source. Fields unavailable in the primary report or RepoRT are marked `NR` and are not imputed.

## Changes that must be completed before this response is submitted

1. Run the isomer-pair analysis and replace every `[INSERT GENERATED RESULTS]` field.
2. Run the five-seed full-range sensitivity experiment and replace the early-retention placeholders.
3. Insert Table S34 and the concrete cell-complex panel into the manuscript source.
4. Recompile the main manuscript and SI, then verify all table, figure, and section references.
