# Audited speaking notes

## Slide 1

Suggested delivery: 8–10 minutes for slides 1–13. Slides 14–17 are references and backup material. Open with: Our project asks whether spiking state actually helps an audio classifier, and whether that usefulness translates into competitive performance. We separate those questions in a paired audit. This presentation is a research draft, not a peer-reviewed publication. Do not imply institutional endorsement.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 2

About 40 seconds. Explain that a hybrid model can perform well using its ordinary direct features. The audit separately tests state dependence, comparison with dense models, and compiler fidelity. Novelty is the bounded audit and its evidence, not a new LIF neuron or a first residual SNN.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

Audit qualification: earlier SHD/SSC official test outcomes informed architecture exploration. Current confidence intervals summarize training-seed variation on known benchmarks, not an untouched external holdout. Global shuffles mix classes as well as samples. Joint removal of spike traces and final membrane does not isolate spike-specific coding. Numerical values and all registered gates are unchanged. Source: gen5/docs/PROJECT_AUDIT_2026_10_14.md and gen5/docs/EVIDENCE1_HARDENING_2026_10_14.md.

## Slide 3

About 55 seconds. Binary events feed a temporal Conv1D. One path applies ReLU and pooling. The second path uses learned leak and threshold, soft reset, spike traces and the analog final membrane. Both paths enter a shared linear readout. Pyramid levels 1, 2, 4, 8 give 15 windows per trace and 31C total readout features. Symmetric padding and whole-utterance pooling make this an offline hybrid model, not streaming-causal or an all-spike system.
Neftci, Mostafa and Zenke, Surrogate Gradient Learning in Spiking Neural Networks (2019). https://doi.org/10.1109/MSP.2019.2931595

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 4

About 50 seconds. Use all official samples. SHD has a stratified training holdout with split seed 2026, and SSC uses its official validation set. Both use binary occupancy across 64 bins and 700 channels, so duplicate events within the same bin/channel collapse. Train all arms for 15 epochs. Select the earliest checkpoint with maximum validation accuracy, never the final test. Common parameter and optimizer budgets do not make optimization difficulty or FLOPs equal.
Cramer et al., The Heidelberg spiking datasets for systematic evaluation of spiking neural networks. https://arxiv.org/abs/1910.07407 and https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
Loshchilov and Hutter, Decoupled Weight Decay Regularization (ICLR 2019). https://arxiv.org/abs/1711.05101

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 5

About 45 seconds. Explain each intervention on the selected trained residual model. Keep the direct features and readout weights fixed. Zeroing deletes both spike-state trace and final membrane. Shuffling globally pairs every sample with a different sample’s state, across minibatches when needed. Three derangements average within each training seed. These probes measure model dependence, not the capability of a retrained branch-deletion model.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 6

About 55 seconds. Read the paired drops as absolute percentage points, not relative percentages. SHD loses 9.43 points with zero state and 4.04 with exchange. SSC loses 13.64 and 8.87. Every paired 95% interval has a positive lower bound, so the causal gate passes on both datasets. Interventions also disrupt learned feature compatibility, so the finding does not isolate timing-specific coding. Native bars show means. Exact paired confidence intervals appear on backup slide 16.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

Audit qualification: earlier SHD/SSC official test outcomes informed architecture exploration. Current confidence intervals summarize training-seed variation on known benchmarks, not an untouched external holdout. Global shuffles mix classes as well as samples. Joint removal of spike traces and final membrane does not isolate spike-specific coding. Numerical values and all registered gates are unchanged. Source: gen5/docs/PROJECT_AUDIT_2026_10_14.md and gen5/docs/EVIDENCE1_HARDENING_2026_10_14.md.

## Slide 7

About 50 seconds. Native chart shows primary-run means across ten seeds. Residual LIF reaches 84.06% SHD and 56.48% SSC. SSC TCN reaches 58.44% and GRU 57.44%. Do not infer statistical decisions from the bars or overlapping marginal intervals. The next slide uses paired differences. SHD GRU is weak for this retained recipe, with no established explanation or general verdict on GRU performance.
Bai, Kolter and Koltun (2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. https://arxiv.org/abs/1803.01271
Cho et al. (2014), Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. https://aclanthology.org/D14-1179/

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 8

About 55 seconds. The preregistered rule requires the paired interval lower bound to exceed minus one absolute percentage point against every dense control on both datasets. In SSC, TCN is ahead and the entire difference interval falls below minus one. GRU also fails the lower-bound rule. SHD versus Conv1D fails the rule despite a positive mean, which does not prove inferiority. No test-derived change to the margin is allowed.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 9

About 60 seconds. NVIDIA L4, batch 256. The plotted speedups compare compiled and eager inference of the same residual checkpoint after warmup, not residual versus a dense baseline. The benchmark excludes data loading and transfers. Validation covers all samples and tail shapes. Both runs preserve every class prediction but repeat the same two logit-tolerance failures. Thus the systems gate fails. All GRU compilation attempts are unsupported, though training and accuracy evaluation complete. Root cause of the residual fidelity failures remains unestablished.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 10

About 35 seconds. Both bundles contain 80 completed model fits, four arms by two datasets by ten seeds. The second execution uses the same identifiers and split, so do not combine runs into twenty independent seeds. Causal deltas are identical and TCN accuracy differs slightly. Both report causal_audit_only. Human-reviewed lock and certified independent clean-clone repetition remain pending.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 11

About 45 seconds. The full classifier uses sample-specific state. State shuffling can also disrupt branch coadaptation or move the model off its training distribution. The experiment does not isolate spike timing from rate or prove a biological mechanism. Dense baselines share a recipe without independent optimization. Parameter matching does not establish equal work. The system performs offline classification, with no measured energy, hardware-sparse, whole-fly or online-learning result.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

Audit qualification: earlier SHD/SSC official test outcomes informed architecture exploration. Current confidence intervals summarize training-seed variation on known benchmarks, not an untouched external holdout. Global shuffles mix classes as well as samples. Joint removal of spike traces and final membrane does not isolate spike-specific coding. Numerical values and all registered gates are unchanged. Source: gen5/docs/PROJECT_AUDIT_2026_10_14.md and gen5/docs/EVIDENCE1_HARDENING_2026_10_14.md.

## Slide 12

About 40 seconds. First complete the evidence audit and reviewed decision record while keeping Evidence-1 active. Then diagnose compiler discrepancies using frozen checkpoints and validation only, without test-based tuning or a large retraining budget. After the lock, preregister timing-versus-rate controls and matched readout or training comparisons. Context routing and fly-connectome priors are deferred hypotheses, not results in this presentation.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 13

About 30 seconds. Close with: Removing the state or attaching another sample’s state reduced accuracy. That establishes a useful internal state-dependence result. The model does not satisfy every accuracy comparison and compiled logits fail at two checkpoints. Our contribution is a reproducible, bounded audit that keeps these outcomes separate. Invite questions.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

Audit qualification: earlier SHD/SSC official test outcomes informed architecture exploration. Current confidence intervals summarize training-seed variation on known benchmarks, not an untouched external holdout. Global shuffles mix classes as well as samples. Joint removal of spike traces and final membrane does not isolate spike-specific coding. Numerical values and all registered gates are unchanged. Source: gen5/docs/PROJECT_AUDIT_2026_10_14.md and gen5/docs/EVIDENCE1_HARDENING_2026_10_14.md.

## Slide 14

Reference slide. Full primary-source links are in these notes and in the paper. The retained result bundles and preregistration provide the experimental evidence.
Cramer et al., The Heidelberg spiking datasets for systematic evaluation of spiking neural networks. https://arxiv.org/abs/1910.07407 and https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
Neftci, Mostafa and Zenke, Surrogate Gradient Learning in Spiking Neural Networks (2019). https://doi.org/10.1109/MSP.2019.2931595
Bai, Kolter and Koltun (2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. https://arxiv.org/abs/1803.01271
Cho et al. (2014), Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. https://aclanthology.org/D14-1179/
Loshchilov and Hutter, Decoupled Weight Decay Regularization (ICLR 2019). https://arxiv.org/abs/1711.05101

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 15

Use for protocol questions. Every model falls inside 95%–100% of the dense parameter ceiling 133,631. Training is eager and fixed-length, with no scheduler, early stopping or per-arm optimization. Validation determines the selected checkpoint. Each seed pairs the minibatch order, not identical initialization tensors across unlike architectures. Parameter equality alone does not imply equal FLOPs or optimization difficulty.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 16

Paired Student-t 95% intervals across ten training seeds. Each seed first averages its three shuffled-state trials. These are absolute percentage-point differences in final-test accuracy. Positive lower bounds for all four effects satisfy the registered causal rule. The intervals are marginal, not a simultaneous familywise-confidence assertion.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.

## Slide 17

Differences in percentage points with paired two-sided Student-t 95% intervals. A passing lower bound must exceed minus one percentage point. This is a noninferiority condition, not strict superiority or two-sided equivalence. SHD GRU has weak performance under the common recipe, and no established cause. No interval here justifies universal conclusions about spiking or dense architectures.

Local sources: gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md, gen5/ammc_gen5/evidence1_audio.py, output/pdf/evidence1_paper_draft/verified_metrics.json and evidence_audit.json. Retained bundles: research_restart_results.zip and research_restart_replication_01.zip.
