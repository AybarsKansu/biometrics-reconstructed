# Biometrics reconstruction plan

This document defines the implementation scope of the new project. `reconstruct.md` is the Turkish equivalent of this plan. Existing code may serve as a reference; do not carry over conflicting decisions from the old architecture, README, or roadmap. The current task only writes the plan; do not rebuild the application yet.

## 1. Objective and experiment definition

Measure whether an attacker can find an accepted vector in an embedding-based biometric system using only its permitted feedback. The attacker submits embeddings directly, not photographs. Model training, fine-tuning, and image reconstruction are out of scope.

Record two separate outcomes:

- Acceptance: The submitted vector is accepted by the 1:N verifier.
- Identity recovery: The same first accepted vector also sufficiently resembles held-out image embeddings of the identity matched at that acceptance. This does not mean generating a face image.

An initial-stage attack run stops at its first acceptance. Evaluate recovery once, only for that first accepted vector. Do not continue optimization after acceptance. Without acceptance, recovery is unsuccessful; retain runs that exhaust their budgets.

## 2. Architecture and file simplicity

- Use understandable Python scripts that run directly. Do not build an HTTP API, server, microservices, or a general-purpose framework.
- Use one configuration, one data index, and one embedding store.
- Small shared functions are allowed. Helper files may prevent duplication; avoid unnecessary classes and layers.
- The data index contains image ID, identity ID, relative path, validity, partition, and role. Associate embedding extraction failures with this index.
- Explicitly map embedding rows to image IDs. Select G/P/H and enrollment/held-out views through that mapping.
- Do not produce partition-specific embedding copies, a second enrollment database, or separate attacker_pool/image_roles/split_report artifacts. Select views from the index.
- Choose CSV, JSON, or NPZ according to the actual need; do not save the same information repeatedly in different formats.
- Experiment records and final tables/plots derived from them serve distinct purposes and may both be retained.

## 3. Layer 1 — Configuration and environment

- Declare Python dependencies in one environment/dependency file.
- Keep paths, seeds, model settings, calibration settings, attack parameters, repeat count, and query budget in one configuration.
- Use paired seeds and starting candidates across comparisons. Population/simplex methods use the pool ordering associated with the same seed.
- Save the configuration once per experiment batch. Use direct scripts and config.json without command-line options or file fingerprints.

## 4. Layer 2 — Data preparation

- Use LFW and read an already extracted image directory. Do not develop extensive archive management as part of the application.
- Check image readability, identity information, and actual counts.
- Create seeded, identity-disjoint partitions:

| Partition | Identities | Purpose |
|---|---:|---|
| G | 1,000 | Enrolled targets |
| P | 2,000 | Attacker initialization and zero-effort |
| H | 2,500 | Operating and recovery threshold calibration |
| Remainder | Nominally 249 | Unused |

- A separate development partition and hyperparameter search are outside the initial scope.
- Each G identity needs at least two usable images: one enrollment and the remaining images held out.
- Explicitly flag images without embeddings. If a G identity loses usable enrollment or held-out data, it may be swapped deterministically with a P identity having at least two usable images, without changing H. Record swaps and final counts.
- The attacker may use P; do not give it G enrollment or held-out data. H is for calibration.

## 5. Layer 3 — Model and embeddings

- Model: The ArcFace-based InsightFace `buffalo_l` recognizer. Do not switch to iResNet100 or perform model comparisons.
- Use compatible detection/alignment and preprocessing; make the per-image face selection policy explicit and fixed.
- Store 512-dimensional float32 embeddings and L2-normalize them.
- Check NaN, infinity, zero norms, and image-to-row alignment.
- Reuse embeddings while the model and preprocessing remain unchanged.

One small metadata record is sufficient:

- Model name and recognition checkpoint filename.
- InsightFace version and face selection/preprocessing settings.
- Dataset source and the LFW edition/alignment variant used.
- Split seed, partition counts, and number of failed embedding extractions.

Do not build a per-image hash inventory or audit overlap between model training identities and LFW. Report unaudited overlap as a limitation. Do not redistribute raw face images or model weights; briefly record provenance and use terms.

## 6. Layer 4 — Verifier and calibration

- The primary system is 1:N: one enrollment vector per G identity, 1,000 vectors total.
- Compute exact cosine scores with NumPy matrix multiplication. No FAISS or separate vector database.
- The score is the maximum across the gallery. Accept when `score >= threshold`.
- Keep the top-1 identity internal for evaluation and, when needed, defense counters; never return it to the attacker.
- Implement 1:1 comparisons for recovery evaluation. A separate 1:1 attack experiment matrix is outside the initial scope.
- Draw 200 seeded surrogate galleries from H. Each gallery has 1,000 identities with one enrollment each; images of other H identities are impostor probes.
- Derive system-level FMR 1% and 0.1% thresholds from these 1:N top-1 scores. Record target and achieved empirical FMR.
- Derive the recovery threshold at pairwise FMR 0.01% from different-identity 1:1 scores within H. Process large comparisons in blocks if needed.
- Do not compute confidence intervals or bootstrap estimates. Describe uncertainty from rare error rates and correlated scores in the report.
- Separately check the model using the standard LFW 6,000-pair / 10-fold protocol. Never present published model accuracy as a local measurement.

Feedback:

- Float: cosine score.
- 8-bit and 4-bit: separate equal-width bins below and above the threshold. Set `lo` to the 0.1st percentile of H top-1 impostor scores. Half the codes represent rejection and half acceptance, preserving decisions.
- 1-bit: accept/reject only.
- Explicitly handle thresholds, lower/upper bounds, saturation, and boundary decisions. Associate calibration settings with each operating threshold.

## 7. Layer 5 — Shared experiment loop

- The attacker uses only initialization from P and permitted oracle responses.
- Centralize query accounting, budgets, stopping, and logging.
- Count every oracle call for initialization, populations, simplex vertices, restarts, and candidate evaluation. Repeated queries of the same candidate still count.
- Private recovery comparisons are not oracle queries and must never influence attack updates.
- At first acceptance, obtain the internal top-1 identity; compute the median cosine between the same accepted vector and that identity's held-out embeddings. Recovery succeeds when this median reaches the H-derived recovery threshold.
- Record the first acceptance score, acceptance query number, recovery outcome, and total query count. Retain budget-exhausted and blocked failures.
- Separate attacker-visible responses from private evaluator records; do not dump the gallery or private scores for every query.

Repetitions and budget:

- Start with 1,000 oracle queries per run.
- Use 20 repetitions per setting as the initial implementation default. This is a low-cost starting value; the user has not separately fixed an exact repeat count. Do not automatically scale to large repetition counts.
- First generate the real image embeddings, then run a small real-data pilot before the initial batch.
- If many runs exhaust their budgets unsuccessfully, decide whether to increase the query budget based on results. Do not preemptively move to 10,000 queries or 1,000 repetitions.
- Compare methods under equal budgets. Do not merge batches with different budgets without labeling them.

## 8. Layer 6 — Attacks and baseline

Initial stage:

1. Random-perturbation hill climbing: Start from P and improve the visible score/level using normalized perturbations.
2. Uphill-simplex: Initialize simplex vertices from the pool; use reflection, expansion, and contraction based on visible scores/levels. Normalize every candidate.
3. Genetic algorithm: Initialize 20 candidates from P; use visible fitness for selection, spherical crossover, and Gaussian mutation.

- Run these three attacks with float, 8-bit, and 4-bit feedback at two thresholds: `3 × 3 × 2 = 18 settings`.
- Quantized attacks must not use hidden float scores. Make tie behavior, restarts, and step settings explicit in the configuration.
- Do not automatically fall back to decision-based attacks in the initial stage, because that method is deferred.
- The zero-effort baseline submits existing P embeddings unchanged in seeded random order. At first acceptance, evaluate recovery and stop; otherwise stop at budget or pool exhaustion. Pool size is the number of available embeddings, not identities.
- Since zero-effort acceptance decisions are independent of feedback precision, run the baseline per threshold and reuse it in the relevant comparisons.

Decision-based attacks are a later stage:

- Boundary/HSJA-style methods require an accepted initial vector; their main search follows this initialization.
- Separately decide how initialization is found/provided, how its cost is counted, the target identity, and the post-acceptance stopping rule.
- Do not automatically apply the other attacks' first-acceptance stopping rule to this method.
- Defer 1-bit attack experiments to this stage; implement 1-bit oracle support initially.
- The eventual full scope may be `18 + 4 × 2 = 26 settings`. Do not claim all 26 settings are complete in the initial stage.

Add defenses after the basic experiments work: score coarsening comparisons, per-account and internal top-1 template query caps, and detection of similar query sequences. Because no separate development partition is used, separately decide detector settings/data and adaptive experiment scope when entering this stage; do not tune on G held-out data.

## 9. Layer 7 — Results and validation

- Report acceptance and recovery at first acceptance separately; explicitly state whether a recovery rate uses all runs or accepted runs as its denominator.
- Produce both success-versus-query curves and include zero-effort comparisons.
- Retain unsuccessful runs. If median/p90 is not reached within budget, report `> budget / not reached`; do not present successful-only medians as overall outcomes.
- No confidence intervals; acknowledge that few repetitions limit interpretation of small differences.
- Report H baseline score distributions and DET/ROC outputs, plus the standard LFW verification result.
- Derive results from measured records; distinguish published results and planned experiments from actual measurements.

Use short, meaningful checks for identity separation, enrollment/held-out separation, embedding mapping and norms, correct 1:N/1:1 scores, quantizer decision equivalence, absence of private information leakage, exact query accounting, and stopping at first acceptance. Do not build excessive implementation-mirroring tests or unnecessary test infrastructure.

## 10. Implementation order

1. Environment/configuration and one data index.
2. Embedding extraction and final G/P/H roles.
3. NumPy verifier, H calibration, quantizer, and LFW baseline check.
4. Shared loop, zero-effort, and recovery at first acceptance.
5. A small end-to-end random hill climbing experiment.
6. Uphill-simplex and the genetic algorithm; the initial 18-setting batch.
7. Inspect results and increase query budgets if needed.
8. Separately define and implement the decision-based protocol.
9. Defenses and final report.

End each stage with a small working result. Do not launch large experiments, model downloads, or additional scope as part of the request to write these plan files.
