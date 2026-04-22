# Literature Review — Multimodal Turn-Taking (Idea 1)

**Date:** 2026-04-21 (CSCI-535, USC, Spring 2026)
**Research question:** Does adding OpenFace-derived facial behavior to a HuBERT audio encoder measurably extend the prediction horizon of end-of-turn detection in dyadic conversation, relative to an audio-only baseline?
**Deliverable owner:** Lit-review session (sibling to data/feature-extraction session).
**Primary metric under review:** accuracy / F1 vs. prediction-horizon τ (curve, not point).
**Scope:** 2024–2026 currency required; earlier work only where foundational.

---

## Item 1 — 2024–2026 novelty sweep: has "horizon gain from face" been published?

**Sub-questions.** (a) Is there a 2024–26 paper combining VAP-family audio baseline + OpenFace-family visual + a horizon/anticipation metric on a dyadic conversational corpus? (b) Do multimodal VAP successors report horizon explicitly, or only point-τ accuracy / hold-shift? (c) Any negative finding — face did *not* help?

**Verdict up front.** The exact claim "multimodal turn-taking beats audio-only" is *no longer novel* in 2024–2026. The *narrower* claim — horizon-gain-from-face on a face-to-face dyadic corpus with a permuted-dyad control and an explicit τ-curve — is *close-but-not-exact*. The closest prior work (MM-VAP, Russell & Harte, ACL Findings 2025) uses OpenFace AUs + gaze + head pose and stratifies by silence duration; that is the same intent as our horizon metric, on a different corpus (Candor videoconferencing), without a permuted-dyad control, and without a HuBERT-GRU baseline.

### Core citations

**[1.1] Russell, S. O'C. & Harte, N. (2025). *Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction.* Findings of ACL 2025, pp. 209–221 (arXiv:2505.21043).**
*Summary.* Introduces MM-VAP: a VAP-style predictive turn-taking model augmented with OpenFace action units, head pose, and gaze, trained on the Candor videoconferencing corpus. Reports 84% hold/shift accuracy vs. 79% for audio-only VAP. Ablation shows facial-expression AUs are the dominant contributor; gaze in isolation does not help. Groups holds/shifts by inter-turn silence duration — functionally equivalent to stratifying by short- vs. long-horizon transitions.
*What it contributes to our design.* This is the closest published prior art and the principal positioning challenge for our research question. Differences we can legitimately lean on: Candor is remote videoconferencing (≠ Seamless face-to-face, co-located); no permuted-dyad control; silence-duration stratification is not the same artifact as a τ-swept accuracy/F1 curve evaluated at future-window onsets; and their visual+audio fusion goes into the VAP backbone (not a HuBERT-GRU). Our design decision: keep the project, but re-frame contribution explicitly as (i) replicating MM-VAP's direction of effect on a *face-to-face* corpus, (ii) adding the permuted-dyad dyadic-coordination control they do not run, and (iii) reporting horizon as a curve rather than a silence-duration stratification.

**[1.2] Russell, S. O'C. & Harte, N. (2025). *Visual Cues Support Robust Turn-taking Prediction in Noise.* Interspeech 2025, pp. 1073–1077 (DOI 10.21437/Interspeech.2025-668; arXiv:2505.22088).**
*Summary.* Follow-up to [1.1] showing audio-only PTTMs collapse from 84%→52% hold/shift accuracy at 10 dB music noise, while the multimodal variant retains 72%. Reports that successful training depends on accurate transcription — ASR-derived transcripts degrade in noise.
*What it contributes to our design.* Establishes that the "visual helps" result is not a one-paper fluke; the same authors re-confirm in a second venue. For us, this tightens the novelty discussion: "face helps turn-taking" is now a settled empirical claim in the Candor/VAP community. Also gives cover for our caveat about WhisperX reliability (Item 4) — they also flag ASR-transcript fragility.

**[1.3] Saga, T. & Pelachaud, C. (2025). *Voice Activity Projection Model with Multimodal Encoders.* arXiv preprint arXiv:2506.03980.**
*Summary.* Proposes a VAP model enhanced with pretrained audio and face encoders (replacing hand-crafted OpenFace AUs). Claims competitive/better performance versus the prior multimodal VAP SOTA (Hisada et al., [1.4]). Released code at github.com/sagatake/VAPwithAudioFaceEncoders.
*What it contributes to our design.* Evidence that the research community is already iterating *past* the MM-VAP formulation we would be replicating — the frontier has moved to learned face encoders. We should cite this to acknowledge our OpenFace-features choice is a principled *simplicity* decision (CSCI-535 compute budget, interpretable AU ablations), not the current SOTA feature pipeline.

**[1.4] Hisada, S., Inoue, K., Lala, D., Nakamura, S. & Kawahara, T. (2024). *Multimodal Voice Activity Projection for Turn-taking and Effects on Speaker Adaptation.* IEICE Transactions on Information and Systems, Vol. E107-D, 2024HCP0002 (peer-reviewed journal).**
*Summary.* First peer-reviewed multimodal-VAP paper. Adds gaze direction, facial action units, head pose and articular keypoints to a VAP backbone; reports per-participant speaker adaptation improvements. Predates MM-VAP.
*What it contributes to our design.* Confirms the multimodal-VAP line is at least two years old in peer-reviewed form. If we want to cite the foundational multimodal-VAP citation as "Hisada et al., 2024, IEICE" (peer-reviewed) rather than the ACL Findings paper, this is the right anchor. Also validates our OpenFace-AU + head-pose feature set as the modal choice in this literature.

**[1.5] Heo, J. et al. (2025). *Gaze-Enhanced Multimodal Turn-Taking Prediction in Triadic Conversations.* Interspeech 2025 (arXiv:2505.13688).**
*Summary.* Extends VAP to 3-party settings; integrates gaze with a speaker-localization module under a spatial constraint. Reports gaze from a single user already improves prediction; multi-user gaze improves further. Lightweight relative to MM-VAP.
*What it contributes to our design.* Narrows positioning: they claim triadic novelty, we can claim dyadic + permuted-control novelty. Also demonstrates that in 2025 gaze is a first-class modality — we should include gaze features (OpenFace eye-gaze vectors) alongside AUs + head pose, not rely on AUs alone.

**[1.6] Lee, M.-C. & Deng, Z. (2025). *Learning Multimodal Motion Cues for Online End-of-Turn Prediction.* ICMI 2025 (graphics.cs.uh.edu/wp-content/papers/2025/2025-ICMI-LearningMultimodalCuesforEOT.pdf). Related: same authors, ICMI 2024, *Online Multimodal End-of-Turn Prediction for Three-party Conversations*.**
*Summary.* Triadic end-of-turn model using audio, gaze, interlocutor-state vector, and gestural backchannel features. Reports hand-motion + gestural backchannel jointly improve EOT classification.
*What it contributes to our design.* Evidence that the non-speech modality that carries signal isn't strictly face — body/gesture also contributes. We do not plan to include body motion in the POC (SMPL-H is expensive); this paper is why the scope section of the report must explicitly own that limitation. It also provides precedent for using "online" / streaming EOT framing, which is cleaner than offline F1 at a single τ.

### Flags

- **Close-but-not-exact prior art exists (MM-VAP; Hisada et al.).** If we are asked "has someone already published this?" — yes, for face-helps-turn-taking in general. The τ-curve-on-face-to-face-Seamless-with-permuted-control framing is still unclaimed. We MUST position against MM-VAP explicitly in the report; claiming simple novelty will get us flagged.
- **The research community has already iterated past the MM-VAP formulation.** Saga & Pelachaud's 2025 preprint replaces OpenFace with a learned face encoder. Our OpenFace-AU pipeline is *already one iteration behind the frontier* as of June 2025. This does not kill the POC (interpretability + compute budget are legitimate defenses) but reviewers will notice.
- **Facial expression, not gaze, is the dominant visual contributor in MM-VAP's ablation.** This is good news — our plan to include 17 AUs + head pose aligns with the effective feature. But it contradicts the Kendon-1967 "gaze-return is the canonical turn-ending cue" framing (see Item 3). Our literature synthesis must reconcile this.
- **No negative findings surfaced.** I did not find a 2024–26 paper where adding face *hurts* turn-end prediction. Absent a counterexample, we must assume the expected effect exists — which raises the stakes on the permuted-dyad control (if face helps *both* real and permuted dyads equally, we have a monadic-not-dyadic effect, which is scientifically the more interesting finding).
- **Horizon-as-curve framing is not dominant in 2024–26 VAP literature.** VAP itself emits a probability distribution over future windows; MM-VAP reports silence-duration-stratified point accuracy. Our τ-curve framing is a positioning opportunity (we'd argue we're standardizing the horizon reporting) and a positioning risk (reviewers may ask why we're not reporting the VAP-native probability-mass metric). See Item 2.

---

## Item 2 — Prediction-horizon metric: justification vs. frame-level F1

**Sub-questions.** (a) What operational definitions of horizon/anticipation are active in 2024–26 (VAP-native projection windows, Skantze-2017 future-window classification, Levinson-Stivers ~200 ms psycholinguistic gap)? (b) Is there a citable *advocate* for the τ-swept F1 curve in 2024–26, or are we reviving a 2017-era idea? (c) Has horizon-as-curve been critiqued or displaced by the VAP-native probability-mass output?

**Verdict up front.** The horizon/anticipation-window idea is foundational (Skantze 2017; Roddy et al. 2018) but has *evolved*, not disappeared. In 2022+ the community converged on VAP's projection-window probability distribution as the native metric. 2024–26 VAP-family papers do not typically publish a τ-curve; they publish (i) hold/shift accuracy at a single onset, often stratified by silence duration, or (ii) zero-shot task F1 at VAP's fixed 2 s projection window. The τ-curve is thus *defensibly underused*, not *cleanly replaced*. We can legitimately argue for it as a report-level reporting choice, but we must not claim it is the field's standard.

### Core citations

**[2.1] Skantze, G. (2021). *Turn-taking in Conversational Systems and Human-Robot Interaction: A Review.* Computer Speech & Language, 67, 101178.**
*Summary.* The post-2020 review of turn-taking for spoken dialogue. Frames end-of-turn prediction as an "anticipation problem" and distinguishes reactive (IPU / silence-threshold) vs. predictive (future-window classification) approaches. Explicitly argues predictive models should be evaluated at multiple lookahead windows.
*What it contributes to our design.* Best single citation to anchor the τ-swept reporting choice. Skantze-2021 is the review we cite when we want to defend the horizon-curve framing against "why not just one τ?" pushback. It is also published in a peer-reviewed journal, which pushes the citation weight above pure arXiv.

**[2.2] Ekstedt, E. & Skantze, G. (2022). *Voice Activity Projection: Self-supervised Learning of Turn-taking Events.* Interspeech 2022, pp. 5190–5194 (arXiv:2205.09812).**
*Summary.* Defines the VAP objective — self-supervised prediction of a probability distribution over discrete joint voice-activity states in a 2 s projection window. Reports four zero-shot downstream tasks (turn-shift prediction, short-vs-long prediction, backchannel prediction, overlap). Projection-window probability *is* the native metric; τ is baked in at 2 s with 4 sub-windows.
*What it contributes to our design.* The alternative metric we must explicitly address. VAP's probability-mass-over-future-states is cleaner than a τ-curve but requires the VAP head, which we do not plan to train (our baseline is HuBERT features → GRU → hold/shift). Our report has to say one of two things: (a) "we deliberately use a τ-curve because our POC does not adopt the VAP head," or (b) "we adopt VAP's projection-window metric on top of our HuBERT-GRU for commensurability." Option (a) is cheaper; option (b) is more defensible.

**[2.3] Skantze, G. (2017). *Towards a General, Continuous Model of Turn-taking in Spoken Dialogue using LSTM Recurrent Neural Networks.* SIGDIAL 2017, pp. 220–230 (ACL Anthology W17-5527).**
*Summary.* Proto-horizon-metric paper: LSTM predicts upcoming speech-activity probability at multiple future windows simultaneously and reports F1 as a function of window placement. Establishes that human observers are worse than the model at pause-based turn-shift prediction.
*What it contributes to our design.* The foundational reference for the τ-curve reporting style we're adopting. Cite this, together with Roddy 2018, as the method lineage we are reviving and adapting for a multimodal 2026 setting. Pre-VAP but in 2024–26 literature it is still cited whenever a paper wants an anticipation-window framing.

**[2.4] Roddy, M., Skantze, G. & Harte, N. (2018). *Multimodal Continuous Turn-Taking Prediction Using Multiscale RNNs.* ICMI 2018, pp. 186–190 (arXiv:1808.10785).**
*Summary.* Extends Skantze 2017 to multimodal features with per-modality LSTMs at independent timescales fused by a master LSTM. Evaluates on continuous future-window speech-activity prediction and is the canonical "continuous TT" citation.
*What it contributes to our design.* (a) It is the lineage ancestor of MM-VAP (Harte co-authors both), so we can cite it alongside MM-VAP to show our framing is continuous with a single 8-year research line. (b) It validates multimodal fusion *at the window-prediction level* — i.e., the exact thing our τ-curve is measuring.

**[2.5] Inoue, K., Lala, D., Nakamura, S. & Kawahara, T. (2024). *Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection.* IWSDS 2024 (arXiv:2401.04868).**
*Summary.* VAP deployed real-time (streaming) with strict latency budgets. Shows VAP can anticipate end-of-turn and backchannel timings up to ~400 ms in advance with usable accuracy.
*What it contributes to our design.* (a) Direct evidence that 400 ms is an operational anticipation horizon in 2024 VAP deployments — this anchors the upper-end of our τ sweep ({100, 200, 400, 800, 1600 ms}) at the known-useful-in-practice point. (b) Gives an explicit citation-supported τ for the report's "what does this amount of horizon gain mean in operational terms?" paragraph.

**[2.6] Levinson, S. C. & Torreira, F. (2015). *Timing in turn-taking and its implications for processing models of language.* Frontiers in Psychology, 6, 731.**
*Summary.* Psycholinguistic foundation for the ~200 ms modal inter-turn gap across languages. Responses this fast require anticipation, not reaction, because language production latency exceeds 600 ms.
*What it contributes to our design.* The psycholinguistic anchor for τ = 200 ms as a non-arbitrary choice. We cite Levinson & Torreira once in the metric-motivation paragraph to justify why 200 ms is on the τ sweep at all. Do not over-cite — this is one-sentence material.

### Flags

- **The τ-curve is underused in 2024–26 VAP literature — but not repudiated.** The community shifted toward VAP's projection-window probability distribution because it's the model's native output. Our τ-curve will look old-school unless we explicitly justify it as a metric-at-report-time design choice appropriate to a model that does not have the VAP head. Acceptable defense; must be stated.
- **We do not implement VAP, so we cannot publish the VAP-native metric without adding that architecture.** Reviewers will likely ask "why no VAP-native metric?" The cheap mitigation is reporting hold/shift accuracy at a single τ (e.g., τ = 500 ms) alongside our τ-curve, so the numbers are commensurable with MM-VAP. Skipping this comparison is a real risk.
- **400 ms and 200 ms are non-arbitrary τ values.** Inoue et al. 2024 anchors 400 ms (operational VAP anticipation), Levinson & Torreira 2015 anchors 200 ms (psycholinguistic modal gap). Our τ grid {100, 200, 400, 800, 1600 ms} is therefore theoretically motivated on two of its points. The 100 ms, 800 ms, 1600 ms points should also be justified in a sentence each; otherwise the grid reads as ad hoc.
- **No paper in 2024–26 explicitly critiques horizon-as-curve as misleading.** So we are not walking into a known landmine — just a metric the field has drifted past.
- **The metric choice interacts with the label schema.** Our four-class labels (HOLD, YIELD, BACKCHANNEL, INTERRUPT) are derived from discrete events, while VAP's metric lives in a continuous probability space. Reporting F1 per class over τ makes the heterogeneity transparent; a single "hold/shift accuracy" rolled-up number would hide INTERRUPT base-rate effects (see Item 5). Favor the per-class τ-curve.

---

## Item 3 — Visual turn-taking cues: is Kendon's (1967) gaze-return still canonical?

**Sub-questions.** (a) Has the Kendon-1967 "speaker averts gaze during own turn and returns gaze at turn-end" claim been re-tested, refined, or contradicted in 2020+ work? (b) Which specific visual features does recent empirical work treat as load-bearing for turn-end prediction — AUs, head pose, gaze, eyelid, mouth-preparatory movement, blinks? (c) Does any 2024–26 finding recommend a feature that OpenFace does not expose, forcing us to revise the feature pipeline?

**Verdict up front.** Kendon's gaze-return claim survives in a *qualified* form: gaze coordination does facilitate turn-yielding, but 2020s empirical work — especially Degutyte & Astell 2021's systematized review and Kendrick, Holler & Levinson 2023 — shows the effect is smaller, noisier, and more context-dependent than the 1967 framing implied. More importantly, 2021+ Donders/MPI work (Nota, Trujillo & Holler) establishes that *facial* signals (eyebrow movements, mouth movements, head tilts) carry independent turn-boundary information, and MM-VAP's 2025 ablation confirms this computationally: facial-expression AUs are the dominant visual contributor, gaze alone does not help. The Kendon gaze-return line should be cited as foundational background but not as the load-bearing theoretical anchor for our feature choice.

### Core citations

**[3.1] Kendon, A. (1967). *Some functions of gaze-direction in social interaction.* Acta Psychologica, 26, 22–63.**
*Summary.* The foundational empirical claim: in dyadic conversation, speakers look away from their addressee while taking the floor and planning speech, and return gaze to the addressee just before yielding the turn. Establishes gaze-aversion-then-return as a turn-regulatory signal.
*What it contributes to our design.* Historical anchor for why we include OpenFace gaze vectors at all. Cite in the introduction/background; do NOT cite as current empirical support because the effect is weaker than originally reported (see [3.2]).

**[3.2] Degutyte, Z. & Astell, A. (2021). *The Role of Eye Gaze in Regulating Turn Taking in Conversations: A Systematized Review of Methods and Findings.* Frontiers in Psychology, 12, 616471.**
*Summary.* Systematic review of 29 studies of eye gaze in conversational turn-taking in healthy adults. Consolidates findings: gaze facilitates turn-*yielding* (consistent with Kendon), plays a role in speech monitoring, and modulates interruption. For *turn-initiation*, however, findings are inconsistent across studies — no reliable gaze-return-as-cue-to-next-speaker effect.
*What it contributes to our design.* The one-citation answer to "is Kendon still canonical?" — yes for yielding, no/unclear for initiation. Also documents the heterogeneity of gaze measurement across studies, which explains why the computational multimodal-VAP literature (MM-VAP, Heo 2025) treats gaze as a weak-to-moderate feature whose value depends heavily on how it's integrated.

**[3.3] Kendrick, K. H., Holler, J. & Levinson, S. C. (2023). *Turn-taking in human face-to-face interaction is multimodal: gaze direction and manual gestures aid the coordination of turn transitions.* Philosophical Transactions of the Royal Society B, 378(1875), 20210473.**
*Summary.* 10 dyadic English conversations from the Eye-tracking in Multimodal Interaction Corpus (EMIC). Finds that *gaze aversion* at a point of possible turn completion suppresses speaker change (addressee holds off) and that unfinished gestures-in-progress likewise inhibit transitions. Reframes Kendon's claim as a bidirectional coordination signal: gaze direction *gates* transition rather than signalling it.
*What it contributes to our design.* The 2020s canonical citation for gaze in face-to-face (not videoconferencing) turn-taking. Seamless is face-to-face co-located, so this is more empirically relevant than Candor-based MM-VAP for our gaze-feature defense. Provides precedent for framing gaze as an inhibitory/gating signal, which is exactly how a classifier would pick it up (low gaze-to-partner ≈ lower P(yield)).

**[3.4] Nota, N., Trujillo, J. P. & Holler, J. (2021). *Facial Signals and Social Actions in Multimodal Face-to-Face Interaction.* Brain Sciences, 11(8), 1017.**
*Summary.* Annotated 6,778 questions and 4,553 responses from the CoAct corpus for 12 facial signals (eyebrow raises/frowns, mouth actions, head tilts, blinks, gaze). Reports that mouth movements, eyebrow raises, and unilateral brow raises occur relatively late in utterances, consistent with turn-end marking. Follow-up work (Nota et al. 2023, Scientific Reports; Nota et al. 2023, PLOS ONE) shows eyebrow frowns facilitate question identification and that specific facial signals associate with categories of social action.
*What it contributes to our design.* The strongest 2020s empirical case that *specific facial features* carry end-of-turn information independent of gaze. Directly motivates including OpenFace's AU01/02/04 (brow raise/frown), AU25/26 (mouth/jaw), and head-pose angles in the feature vector. Cite this as justification for our AU list against a "why not just gaze?" challenge.

**[3.5] Russell, S. O'C. & Harte, N. (2025). *Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction.* Findings of ACL 2025 (already cited as [1.1]).**
*Summary (for this item).* The key computational datapoint for feature-importance: facial expression AUs are the *dominant* contributor to MM-VAP's hold/shift gain; gaze in isolation does not improve the model; head pose contributes modestly. Direct empirical ranking that aligns feature priority for our POC.
*What it contributes to our design.* Tells us where to invest feature-engineering effort: AUs > head pose > gaze. If we have to trim features for compute, gaze-only features are the first to go. Also this is a reason to not over-invest in gaze-specific preprocessing (e.g., head-pose-compensated gaze vectors) relative to AU-stream quality.

**[3.6] Trujillo, J. P. & Holler, J. (2023). *Interactionally Embedded Gestalt Principles of Multimodal Human Communication.* Perspectives on Psychological Science, 18(5), 1136–1159.**
*Summary.* Theoretical synthesis arguing that multimodal human communication is organized as interactionally-embedded gestalts — gaze, facial signals, gestures, and prosody combine in context-sensitive configurations rather than acting as independent additive cues. Reviewed by same author as [3.4] and frequently co-cited with Kendrick et al. 2023.
*What it contributes to our design.* Theoretical cover for late fusion: if the visual modality is gestalt-organized, early fusion of AUs with audio at each frame may recover dyadic coordination better than separate unimodal heads. Cite briefly in methods to justify early/mid-fusion architecture choice.

### Flags

- **Kendon's gaze-return claim is substantially weakened in 2020s empirical work.** Degutyte & Astell 2021 find gaze helps *yielding* but not initiation, and Kendrick et al. 2023 reframe the effect as inhibition/gating. Our report must not lean on Kendon as the theoretical load-bearer — that citation is background only.
- **Tension between computational and behavioral literature on gaze.** MM-VAP says gaze-only features do not help, behavioral work (Kendrick 2023) says gaze modulates transitions. Resolution is probably measurement: MM-VAP uses OpenFace eye-gaze vectors averaged over windows, behavioral work uses hand-coded fixation targets. Our OpenFace pipeline is closer to MM-VAP's; expect gaze to look weak empirically even though theory predicts it matters. Flag explicitly in results.
- **OpenFace does not expose manual gestures.** Kendrick et al. 2023 find that *gestures-in-progress* also inhibit turn transitions. We have SMPL-H motion available in Seamless but are not using it in the POC. This is a known scope limitation the report must own, especially given gesture is an independent predictor in behavioral work.
- **Mouth/jaw AUs are likely the single highest-leverage feature.** Nota et al. 2021 find mouth movements appear late in utterances (turn-end-adjacent) and MM-VAP's ablation ranks facial expression as dominant. If the POC must trim features to fit compute, keep AU25/26 (lips parted, jaw drop) + AU12/15 (lip corner puller/depressor) and cut gaze vectors before cutting these. **Actionable feature-priority ranking: AUs > head pose > gaze.**
- **No 2024–26 finding demands a feature OpenFace does not expose.** Gaze, AUs, head pose are all in OpenFace. Optional additions the literature flags but we will defer: micro-expression timing, upper-body gesture (SMPL-H), blink rate dynamics. None of these are blocking for the POC.
- **Face-to-face vs. videoconferencing generalization is a live issue.** Kendrick 2023 studies face-to-face; MM-VAP studies videoconferencing (Candor). Seamless is face-to-face co-located. If our result aligns with Kendrick (gaze matters, gestures matter) but diverges from MM-VAP (where gaze in isolation did not help), that's scientifically interesting and defensible. If our result closely tracks MM-VAP, we've replicated on a new face-to-face corpus — also a contribution. Either outcome is publishable-at-course-scope; no existential risk here.

---

## Item 4 — WhisperX / short-utterance timestamp alignment accuracy

**Sub-questions.** (a) How large is WhisperX's timestamp error on short utterances and backchannels specifically? (b) Has CrisperWhisper (Wagner et al. 2024) or another successor measurably improved short-utterance timestamp accuracy? (c) Given Seamless distributes pre-computed WhisperX, what is the minimum defensible story for our BC-class labels in the report?

**Verdict up front.** WhisperX's documented weakness on short utterances is real, measurable, and specifically harmful to backchannel labels because backchannels *are* the short-utterance class. CrisperWhisper (Wagner et al., Interspeech 2024) directly addresses this by improving verbatim timestamping and explicitly targets the timed detection of filler events, which are the acoustic cousins of backchannels. For the POC, we can stay on Seamless's bundled WhisperX (no re-transcription cost), but we must (i) quantify empirical short-utterance error on a 30–60 s hand-annotated slice of one POC dyad, (ii) cite CrisperWhisper as the mitigation path if BC-class F1 is structurally low, and (iii) report the Seamless paper's own Appendix A.1.4 3σ-outlier warning as the known limitation. No paper found treats short-utterance timestamp error as a killer for downstream turn-taking research — Ekstedt & Skantze 2022 (VAP) and MM-VAP both use ASR-derived text without full alignment verification.

### Core citations

**[4.1] Bain, M., Huh, J., Han, T. & Zisserman, A. (2023). *WhisperX: Time-Accurate Speech Transcription of Long-Form Audio.* Interspeech 2023 (arXiv:2303.00747).**
*Summary.* The WhisperX paper itself. Combines Whisper transcription with a wav2vec2-based forced-alignment pass for word-level timestamps and a VAD-based chunking pass for long-form audio. Reports per-word timestamp accuracy in the ~20–40 ms range on read-speech benchmarks. Does not report a separate short-utterance breakdown.
*What it contributes to our design.* The citation we use when naming what Seamless bundles. Cite here as the upstream tool and then pivot to Appendix A.1.4 of the Seamless paper for the short-utterance caveat.

**[4.2] Seamless Interaction (Meta, 2025), Appendix A.1.4, p. 54.**
*Summary (in our corpus).* Meta's own evaluation of their bundled WhisperX transcripts reports that 98% of sessions and 87% of interactions contain ≥1 word whose timestamp-derived length is >3σ from mean — i.e., short-utterance outliers are the rule, not the exception. Paper explicitly positions these transcripts as adequate for "preliminary turn-taking analyses in the spirit of Heldner & Edlund 2010" but not for fine-grained phone-level analysis.
*What it contributes to our design.* The one citation that makes our "we are using them as-is with documented caveat" position defensible. Cite in methods immediately after introducing WhisperX as the source of BC-class labels. Heldner-Edlund-style preliminary analysis is explicitly our scope; that sentence in A.1.4 is load-bearing cover.

**[4.3] Wagner, L., Zusag, M. & Bleeker, T. (2024). *CrisperWhisper: Accurate Timestamps on Verbatim Speech Transcriptions.* Interspeech 2024 (arXiv:2408.16589).**
*Summary.* Fine-tuned Whisper variant with a re-tokenized architecture that produces verbatim speech transcriptions (including fillers) and applies dynamic time warping to decoder cross-attention scores for word-level timestamps. Benchmarks on TIMIT, LibriSpeech, AMI. Explicitly targets the timed detection of filler events, which is a direct technical analog to backchannel timing.
*What it contributes to our design.* The named mitigation path if our empirical short-utterance error characterization is bad enough to undermine BC-class F1. The report section on "limitations and future work" should cite CrisperWhisper as the clear upgrade. If we have compute left at the end of the POC, re-running one dyad through CrisperWhisper and measuring label agreement with Seamless's WhisperX transcripts would be a strong extension.

**[4.4] Yeh, S.-L., Meng, Y. & Tang, H. (2025). *Whisper Has an Internal Word Aligner.* arXiv:2509.09987.**
*Summary.* Shows that Whisper's own cross-attention dynamics encode internal word boundaries at high temporal resolution, sometimes outperforming external alignment passes (WhisperX's wav2vec2 step, CrisperWhisper's retokenized DTW). Reports head-level selection improves word-level alignment on standard benchmarks.
*What it contributes to our design.* Optional extension citation, not load-bearing. We will not implement it for the POC. Mention once in limitations as "word-alignment research is active in 2025; Seamless's WhisperX snapshot is one generation behind the 2024–25 frontier."

**[4.5] Cangemi, F., Niebuhr, O. & Cwiek, A. (2025). *What automatic speech recognition can and cannot do for conversational speech transcription.* Speech Communication.**
*Summary (journal review).* Systematically documents ASR failure modes on conversational speech: backchannels (mm-hmm, uh-huh) are particularly prone to being dropped, misspelled, or merged with adjacent words; timestamp accuracy on these tokens is noticeably worse than on content words. Recommends routine auditing of cue lists and confidence thresholds when using ASR output for downstream conversational analysis.
*What it contributes to our design.* The 2025 peer-reviewed statement that ASR backchannel reliability is a known, documented limitation — not something we're uniquely encountering. Cite in limitations. Also motivates reporting BC-class F1 separately from HOLD/YIELD/INTERRUPT F1 rather than rolling into a single score: if BC F1 is poor, we want that visibly attributable to the ASR pipeline, not to our classifier.

**[4.6] Amoyal, C., Bigi, B. & Priego-Valverde, B. (2024). *Annotation of Transition-Relevance Places and Interruptions for the Description of Turn-Taking in Conversations in French Media Content.* LREC-COLING 2024.**
*Summary.* Hand-annotated TRPs and interruptions in French conversational corpora; compares automatic against manual annotations and quantifies the tooling gap on short speech events. Relevant comparable-cost reference point for our optional 30–60 s hand-annotation sanity check.
*What it contributes to our design.* Gives us a published precedent and methodology for the hand-annotation sanity pass listed in `project_turn_taking_validity_risks.md` Risk 1. We can cite this as the methodological template and reuse a subset of their annotation scheme (TRPs + interruptions) if we do the sanity check.

### Flags

- **Short-utterance timestamp error is real but not a killer for POC scope.** The Seamless paper A.1.4 warning is a warning, not a disqualification. Heldner-Edlund-style preliminary turn-taking analysis is explicitly what Seamless recommends the transcripts for, and that is our scope.
- **BC-class F1 may be structurally lower than HOLD/YIELD F1 because of label noise, not model capacity.** Cangemi et al. 2025 confirms this is an ASR-level limitation; our result reporting must separate per-class F1 so this is visible, not hidden in a macro-average.
- **CrisperWhisper is the named upgrade path.** If BC F1 is unacceptably low and we have compute, swapping Seamless's WhisperX transcripts for CrisperWhisper on the POC's 24 dyads is plausible at POC scale (not at 45k-interaction scale). Cost estimate is small — Whisper-family models run faster than real time on T4.
- **Do not claim word-level phonetic precision from WhisperX.** Constrain language in the report: "coarse word-onset times adequate for backchannel-class labeling at 10 Hz frame resolution," not "precise onsets."
- **The 30–60 s hand-annotation sanity check is cheap and disproportionately valuable for integrity.** Time cost ≈ 30–60 min in Praat; gives the report a quantitative paragraph that preempts reviewer skepticism. Recommended — flagged in Risk 1 in the memory, now reinforced by Amoyal 2024 as a validated methodological template.
- **No 2024–26 paper was found that claims WhisperX short-utterance error makes downstream turn-taking analysis invalid.** Absent such a paper, we are on safe literature footing by reporting the known limitation and proceeding.

---

## Item 5 — Treatment of turn-competitive overlap / interrupted turns

**Sub-questions.** (a) How do 2022–26 papers label or filter overlapping turns? Is INTERRUPT an accepted fourth class or is it absorbed into a continuous variable / filtered out? (b) What base rate does overlap have in conversational corpora and therefore in Seamless? (c) Does recent work distinguish competitive from cooperative overlap, and should our label schema?

**Verdict up front.** Overlap is frequent (~40% of transitions in Heldner & Edlund 2010 across corpora; similar orders of magnitude in later corpora) and the 2024–26 literature does *not* converge on filtering it. Instead, the field has drifted toward finer-grained labeling: competitive vs. cooperative, or 4+ intent categories (disruptive / cooperative-agreement / assistance / clarification). INTERRUPT as a fourth class is defensible at POC scope, but we must expect base-rate imbalance (INTERRUPT is the rarest of the four) and probably can't differentiate competitive from cooperative without per-turn hand labels. Report F1 per class; do not collapse INTERRUPT into HOLD/YIELD/BACKCHANNEL.

### Core citations

**[5.1] Heldner, M. & Edlund, J. (2010). *Pauses, gaps and overlaps in conversations.* Journal of Phonetics, 38(4), 555–568.**
*Summary.* Foundational quantitative framework: "between-speaker intervals" (BSI) with negative values = overlap, positive = gap. Across Dutch, English, Swedish corpora, ~40% of speaker transitions involve overlap ≥10 ms; the modal overlap is <50 ms; true zero-gap transitions are <1%. Concludes the idealized "one-speaker-at-a-time, no-gap-no-overlap" target is wrong.
*What it contributes to our design.* (a) Base-rate anchor: expect ~40% of YIELD transitions in Seamless to be overlap-including. Our INTERRUPT label uses Seamless's `overlapping=True` flag, so the overlap rate feeds directly into class balance. (b) Conceptual cover: we are not pathologizing overlap by labeling it; overlap *is* normal conversational turn-taking. Cite in methods when introducing the four-class label schema.

**[5.2] Levinson, S. C. & Torreira, F. (2015). *Timing in turn-taking and its implications for processing models of language.* Frontiers in Psychology, 6, 731. (Already cited as [2.6].)**
*Summary (for this item).* ~200 ms modal gap across languages; implications for overlap = responses faster than ~200 ms are either anticipatory or competitive. Provides psycholinguistic baseline against which our INTERRUPT frequency can be interpreted.
*What it contributes to our design.* If INTERRUPT-rate in Seamless is substantially higher than Heldner-Edlund-style overlap rate on comparable corpora, that's suspicious — possibly VAD artefact. Cite Levinson-Torreira for the timing anchor.

**[5.3] Amoyal, C., Bigi, B. & Priego-Valverde, B. (2024). *Annotation of Transition-Relevance Places and Interruptions for the Description of Turn-Taking in Conversations in French Media Content.* LREC-COLING 2024. (Already cited as [4.6].)**
*Summary (for this item).* Hand-annotates both TRPs and interruptions separately in French media corpora. Demonstrates that "interruption" is a distinct annotation category from "turn transition", even though current automated tooling mostly collapses them. Provides a taxonomy that distinguishes interruptions-at-TRP from interruptions-mid-TCU.
*What it contributes to our design.* Supports the INTERRUPT-as-fourth-class decision: the annotation community does keep interruption distinct. Also suggests a more fine-grained future axis (interruption-at-TRP vs. mid-turn), which we defer.

**[5.4] Kurtic, E., Brown, G. J. & Wells, B. (2010). *Resources for turn competition in overlap in multi-party conversations.* Interspeech 2010.** (Foundational, still cited in 2024–25 work.)
*Summary.* Identifies acoustic-prosodic resources used by speakers competing for the floor in overlap: raised pitch, raised intensity, slowed articulation. Distinguishes competitive from cooperative/non-competitive overlap.
*What it contributes to our design.* (a) Reminder that competitive vs. cooperative is a real linguistic distinction; our binary INTERRUPT label flattens it. (b) Suggests that if the HuBERT audio encoder has access to prosody (it does — HuBERT preserves pitch/energy contours), the classifier can potentially learn to discriminate competitive interruption without explicit prosodic labels. We do not need to hand-label competitiveness for the POC.

**[5.5] Sell, E. et al. (2025). *InteractSpeech: A Speech Dialogue Interaction Corpus for Modelling Interruptions.* Findings of EMNLP 2025.**
*Summary.* 2025 corpus specifically annotated for interruptions in spoken dialogue. Exists because the authors document that existing dialogue corpora either filter overlap, label it coarsely, or lack per-interruption annotations.
*What it contributes to our design.* Evidence that as of 2025 the field still views interruption annotation as underdone. Our INTERRUPT-class contribution, even at coarse `overlapping=True` grain, is a step toward something that the 2025 community considers worth building a whole corpus for. Cite in positioning.

**[5.6] Doyle, D. & Serban, L. (2024). *Analysing speech interruptions to create more human-like AI chatbots.* Imperial College London / Group Affect and Performance Dataset (GAP) derived corpus.**
*Summary.* 200 manually annotated interruptions extracted from 355 overlapping utterances; categorizes true vs. false interruptions and uses prompt-engineered LLMs to classify interrupter intention into cooperative-agreement, cooperative-assistance, cooperative-clarification, and disruptive. Argues that human-like AI dialog systems need this granularity.
*What it contributes to our design.* Direct 2024 evidence that multi-way interruption classification is a live research direction. For the POC we stay at binary INTERRUPT; for the discussion/future-work section we can cite Doyle & Serban as the path forward if our binary result is promising.

### Flags

- **INTERRUPT will be the rarest class and may drive per-class F1 into the noise.** Overlap rate is ~40% of transitions in Heldner-Edlund, but not all overlaps become INTERRUPT labels under our definition (`overlapping=True` during a particular participant's turn). Expect INTERRUPT to be ~5–15% of POC frames; smaller if the 180 ms inter-word threshold is strict. Acceptable at POC scope but the macro-F1 number may look bad.
- **We do not label competitive vs. cooperative overlap, and we are not going to.** The POC's `overlapping=True` flag is competitive-agnostic. Kurtic 2010 and Doyle 2024 show this is a real axis; we cite them to acknowledge the simplification and propose competitive/cooperative splitting as future work.
- **INTERRUPT may be an artefact of VAD rather than genuine floor-taking.** If Silero VAD occasionally fires on backchannels and we're interpreting that as `overlapping=True` INTERRUPT, we will inflate the INTERRUPT rate. Mitigation: when computing per-frame labels, require the overlapping VAD segment to exceed a minimum duration (e.g., 500 ms) and not be labeled as a backchannel by our lexical rule (Whisper-derived). Worth calling out in methods.
- **Not filtering overlap is consistent with 2024–26 best practice.** No paper found recommends filtering overlap out of the analysis. If anything, 2024–26 work (InteractSpeech, Doyle 2024) moves the other way.
- **Base-rate reporting is mandatory.** Given overlap is ~40% of transitions, our report must show the actual class distribution in the POC (train, val, test) — not hide it in an appendix. A reviewer will otherwise suspect the classifier is picking up base rate rather than signal.
- **HuBERT probably learns prosodic interruption cues without explicit labels.** Kurtic 2010's prosodic-resources finding plus HuBERT's known pitch/energy sensitivity means the audio-only baseline might perform surprisingly well on INTERRUPT via prosody alone. This would *reduce* the expected multimodal horizon gain on INTERRUPT specifically. If horizon-gain-from-face is concentrated in HOLD/YIELD but absent in INTERRUPT, that's a scientifically interesting per-class breakdown that our reporting plan will already show.

---

## Item 6 — Audio-only baseline: VAP vs. from-scratch GRU on HuBERT

**Sub-questions.** (a) Is VAP the default audio-only baseline in 2024–26 turn-taking papers? (b) Do any recent papers use a HuBERT/WavLM → GRU/LSTM head as a legitimate audio-only baseline, or is that treated as outdated? (c) What is the minimum defensible audio-only baseline set for a POC that is not publishing to a TT venue?

**Verdict up front.** VAP (Ekstedt & Skantze 2022 and successors) *is* the de facto audio-only baseline in 2024–26 turn-taking papers. A from-scratch HuBERT→GRU head is weaker than VAP on the published benchmarks and will look like a strawman if it is our *only* audio-only reference. Minimum defensible baseline set for the POC: majority-class + HuBERT→GRU + a publicly available VAP checkpoint run zero-shot on our POC test set. The VAP checkpoint does not need to be retrained; using Ekstedt & Skantze's 2022 release or Inoue et al.'s 2024 multilingual/real-time release and reporting its hold/shift accuracy on our POC at τ = 500 ms is sufficient and cheap.

### Core citations

**[6.1] Ekstedt, E. & Skantze, G. (2022). *Voice Activity Projection: Self-supervised Learning of Turn-taking Events.* Interspeech 2022 (arXiv:2205.09812). (Already cited as [2.2].)**
*Summary (for this item).* The VAP baseline. Self-supervised pretraining on dyadic dialogue; zero-shot evaluation on downstream turn-taking tasks. Public checkpoints via `erikekstedt.github.io/VAP/`.
*What it contributes to our design.* The audio-only reference the community expects us to beat or at least match. Zero-shot evaluation on our POC test set is cheap and publishable-quality as a baseline.

**[6.2] Inoue, K., Jiang, B., Ekstedt, E., Kawahara, T. & Skantze, G. (2024). *Multilingual Turn-taking Prediction Using Voice Activity Projection.* LREC-COLING 2024 (arXiv:2403.06487).**
*Summary.* Trains VAP on English + Mandarin + Japanese; multilingual model is on par with monolingual across all three languages. Establishes VAP as corpus-agnostic and language-robust.
*What it contributes to our design.* Two things: (a) the VAP checkpoint we would use zero-shot on Seamless is validated across corpora, so our zero-shot transfer is methodologically supported; (b) the multilingual finding provides prior for our single-language (English) result — we are not operating at the edge of the model's capability.

**[6.3] Inoue, K., Lala, D., Nakamura, S. & Kawahara, T. (2024). *Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection.* IWSDS 2024 (arXiv:2401.04868). (Already cited as [2.5].)**
*Summary (for this item).* Streaming real-time VAP. The 400 ms anticipation horizon cited in Item 2; architecturally the same model but operationalized in streaming form.
*What it contributes to our design.* Our τ-curve upper end (1600 ms) exceeds this model's reported 400 ms operational anticipation. Cite as the anchor for what "audio-only at short horizon" can do; a good honesty check on our baseline's ceiling.

**[6.4] Skantze, G. (2017). *Towards a General, Continuous Model of Turn-taking in Spoken Dialogue using LSTM Recurrent Neural Networks.* SIGDIAL 2017. (Already cited as [2.3].)**
*Summary (for this item).* Proto-GRU/LSTM baseline — predicts future-window speech activity from hand-engineered features + word embeddings using an LSTM. Comparable architectural class to our HuBERT→GRU.
*What it contributes to our design.* Prior art that validates the LSTM-family architecture for turn-taking prediction. Cite as lineage: our HuBERT→GRU is a modern-features reimplementation of the Skantze 2017 template. This gives the baseline a legitimate intellectual history, not just expediency.

**[6.5] Onishi, K., Inoue, K., Lala, D. & Kawahara, T. (2025). *Prompt-Guided Turn-Taking Prediction.* arXiv:2506.21191 (likely Interspeech 2025 / SIGDIAL 2025).**
*Summary.* Builds on VAP with text-prompt conditioning (LLM-derived turn-taking prior); shows consistent improvement over vanilla VAP. The current frontier on audio-textual turn-taking prediction.
*What it contributes to our design.* Evidence that 2025 is iterating *past* vanilla VAP with LLM-prompted variants. We do not adopt this — it's out of POC scope — but for the report we note that our VAP baseline is the 2022–24 baseline, not the current 2025 frontier. This partially defuses the "you should have used the latest VAP" critique.

**[6.6] Russell, S. O'C. & Harte, N. (2025). *Visual Cues Enhance Predictive Turn-Taking…* ACL Findings 2025 (already cited as [1.1]).**
*Summary (for this item).* MM-VAP reports audio-only VAP at 79% hold/shift; multimodal at 84%. That 79% number is the concrete target our HuBERT→GRU audio-only needs to be in the neighborhood of on comparable hold/shift framing.
*What it contributes to our design.* Numeric anchor. If our HuBERT→GRU audio-only reports <70% hold/shift at τ ≈ 500 ms on Seamless, we know the baseline is below community expectation and needs strengthening (likely via a temporal context model or an actual VAP head trained from scratch).

### Flags

- **HuBERT→GRU alone is probably a weak baseline by 2024–26 standards.** Without a VAP checkpoint comparison, our audio-only baseline is a strawman and reviewers will mark it down. Mandatory addition to the baseline set: zero-shot VAP checkpoint on POC test. Cost is low (inference only, T4-feasible).
- **Zero-shot VAP transfer to Seamless is methodologically defensible.** Inoue et al. 2024 show VAP is corpus- and language-robust. We do not need to retrain VAP for the POC; zero-shot inference on our POC test set is sufficient for a publication-credible audio-only baseline comparison.
- **Our goal is not to beat VAP; it is to show horizon gain from face.** The comparison we actually care about is (HuBERT+AUs)→GRU vs. (HuBERT-alone)→GRU, on the same architecture. VAP serves as an external sanity check that our audio-only reference is in the right ballpark, not as the thing our multimodal model must exceed. State this clearly in the baselines paragraph of the report.
- **VAP's native metric is not our τ-curve.** Reporting VAP numbers at τ = 500 ms (single-point hold/shift) alongside our τ-curve is the cheapest way to make the comparison commensurable. Skipping this leaves the reader with two incommensurable numbers.
- **Prompt-Guided VAP (Onishi 2025) is the 2025 frontier and we are not including it.** This is a legitimate limitation — the frontier has moved. Own it in the discussion section with a sentence of "limitations" coverage; do not hide it.
- **Do not use the majority-class baseline to make HuBERT→GRU look better.** Majority-class in 4-way imbalanced labels will be near 60%+ accuracy just by predicting HOLD. Macro-F1 is the honest comparison; report both but defend decisions based on macro-F1, not accuracy.

---

## Item 7 — Audio encoder choice: HuBERT-base vs. WavLM vs. larger variants

**Sub-questions.** (a) Is HuBERT-base still state-of-practice for conversational prosody encoding in 2024–26, or has WavLM (base/large) displaced it? (b) Does encoder choice plausibly change horizon-prediction results? (c) What is the minimum defensible encoder story for the POC given the T4 / ~280-compute-unit budget?

**Verdict up front.** HuBERT-base is defensible but not optimal for conversational prosody. WavLM-base+ is the 2024–26 benchmark-leader on SUPERB paralinguistic tasks and a cleaner choice for prosody-sensitive downstream tasks, with comparable inference cost to HuBERT-base. The difference is unlikely to be huge in absolute accuracy terms (both encoders expose prosodic features richly), but the defensible story is: pick WavLM-base+ for the POC's primary pipeline *or* include WavLM as a quick sanity-check ablation if HuBERT is kept as default. On T4 compute, either encoder is fine; HuBERT-large / WavLM-large inflate memory beyond safe POC budget and should not be used without a compute case.

### Core citations

**[7.1] Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R. & Mohamed, A. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.* IEEE/ACM Transactions on Audio, Speech, and Language Processing.**
*Summary.* The HuBERT paper. Masked prediction with offline clustered acoustic units; strong transfer on ASR and downstream speech tasks. Pretrained on LibriSpeech (read speech).
*What it contributes to our design.* Foundational citation for our current audio feature. Must cite. The LibriSpeech-read-speech pretraining is the known weakness for conversational prosody.

**[7.2] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian, Y., Qian, Y., Zeng, M., Yu, X. & Wei, F. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.* IEEE Journal of Selected Topics in Signal Processing, 16(6), 1505–1518 (arXiv:2110.13900).**
*Summary.* Extends HuBERT's masked prediction with a denoising objective (speech mixed with other speech / environmental noise during pretraining) and gated relative-position encoding. Pretrained on a mix including diverse conversational audio beyond LibriSpeech. WavLM Large is SOTA on SUPERB; WavLM Base+ outperforms HuBERT-large and wav2vec 2.0 on all downstream tasks at a fraction of the cost.
*What it contributes to our design.* The single strongest case for swapping HuBERT-base → WavLM-base+ in the POC pipeline. SUPERB paralinguistic and prosody tasks are precisely the capability cluster we care about for turn-taking. Drop-in replacement — same input rate, similar memory footprint.

**[7.3] Yang, S.-W. et al. (2021). *SUPERB: Speech processing Universal PERformance Benchmark.* Interspeech 2021 (arXiv:2105.01051).**
*Summary.* The benchmark against which WavLM's prosody advantage is established. Includes tasks across content, speaker, semantics, and paralinguistics.
*What it contributes to our design.* The citation we use to justify "WavLM-base+ is the defensible encoder for paralinguistic/prosodic tasks as of 2024–26." Cite SUPERB together with WavLM 2022 when making the encoder-choice argument in methods.

**[7.4] Hisada, S., Inoue, K., Lala, D., Nakamura, S. & Kawahara, T. (2024). *Multimodal Voice Activity Projection for Turn-taking and Effects on Speaker Adaptation.* IEICE 2024 (already cited as [1.4]).**
*Summary (for this item).* Uses CPC features (VAP's native) — not HuBERT, not WavLM. This is a reminder that VAP's audio branch is its own pretraining recipe, not a SUPERB-style encoder.
*What it contributes to our design.* Tells us our HuBERT/WavLM + GRU design is a different architectural family from VAP, which is fine — it doesn't need to converge with VAP. But it reinforces that our baseline story depends on an external VAP checkpoint (Item 6) for comparison; our own encoder choice is about maximizing our model's own horizon-prediction performance, not matching VAP's architecture.

**[7.5] Mohamed, A. et al. (2022). *Self-Supervised Speech Representation Learning: A Review.* IEEE Journal of Selected Topics in Signal Processing.**
*Summary.* Review of SSL speech encoders including wav2vec 2.0, HuBERT, WavLM, data2vec. Characterizes per-encoder strengths. Identifies WavLM as the leading all-round encoder for speaker + paralinguistic + content tasks.
*What it contributes to our design.* One-citation backing for "encoder choice is a studied axis in SSL speech representation, and WavLM is the paralinguistic leader." Cite briefly in the encoder-choice paragraph.

**[7.6] Russell, S. O'C. & Harte, N. (2025). *Visual Cues Enhance Predictive Turn-Taking…* ACL Findings 2025 (already cited as [1.1]).**
*Summary (for this item).* MM-VAP uses CPC (VAP's encoder), not HuBERT/WavLM. Does not evaluate encoder choice.
*What it contributes to our design.* Means the encoder-choice question is *still open* in the multimodal-turn-taking literature. Our POC running WavLM-base+ and reporting against a HuBERT-base baseline would be a small but real methodological contribution — not on the central horizon-gain axis, but on the auxiliary "does encoder choice matter at all for multimodal TT" axis.

### Flags

- **HuBERT-base is trained on LibriSpeech read speech. This is a known mismatch for conversational prosody.** The report must own this. Either switch to WavLM-base+ (cleaner story, minimal cost) or keep HuBERT and cite this as a principled simplicity choice. Recommendation: switch.
- **WavLM-base+ vs. HuBERT-base as an ablation is the cheapest quality-improvement lever available.** Swap in feature extraction; the GRU head is unchanged. Even if multimodal gain is small, reporting both encoders demonstrates robustness and is a partial defense against "you picked a weak encoder" critiques.
- **Do NOT use HuBERT-large or WavLM-large for the POC.** At T4 + ~280 compute units, large variants will eat memory and compute unacceptably without a corresponding gain justified by literature. MM-VAP uses CPC (smaller than HuBERT-base) and still reports strong results.
- **Encoder choice is unlikely to change the horizon-gain-from-face result direction, only its magnitude.** The permuted-dyad control is the load-bearing scientific test; encoder choice affects absolute F1, not the relative gain. Do not let encoder-choice anxiety substitute for the real experimental design.
- **If switching encoder, document the switch reason in methods.** Phrasing template: "We report results with WavLM-base+ as the audio encoder; WavLM is the SUPERB paralinguistic-task leader and incorporates conversational noise in its pretraining, addressing a known limitation of HuBERT-base whose pretraining is read-speech-dominant (Chen et al. 2022)."
- **Keep encoder-choice as a one-line ablation, not a deep study.** This is a CSCI-535 POC. The encoder-choice axis is *not* our scientific question. One well-cited paragraph + one ablation row in the results table is sufficient.

---

## FINAL Overall Integrity Assessment of the Research Question

**Research question:** *Does adding OpenFace-derived facial behavior to a HuBERT audio encoder measurably extend the prediction horizon of end-of-turn detection in dyadic conversation, relative to an audio-only baseline?*

### Novelty — **weakly novel, defensible positioning available**

The macro-claim ("face helps audio-only turn-taking prediction") is *no longer novel* — Hisada et al. 2024 (IEICE), Russell & Harte ACL Findings 2025 (MM-VAP), Russell & Harte Interspeech 2025 (noise-robustness), Saga & Pelachaud 2025 (learned face encoders), and Heo et al. 2025 (triadic gaze) have all published in this direction. *Our* research question differentiates on three non-trivial axes that 2024–26 work does not simultaneously address: (1) a face-to-face co-located corpus (Seamless) rather than videoconferencing (Candor) or triadic setups; (2) a permuted-dyad control that rules out the monadic-only account of the visual gain; (3) an explicit τ-swept accuracy/F1 curve, framed as the primary reporting metric, rather than silence-duration-stratified point accuracy or VAP-native projection-window probability mass. We can honestly claim "targeted replication + controlled extension" on a face-to-face corpus with a dyadic-coordination control.

### Methodological defensibility — **yes, with mandatory additions**

After seven items the baseline set expands from {majority, HuBERT→GRU audio-only} to **{majority, WavLM-base+→GRU audio-only, VAP-checkpoint zero-shot, multimodal WavLM+OpenFace→GRU, permuted-dyad eval set}**. The metric is a per-class F1-vs-τ curve reported at τ ∈ {100, 200, 400, 500, 800, 1600} ms (500 ms added for MM-VAP commensurability). BC-class F1 is reported separately, with the Seamless A.1.4 / Cangemi 2025 short-utterance-alignment limitation cited explicitly. INTERRUPT labels are guarded with a minimum-duration filter on overlapping VAD segments. Feature-priority order for per-feature-group ablations: audio → +AUs → +head pose → +gaze (following MM-VAP and Nota-Holler feature-importance evidence).

### Feasibility at CSCI-535 scope — **yes**

POC is 24 dyads at 16/4/4 split; features are pre-computed by Seamless (except OpenFace AUs for our subset). T4 / ~280 compute units is adequate for WavLM-base+ inference + GRU training on this scale. Zero-shot VAP inference is cheap. Hand-annotation sanity check for BC labels is ~45 min of manual work. No item in the review identified a literature-driven requirement that breaks the compute budget.

### Verdict

**Proceed. Research question is weakly novel, methodologically defensible, and feasible at CSCI-535 scope — conditional on implementing the five load-bearing additions flagged across items 1–6 and on surfacing per-class metrics rather than macro-averages.** If only one addition is implemented, it should be the **permuted-dyad control evaluation set** (Risk 2 in the memory; load-bearing for the dyadic-coordination claim). If two, add the **zero-shot VAP checkpoint on POC test** (Item 6).

---

## Recommended Improvements — Question Framing and Methods

Every recommendation is tagged with cost/complexity: **[low]** = ≤ 1 day of work or already in plan; **[mid]** = 1–3 days; **[high]** = >3 days or a meaningful scope change.

### Framing

**F1. Reframe the stated contribution from "novel multimodal turn-taking model" to "targeted replication + controlled extension on a face-to-face corpus, with a dyadic-coordination control that prior art lacks." [low]**  The novelty sweep (Item 1) shows MM-VAP and successors have established the macro-result. Making our contribution a replication-plus-control rather than a novelty claim is more honest and more defensible in a CSCI-535 report.

**F2. Move "horizon gain from face" to the title as the primary claim; keep "multimodal turn-taking" as descriptive. [low]**  Horizon-gain is our distinct metric framing. Leading with it clarifies that we are not claiming a new model, we are measuring a specific quantity (τ at which multimodal matches a target F1 floor that audio-only cannot achieve).

**F3. Explicitly cite MM-VAP (Russell & Harte 2025) as the closest prior art in the introduction, not deep in related work. [low]**  This preempts the "did you check 2025 ACL Findings?" critique. Explicit positioning against MM-VAP is the single highest-leverage framing move.

**F4. State the face-to-face vs. videoconferencing distinction as a scientific claim, not a corpus choice. [low]**  The Kendrick-Holler-Levinson 2023 finding (gaze inhibits transitions in face-to-face) and MM-VAP's finding (gaze in isolation does not help in videoconferencing) together imply a testable hypothesis: visual cues in general and gaze specifically should help *more* in face-to-face than in videoconferencing. Our result is interpretable against that hypothesis whether it replicates MM-VAP or deviates. Make this a paragraph in the introduction.

**F5. Be explicit that INTERRUPT is the most uncertain class and that per-class rather than macro-F1 is the scientific output. [low]**  Items 5 and 6 converge on this. Stating it up front removes the temptation to roll up metrics and aligns reader expectations with the per-class τ-curve we will actually publish.

**F6. Acknowledge the OpenFace/no-gesture limitation in the introduction, not only in the limitations section. [low]**  Kendrick 2023 shows gestures-in-progress inhibit transitions; we are not using SMPL-H in the POC. Surface this once, early.

### Methods

**M1. Swap primary audio encoder from HuBERT-base to WavLM-base+. Keep HuBERT-base as the encoder ablation row. [low]**  Item 7. Drop-in, same interface, SUPERB paralinguistic leader, conversational-noise pretraining addresses HuBERT's LibriSpeech-read-speech weakness.

**M2. Add zero-shot VAP checkpoint inference on the POC test set at τ = 500 ms as a baseline row. [low]**  Item 6. Inoue et al. 2024's multilingual VAP checkpoint or Ekstedt-Skantze's 2022 checkpoint. Inference-only; T4-feasible.

**M3. Commit to the permuted-dyad evaluation set from day one of the eval pipeline. [mid]**  Risk 2 / Item 1. Load-bearing for the dyadic-coordination claim. Build the permuted manifest alongside the real eval manifest in `build_poc_manifest.py`. Expected behavior: multimodal model drops to audio-only-baseline level on permuted dyads; if it does not, the "dyadic coordination" framing must be replaced with a "monadic visual end-of-turn cue" framing, which is a legitimate but different paper.

**M4. Report per-class F1 as a curve over τ, with τ = 500 ms as a commensurability point with MM-VAP. [low]**  Items 2 and 5. The τ grid becomes {100, 200, 400, 500, 800, 1600} ms. Do NOT report only accuracy; per-class F1 exposes INTERRUPT and BACKCHANNEL class-noise effects that macro metrics hide.

**M5. Guard the INTERRUPT label with a minimum-duration filter on the overlapping VAD segment (≥ 500 ms) and a NOT-a-backchannel lexical predicate. [low]**  Item 5. Prevents VAD-on-backchannel artefacts from inflating INTERRUPT base rate.

**M6. Hand-annotate a 30–60 s backchannel-dense slice from one POC dyad in Praat. Report empirical WhisperX timestamp error on backchannels. [low, high-value]**  Item 4 / Risk 1. Cite Amoyal et al. LREC-COLING 2024 as the methodological template. One results paragraph that preempts reviewer skepticism about BC-class label quality.

**M7. Run per-feature-group ablations in feature-priority order: audio → +AUs → +head pose → +gaze. [low]**  Item 3 / MM-VAP evidence. Expected outcome: AUs add the most horizon gain, head pose adds a smaller increment, gaze alone is marginal. Running in this order makes our results directly commensurable with MM-VAP's ablation ranking.

**M8. Publish the POC class distribution (HOLD/YIELD/BC/INTERRUPT counts in train/val/test) in the results section, not only in an appendix. [low]**  Items 5, 6. Transparent base-rate reporting prevents the "classifier is picking up base rate" critique.

**M9. Acknowledge in limitations that Saga & Pelachaud 2025 (learned face encoders) and Onishi 2025 (prompt-guided VAP) are beyond our pipeline's frontier. [low]**  Items 1, 6. Owning the frontier gap is more defensible than pretending our 2022–24 recipe is current.

**M10. (Optional extension, if compute remains.) Re-transcribe one POC dyad with CrisperWhisper (Wagner et al. Interspeech 2024) and report label-agreement with Seamless's WhisperX transcripts on BACKCHANNEL span timing. [mid]**  Item 4. Would turn the A.1.4 limitation from a caveat into a quantified statement. Do this only if M6 reveals a meaningful BC-class error and BC F1 is the binding class.

