# Deep-learning investigation log

Plain-language record of trying to make deep learning work on the structural-break
challenge, after the gradient-boosted tree (LightGBM) approach. Read top to bottom.

## The goal

Each series has a "before" part (reference) and an "online" part that streams in one
point at a time. At every online step we predict: has the distribution changed
(a structural break) yet? Score = TS-AUC: at each step, how well we separate the
series that broke from the ones that didn't, averaged over steps.

The tree model on hand-built features scores about **0.579** on our 2000-series
holdout. That number is the bar everything gets measured against.

## First surprise: the "0.606" baseline was a ghost

The README and notes claimed the tree model scored 0.606. When we actually re-ran it,
it scored 0.579. Same code, same data. The 0.606 does not reproduce — it came from an
older setup that was never saved. **Lesson: always re-run the baseline yourself before
trusting it. A number in a doc is not a measurement.**

## Probe 1: a plain neural net on the SAME features (MLP)

Quickest question first: can a neural net even compete with the tree, given the exact
same 1072 features?

- Result: 0.567, a bit below the tree's 0.579.
- Its predictions were 87% correlated with the tree's.

**Lesson: feeding a net the same features just gives you a worse copy of the tree.**
No new information, so no reason to combine them. If deep learning is going to add
anything, it has to look at something the features don't — the raw sequence itself.

## Probe 2: the encoder on raw sequences

Plan: let the net read the raw numbers. Encode the reference once into a summary,
run the online stream through a network, compare the two at each step. This is the
only way to see "sequence shape" that the aggregate features throw away.

Getting this to run took four separate fixes. Each one is a lesson.

### Bug 1 — the labels were misaligned
The label file only covers the online part of each series, but we sliced it using
positions from the full series (reference + online). Everything was shifted.
**Lesson: check that your labels line up with your data before anything else. A silent
off-by-alignment error looks like "the model just doesn't work".**

### Bug 2 — the RNN was unusably slow
We first used a GRU (a standard sequence network). On this Mac's GPU it hung — even a
tiny test took minutes. The culprit is a specific PyTorch operation for
variable-length sequences that isn't well supported on Apple's GPU.
**Fix: switched to a TCN — a convolution-based sequence network that runs in parallel
and has no such problem. 10x faster. Lesson: match the tool to the hardware; the
"standard" choice isn't always the runnable one.**

### Bug 3 — training was still too slow
The network produced a prediction at every one of ~1000 online steps, and the final
layer ran on all of them. Most of that work was wasted.
**Fix: only train on ~25 spread-out steps per series (same steps the tree uses).
Cut the heavy work by ~40x. Lesson: don't compute predictions you won't use.**

### Bug 4 — the inputs were secretly all zeros
This was the important one. A few reference series are nearly flat, which made our
normalization blow up — some input values reached 250 million. When we then rescaled
everything to a common range, those giant outliers dominated, and the *useful*
variation got squashed to essentially zero. The network was being fed noise and
learned nothing (score ~0.50, random).

We caught it by pushing a simple hand-made "did the variance change?" score through
the exact same pipeline. It scored 0.54 — so the pipeline was fine, the *inputs* were
broken.
**Fix: rescale using robust statistics (median and typical spread) instead of
average and standard deviation, then clip extremes. Lesson: when a model scores
random, test the plumbing with a dumb baseline before blaming the model. And always
look at the actual range of your inputs — outliers quietly destroy everything.**

### The real problem: the net optimized the wrong thing
After all fixes it reached 0.53 — still weak, and still 89% correlated with the tree.
That high correlation was the clue. The score (TS-AUC) only cares about separating
series *at the same step*. But the standard training loss (BCE) can score well just by
learning the obvious trend "the later we are in the stream, the more likely a break
already happened". Both the net and the tree learn that trend, which is why they
looked correlated — but it's a trend the score ignores.

**Fix: a ranking loss.** Instead of "predict the label", train it to "rank the
broken series above the unbroken ones, at each step". That is exactly what the score
measures.

Result: score 0.537, and correlation with the tree dropped from 0.89 to **0.55**.
**Lesson: your training objective should match your evaluation metric. When they
disagree, the model optimizes the metric you trained on, not the one you care about.**

## Where it stands

- Encoder on raw sequences: **0.537**, correlation with the tree **0.55**.
- Still below the tree's 0.579, so on its own it loses.
- But the low correlation (0.55) means it genuinely sees something different. If we
  can push it to ~0.56, blending it with the tree should beat 0.579.
- The thing holding it back now is **overfitting**: it peaks after ~6 passes over the
  data, then memorizes the training series and gets worse on the holdout.

## Overall lessons

1. **Re-run baselines. Don't trust documented numbers.**
2. **A dumb baseline is the best debugging tool.** It tells you if the plumbing works.
3. **Look at your input ranges.** Outliers + rescaling silently kill signal.
4. **Train on the metric you're graded on.** Loss/metric mismatch wastes the model.
5. **Same features in = same answer out.** New value needs a genuinely new input view.
6. **Match the method to the hardware** (TCN over RNN here).
7. Deep learning here is not obviously winning, but the encoder is the one thread that
   sees something new — worth pushing on regularization before calling it.

## Next step

Regularize the encoder (smaller model, more dropout, shorter training, seed-bagging)
to raise its peak and turn the 0.55-correlation into a blend that beats 0.579.
