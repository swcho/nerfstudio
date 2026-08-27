Educational infographic titled "Splatfacto train_iteration(): 9 Steps A-I", clean flat design, 16:9 landscape, white background, soft pastel accents, thin rounded outlines.

Layout: one horizontal pipeline of nine rounded boxes connected left-to-right by arrows, labeled with big letters A through I. The first (A) and last (I) boxes are drawn in a distinct orange color and bracketed above with a label "Trainer.train() callbacks"; the seven middle boxes B-H are blue and bracketed below with a label "train_iteration() — standard PyTorch loop".

Box labels (short, one line each, with a tiny icon):
A "BEFORE callback step_cb" (plug icon: optimizers injected into model)
B "zero_grad" (eraser icon)
C "next_train data" (camera icon)
D "forward get_outputs" (rasterized splats icon)
E "metrics / loss 0.8 L1 + 0.2 (1-SSIM)" (scale icon)
F "backward" (reverse arrow icon)
G "optimizer step 6x Adam" (gear icon)
H "scheduler means lr decay" (small descending curve icon)
I "AFTER callback step_post_backward" (split/clone icon)

Below the pipeline, a two-panel comparison card highlighting G versus I:
Left panel titled "G: changes parameter VALUES" — a fixed grid of the same number of gaussian dots, arrows nudging each dot slightly, caption "same N, every step".
Right panel titled "I: changes parameter COUNT (N)" — a small gaussian splitting into two and another being deleted, plus a bar labeled "Adam exp_avg / exp_avg_sq resized", caption "every 100 steps, densify / prune".
A dashed arrow from I back to A with a label "optimizers shared" to show why the callback injects optimizers.

Style: minimal, high-contrast typography, consistent icon set, no long sentences, all labels in short English (optionally add small Korean subtitles for "callback 콜백", "densify 분할/복제", "prune 컬링").
