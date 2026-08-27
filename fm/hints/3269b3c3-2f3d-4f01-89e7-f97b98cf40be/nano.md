Educational infographic, clean flat design, soft pastel palette with one accent color (orange) for the key signal, white background, thin sans-serif labels. Aspect ratio 16:9, landscape.

Title at top center: "absgrad vs grad: why cancellation hides split candidates" (subtitle in smaller text: "3D Gaussian Splatting densification", with Korean gloss "밀집화 판정").

Layout: two large panels side by side in the upper two-thirds, one wide comparison strip across the bottom third. Reading flow: left panel -> right panel -> bottom strip. A small shared scene sits at the top of both panels.

Shared scene (drawn identically at the top of both upper panels): a 1D pixel row shown as a horizontal strip of small squares. Above it, a ground-truth curve with two sharp narrow peaks, labeled "GT: two fine details" (Korean gloss "실제 디테일 2개"). Overlapping it, one wide translucent blue bell curve centered exactly between the peaks, labeled "one big Gaussian" (Korean gloss "큰 가우시안 하나").

Left panel, header "Regular grad (signed sum)": under the scene, per-pixel gradient arrows drawn on the pixel row. Pixels under the left peak carry red arrows pointing left; pixels under the right peak carry red arrows pointing right, equal in length, mirror-symmetric. Below, a summation symbol combining all arrows into a single tiny dot with a faint zero-length arrow, labeled "sum = 0" and "signals cancel" (Korean gloss "상쇄"). Formula label: "grad = Σ_p g_p".

Right panel, header "absgrad (absolute sum)": same scene and same arrows, but each arrow is redrawn as an orange bar of its length pointing in the same direction (all upward), stacked end to end into one tall orange bar. Labeled "sum |g_p| = large" and "signal preserved" (Korean gloss "신호 보존"). Formula label: "absgrad = Σ_p |g_p| ≥ |grad|".

Bottom strip, header "Densification decision" (Korean gloss "split 판정"): a horizontal threshold line labeled "threshold" spanning the strip. On the left half, a short blue bar far below the line, labeled "grad: below threshold", with a result icon of the same single big Gaussian unchanged and the label "not split, details lost". On the right half, a tall orange bar above the line, labeled "absgrad: above threshold", with a result icon of two smaller Gaussians sitting on the two peaks and the label "split, details recovered". Small caption at bottom right: "gsplat DefaultStrategy(absgrad=True), nerfstudio use_absgrad".

Style notes: consistent icon language, rounded corners, generous whitespace, arrows and bars as the dominant visual metaphor, no photographic textures, no long sentences, all labels 3-5 words.
