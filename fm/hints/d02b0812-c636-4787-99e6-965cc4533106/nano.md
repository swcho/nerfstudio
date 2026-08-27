Educational infographic, clean flat design, 16:9 landscape, white background, soft blue/teal/orange/red accent palette, thin rounded connector arrows, sans-serif labels.

Title at top center: "gsplat DefaultStrategy.step_post_backward" with a small Korean subtitle "스텝 종료 콜백 흐름".

Layout: a vertical flowchart occupying the upper 65% of the canvas, and a horizontal step-timeline band across the bottom 30%. Eye flow: top-down through the flowchart, then left-to-right along the timeline.

Flowchart (top to bottom, centered):
1. Rounded entry pill: "loss.backward() done".
2. Red diamond gate: "step >= 15000?" with a red exit arrow to the right labeled "return (no-op)" ending in a stop icon.
3. Blue rectangle: "_update_state" with two small icons inside: a bar "grad2d += |absgrad|" and a counter "count += 1", and tiny caption "visible Gaussians only".
4. Orange diamond: "refine now?" with three tiny condition chips stacked beside it: "step > 500", "step % 100 == 0", "step % 3000 >= 261".
5. From the YES branch, a grouped orange box containing three stacked steps with icons: "_grow_gs: clone / split" (icon: one dot becoming two), "_prune_gs: opacity < 0.1" (icon: faded dot with X), "reset stats" (icon: bar chart zeroed). Add a small note "Adam state resized too".
6. NO branch bypasses the orange box and rejoins.
7. Teal diamond: "step % 3000 == 0?" with YES arrow to a teal rectangle "reset_opa: opacity <= 0.2" (icon: opacity slider clamped), NO arrow to an end pill "next step".

Bottom timeline band: a horizontal axis from 0 to 15000 with tick marks at 0, 500, 600, 3000, 3300, 6000, 9000, 12000, 15000. Shade region 0-500 light gray labeled "warmup". Place small orange dots every 100 above the axis labeled once "refine every 100". Place teal downward arrows at 0, 3000, 6000, 9000, 12000 labeled "alpha reset". Draw short gray "pause 261" gaps right after each teal arrow. Shade region 15000+ red-hatched labeled "frozen".

Legend at bottom-right: red = "stop gate", blue = "accumulate", orange = "grow / prune", teal = "opacity reset". Keep all labels under 5 words, high contrast, generous spacing, no long sentences.
