Educational infographic, title at top: "Why 9 SH Coefficients Are Enough" (subtitle in smaller text: "Clamped cosine is low-frequency / 클램프 코사인은 저주파"). Clean flat design, white background, muted blue and orange accent palette, thin sans-serif labels, 16:9 landscape aspect ratio. Three panels arranged left to right with arrows guiding the eye.

Panel 1 (left, label "Irradiance = Radiance x Cosine"): a hemisphere dome with a surface normal arrow at its center; incoming light rays from many directions; a small formula chip "E_lm = A_l * L_lm" and two tags "L_lm: lighting (varies)" and "A_l: cosine (constant)".

Panel 2 (center, largest, label "A_l decays fast"): a bar chart with x-axis l = 0 to 6 and bars of height 3.14, 2.09, 0.79, 0, -0.13, 0, 0.05. The first three bars (l = 0,1,2) drawn in bold orange and grouped by a bracket labeled "l <= 2: 99% energy"; the remaining bars drawn in faint grey with a callout "nearly zero". A dashed zero line. Short note "odd l >= 3: exactly 0".

Panel 3 (right, label "Result: 27 floats"): two small identical rendered spheres side by side, one captioned "32x32 cubemap, 24 KB", the other "9 SH coeffs x RGB, 108 B", with an equals sign between them and a badge "~200x smaller". Below, a row of nine small colored squares labeled "l=0 (1)", "l=1 (3)", "l=2 (5)".

Bottom strip: a single-sentence takeaway "Low-frequency cosine filter kills high l terms" in a rounded pill.
