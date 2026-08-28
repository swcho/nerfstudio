Educational infographic, title at top: "Irradiance Map: 24KB vs 108 Bytes" (subtitle in Korean: 구면 조화 함수로 메모리 227배 절감). Clean flat design, soft blue and teal palette with orange accents, white background, 16:9 landscape.

Layout: two large side-by-side panels connected by a big arrow in the middle labeled "227x smaller", and a thin bottom strip.

Left panel, label "Cubemap Irradiance Map": an unfolded cube-map cross of six small 32x32 grid textures, each face tinted sky blue on top and sea teal at bottom. Formula card beneath: "32 x 32 x 6 faces x 4 B = 24 KB". A shaded sphere render preview with sky-lit top and teal-lit bottom.

Right panel, label "Spherical Harmonics (구면 조화 함수)": a small pyramid of 9 SH basis lobes arranged in rows l=0 (1 lobe), l=1 (3 lobes), l=2 (5 lobes), each lobe rendered as a blue/orange positive-negative blob. Beside it, a compact 3x9 grid of tiny colored cells labeled "R, G, B rows x 9 coefficients". Formula card beneath: "3 RGB x 9 coeffs x 4 B = 108 B". An identical shaded sphere render preview, with a green check badge "Nearly identical result".

Bottom strip, label "Why only 9 coefficients?": a small line chart, x-axis "SH band l" from 0 to 6, y-axis "cos weight A_l", points high at l=0,1,2 then dropping to zero at l=3 and flat afterward; highlighted region over l=0..2 labeled "Keep l <= 2", greyed region labeled "Discard".

Flat vector icons, minimal text, legible sans-serif labels, no photorealism.
