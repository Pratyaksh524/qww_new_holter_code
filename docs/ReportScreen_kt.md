# `ReportScreen.kt` — Android ECG Report Screen & PDF Renderer

**File:** [ReportScreen.kt](../ReportScreen.kt) — 1181 lines
**Package:** `com.deckmount.ecgapp.presentation.ui.screen.report`
**Platform:** Android, Jetpack Compose + Material 3
**Original author:** Amarjeet Kumar
**Last revised:** 2026-08-18

---

## 1. Purpose

This file is the Android counterpart to the desktop report generators in
[src/ecg/](../src/ecg/). It renders a finished 12-lead ECG report — grid, header, waveforms,
calibration pulses, conclusions and footer — and does so through **one drawing implementation that
serves both the on-screen preview and the exported PDF**.

That single-implementation property is the central design decision of the file. Every coordinate is
expressed in **millimetres**, and a single scale factor (`pxPerMm`) converts millimetres to whatever
the current output device wants:

| Output | `pxPerMm` | Origin |
|---|---|---|
| Screen preview | `(availableWidthPx / pageWidthMm) × 2.5` | Composable measures itself, then oversamples 2.5× |
| PDF | `595 / 210 ≈ 2.833` (portrait) or `842 / 297 ≈ 2.835` (landscape) | A4 in PDF points, 1 pt = 1/72 in |

Because both paths run `ECGReportRenderer.draw(canvas)`, what the clinician sees on the phone is what
prints on paper. There is no second layout engine to drift out of sync.

---

## 2. File map

| Lines | Element | Kind |
|---|---|---|
| 1–56 | Package and imports | — |
| 62–95 | Page, margin, signal and grid constants | `private const val` |
| 101–330 | `ECGReportScreen` | `@Composable` — public entry point |
| 337–441 | `ECGReportCanvas` | `@Composable` — private preview surface |
| 449–1130 | `ECGReportRenderer` | `internal class` — all drawing |
| 1141–1182 | `generateECGPdf` | top-level function |

Within `ECGReportRenderer`:

| Lines | Member |
|---|---|
| 454–478 | Derived geometry, `isForPdf`, `p()` / `pt()` unit helpers |
| 481–549 | Paint factory and pre-built paints |
| 556–562 | `drawTxt` — TCPDF-compatible text placement |
| 580–595 | `draw` — the top-level draw order |
| 602–618 | `drawGrid` |
| 624–721 | `drawHeader` |
| 729–756 | `draw1x12` — portrait |
| 766–820 | `draw2x6` — landscape |
| 829–885 | `draw3x4` — landscape |
| 891–994 | `drawFooter` / portrait / landscape |
| 1016–1049 | `drawCalibration`, `drawCalibrationPad` |
| 1096–1129 | `drawWaveform` |

---

## 3. External contract

### 3.1 `ECGReportRenderData`

Defined outside this file (in the sibling package
`com.deckmount.ecgapp.presentation.ui.screen.livemonitorecg`). Every field this file reads:

| Field | Type | Used for |
|---|---|---|
| `layout` | `String` | Layout selector: `"1x12"`, `"3x4"`, anything else → `"2x6"` |
| `patientName`, `patientAge`, `patientGender` | `String` | Header column 1; blank renders as `-` |
| `reportDate`, `reportTime` | `String` | Header column 1, pre-formatted — this file does no date formatting |
| `acFilter` | `String`/numeric | Printed into the spec line as `AC:<value>Hz` |
| `hr`, `rr`, `pr`, `qrs`, `qt` | numeric | Header column 2 |
| `qtc`, `qtcf` | numeric | Header column 3 |
| `rv5`, `sv1` | `Float` | Header column 3; printed to 3 dp, summed for the Sokolow-Lyon index |
| `orgName`, `orgAddress`, `orgPhoneNo` | `String` | Header right block; phone is prefixed `+91 ` when non-blank |
| `machineSerial` | `String` | Footer — **last 4 characters only** |
| `conclusions` | `List<String>` | Footer conclusion box |
| `leadData3500` | `Map<String, List<Float>>` | 1×12 portrait strips |
| `leadData1750` | `Map<String, List<Float>>` | 2×6 landscape strips |
| `leadData1250` | `Map<String, List<Float>>` | 3×4 landscape strips |
| `leadData5000` | `Map<String, List<Float>>` | Rhythm strip (lead **II**) in both landscape layouts |

`pqrstAxis` exists on the model but is referenced only inside a commented-out line
([ReportScreen.kt:687](../ReportScreen.kt#L687)) — the P/QRS/T axis is **not** printed today.

A missing map key degrades gracefully: `samples[lead] ?: emptyList()`, and `drawWaveform` returns
immediately when `samples.size < 2`. A missing lead therefore prints as an empty lane with its label and
calibration pulse intact — no crash, no exception.

### 3.2 `ALL_LEADS`

An external `List<String>` imported from the live-monitor package, iterated by `draw1x12`. The 2×6 and
3×4 layouts do **not** use it; they carry their own hardcoded lead orderings (§7.2, §7.3).

### 3.3 Sample-count contract

| Map | Samples | Duration @ 500 Hz | Width at 25 mm/s | Lane width available | Headroom |
|---|---:|---:|---:|---:|---|
| `leadData3500` | 3500 | 7.0 s | 175.0 mm | 185 mm | 10 mm |
| `leadData1750` | 1750 | 3.5 s | 87.5 mm | 123 mm | 35.5 mm |
| `leadData1250` | 1250 | 2.5 s | 62.5 mm | 80 mm | 17.5 mm |
| `leadData5000` | 5000 | 10.0 s | 250.0 mm | 254 mm | 4 mm |

Every stream fits its lane. **Anything longer is silently truncated** — `drawWaveform` breaks out of its
loop the moment `xMm > maxXMm` ([ReportScreen.kt:1120](../ReportScreen.kt#L1120)) with no warning and no
log. If a caller ever supplies more samples than the table above, the tail simply disappears from the
report. Treat these counts as a hard contract.

---

## 4. Constants

```kotlin
private const val A4_P_W = 210f     // portrait page width  (mm)
private const val A4_P_H = 297f     // portrait page height (mm)
private const val A4_L_W = 297f     // landscape page width  (mm)
private const val A4_L_H = 210f     // landscape page height (mm)

private const val MARGIN_PDF    = 0f    // ⚠ declared, never referenced
private const val MARGIN_TOP    = 5f
private const val MARGIN_BOTTOM = 5f
private const val MARGIN_LEFT   = 5f
private const val MARGIN_RIGHT  = 5f

private const val ADC_PER_MV = 1.8f     // ADC units per millivolt
private const val ECG_FS     = 500f     // sampling rate (Hz)

private const val FIXED_WAVE_SPEED = 25f    // mm/s  — not user-adjustable
private const val FIXED_WAVE_GAIN  = 10f    // mm/mV — not user-adjustable

private const val GRID_W_PORTRAIT  = 210f
private const val GRID_H_PORTRAIT  = 297f
private const val GRID_W_LANDSCAPE = 297f
private const val GRID_H_LANDSCAPE = 210f
```

> **Stale comments.** The margin constants are `5f` but their trailing comments still say *"10mm"*, and
> the layout functions carry header comments citing `topOffset=38` / `startY=35` from when the margins
> were 10 mm. The **code** is authoritative; the derived values in §7 are computed from `5f`. The older
> copy of this file at [test_cloud_connection/ReportScreen.kt](../test_cloud_connection/ReportScreen.kt)
> still uses 10 mm margins and an extra `ADC_PER_MM = 2.75f` constant — see §12.

**Paper-speed identities** (worth internalising, they explain every magic number below):

- `mmPerSample = FIXED_WAVE_SPEED / ECG_FS = 25 / 500 = 0.05 mm/sample`
- 1 mm ≡ 20 samples ≡ 40 ms
- One large grid box (5 mm) ≡ 200 ms horizontally, ≡ 0.5 mV vertically at 10 mm/mV
- `1 mV ≡ 1.8 ADC units ≡ 10 mm` on paper

---

## 5. Coordinate system

Everything is authored in millimetres and converted at draw time:

```kotlin
private fun p(mm: Float)      = mm * pxPerMm            // mm → device units
private fun pt(ptSize: Float) = ptSize * 0.352778f * pxPerMm   // font pt → device units
```

`0.352778` is mm-per-point (1/72 inch). `pt()` exists so font sizes can be specified in the same
typographic points the legacy TCPDF templates used, and land at the same physical size on both outputs.

### 5.1 `drawTxt` — TCPDF text semantics

Android's `Canvas.drawText` positions text by its **baseline**. TCPDF's `Text($x, $y, $s)` positions by
the **top-left** of the bounding box. Rather than translate every legacy coordinate, the file adapts the
API:

```kotlin
val fm = paint.fontMetrics
canvas.drawText(text, p(xMm), p(yMm) - fm.ascent, paint)   // fm.ascent is negative
```

**Consequence:** every `yMm` passed to `drawTxt` is the **top** of the text, not the baseline. This is
the single most common source of confusion when editing layout coordinates in this file.

### 5.2 `isForPdf`

```kotlin
private val isForPdf = pxPerMm < 3.5f
```

An inferred flag, not a parameter. PDF rendering runs at ≈ 2.83 units/mm; screen rendering at
`availableWidthPx / pageWidthMm × 2.5`, which on any normal device is 9–13. It selects thinner strokes
for print (§6).

> ⚠️ **Fragile heuristic.** The flag flips to `true` whenever the composable is narrower than
> ≈ 294 px (portrait) or ≈ 415 px (landscape) — reachable in split-screen, a foldable cover display, or a
> small preview pane. The screen preview would then draw with print-weight hairlines. Passing an explicit
> `isForPdf: Boolean` into the constructor would remove the guesswork entirely.

---

## 6. Paints

Built once per `ECGReportRenderer` instance, all anti-aliased.

| Paint | Colour | Stroke (mm) | Purpose |
|---|---|---|---|
| `gridMinor` | RGB(245, 220, 220) | 0.1 | 1 mm grid |
| `gridMajor` | RGB(230, 150, 150) | 0.25 | 5 mm grid |
| `waveP` | Black | PDF 0.20 / screen 0.25 (min 1.5 px) | ECG trace — round join and cap |
| `calibP` | Black | 0.4 (min 1.5 px on screen) | 1 mV calibration pulse — mitre join, square cap |
| `boxP` | Black | 0.3 (min 1.0 px on screen) | Conclusion box outline |
| `dashDivP` | RGB(80, 80, 80) | 0.4 (min 1.2 px on screen) | Dashed 2 mm/2 mm column divider |

The `coerceAtLeast` floors exist because a 0.25 mm line at low screen density can round to zero device
pixels and vanish; the floor guarantees the waveform is always visible on screen. PDF has no floor
because vector output has no minimum stroke.

Text paints are produced by `mkText(ptSize, bold, italic)` and cached as `tp6`…`tp20_5B`. The naming is
positional: `tp10_5B` = 10.5 pt bold.

**Dead paints:** `tp6` and `tp20_5B` are declared but never drawn with; `tp8B` is used only inside a
commented-out logo-fallback block.

---

## 7. Layouts

`draw()` establishes the order — background, grid, header, layout body, footer:

```kotlin
fun draw(canvas: Canvas) {
    canvas.drawColor(GColor.WHITE)
    drawGrid(canvas, 0f, 0f, gridW, gridH)   // grid spans the FULL page, margins included
    drawHeader(canvas)
    when (data.layout) {
        "1x12" -> draw1x12(canvas)
        "3x4"  -> draw3x4(canvas)
        else   -> draw2x6(canvas)            // default
    }
    drawFooter(canvas)
}
```

`drawGrid` walks integer millimetres across the page (`for (i in 0..w.toInt())`), drawing a major line
every 5th and a minor line otherwise. There is no grid clipping to a content area — the pink grid runs
edge to edge, exactly as ECG paper does. The inline comment claiming the grid starts at a 5 mm margin is
stale; `drawGrid` is called at `(0, 0)`.

### 7.1 Header — shared by all three layouts

`yBase = MARGIN_TOP - 3f = 2f`, line height `lh = 5f`. Left column starts at `x = 5` (portrait) or
`x = 10` (landscape).

| Column | Portrait X | Landscape X | Rows (top-of-text, mm) |
|---|---:|---:|---|
| 1 — Patient | 5 | 10 | Name 2 · Age 7 · Gender 12 · `ECG Type: Standard` 17 · Date & Time 22 · spec line 27 |
| 2 — Intervals | 60 | 95 | HR 2 · RR 7 · PR 12 · QRS 17 · QT 22 |
| 3 — Derived | 90 | 125 | QTc 2 · QTcF 7 · RV5/SV1 12 · RV5+SV1 index 17 |
| 4 — Organisation | 140 | 220 | `orgName` 2 · `orgAddress` 7 · `+91 <phone>` 12 |

Two details worth knowing:

- **Spec line.** `"25.0 mm/s   0.5-25 Hz   AC:<acFilter>Hz   10.0 mm/mV"`. Speed and gain are
  interpolated from the constants, so they can never disagree with what was actually drawn. Only the AC
  filter comes from the data.
- **Sokolow-Lyon flag.** `RV5 + SV1 ≥ 3.5 mV` appends a `*` to the index line
  ([ReportScreen.kt:677](../ReportScreen.kt#L677)) — a possible-LVH marker. There is no legend explaining
  the asterisk anywhere on the page.

The header occupies through y = 27 mm. The landscape body starts at y = 30 mm and the portrait body at
y = 33 mm, leaving **3 mm** of clearance in landscape. A longer organisation name does not wrap — it runs
straight into whatever sits to its right.

### 7.2 `draw1x12` — portrait, 12 stacked leads

| Quantity | Expression | Value |
|---|---|---|
| `topOffset` | `MARGIN_TOP + 28` | 33 mm |
| `usableH` | `297 − 33 − 5 − 12` | 247 mm |
| `cellH` | `usableH / 12` | ≈ 20.583 mm |
| Lane centre, lead *i* | `topOffset + i × cellH + cellH/2` | 43.3, 63.9, 84.5 … |
| Label | `x = 10`, `y = midY − 10` | above the trace |
| Calibration | `x = MARGIN_LEFT = 5` | no pad |
| Waveform | `x0 = 18`, `width = 210 − 5 − 5 − 15` | 185 mm |

Lead order comes from the external `ALL_LEADS`. Source data: `leadData3500` (7 s per lead).

### 7.3 `draw2x6` — landscape, six rows of two leads plus a rhythm strip

| Quantity | Expression | Value |
|---|---|---|
| `startY` | `MARGIN_TOP + 25` | 30 mm |
| `rowH` | — | 22 mm |
| `leftMargin` | `MARGIN_LEFT + 8` | 13 mm |
| Left label / waveform | `22` / `27`, width `123` | ends at 150 mm |
| Dashed divider | `13 + 14 + 123 + 5` | 155 mm |
| Right label / waveform | `160` / `165`, width `123` | ends at 288 mm |
| Rhythm strip centre | `30 + 6×22 + 2 + 7.5` | 171.5 mm |
| Rhythm waveform | `x0 = 27`, width `297 − 13 − 5 − 25` | 254 mm |

Pairing is fixed and **not** the conventional Cabrera or standard column order:

```
I / V1    II / V2    III / V3    aVR / V4    aVF / V5    aVL / V6
```

Note the third column: `aVR, aVF, aVL` — **aVF before aVL**. The standard augmented-lead order is
aVR, aVL, aVF. If this is intentional it deserves a comment in the source; if not, it is a report defect
and clinicians will notice.

Row leads use `leadData1750` (3.5 s); the rhythm strip uses `leadData5000["II"]` (10 s).

### 7.4 `draw3x4` — landscape, four rows of three leads plus a rhythm strip

| Quantity | Expression | Value |
|---|---|---|
| `startY` | `MARGIN_TOP + 25` | 30 mm |
| `rowH` | — | 30 mm |
| `leftPad` | `MARGIN_LEFT + 8 + 10` | 23 mm |
| `leadW` | — | 80 mm |
| Column X | `23 + c × 90` | 23, 113, 203 (ends 283 mm) |
| Row centres | `30 + r × 30 + 15` | 45, 75, 105, 135 mm |
| Rhythm strip centre | `30 + 4×30 + 3 + 7.5` | 160.5 mm |
| Rhythm waveform | `x0 = 27`, width `254` | — |

Groups here **are** in standard order:

```
I   II   III
aVR aVL  aVF
V1  V2   V3
V4  V5   V6
```

Dashed dividers are drawn after columns 0 and 1 only. Row leads use `leadData1250` (2.5 s).

### 7.5 Footer

Selected by orientation. Both variants draw a doctor block on the left, a bordered `CONCLUSION` box on
the right, and a centred device/compliance line.

| | Portrait | Landscape |
|---|---|---|
| Doctor block top | y = 277 | y = 192 |
| Conclusion box | x 95, y 272, 105 × 18 mm | x 140, y 185, 145 × 20 mm |
| Box title | `CONCLUSION`, 7 pt bold, centred | `CONCLUSION`, 9 pt bold, centred |
| Conclusion grid | 3 columns, 3.5 mm row pitch | 3 columns, 5 mm row pitch, 5 mm column gap |
| Centred footer line | y = 292.5, 7 pt | y = 205.5, 8 pt |

Footer text: `Deckmount Electronics Pvt Ltd | Rhythm Ultra Max | IEC 60601 | <last 4 of serial> | Made in India`,
horizontally centred by measuring the string and dividing the slack.

Two behaviours in the conclusion box are easy to miss:

1. **Zero-HR override.** When `data.hr <= 0` the list is replaced wholesale with
   `listOf("No ECG data available")` — the app must never print an interpretation for a dead trace. This
   mirrors the desktop `RPT-13` rule. Otherwise blanks and `"---"` placeholders are filtered out.
2. **Silent overflow.** Items whose row would exceed the box height are skipped
   ([:935](../ReportScreen.kt#L935), [:985](../ReportScreen.kt#L985)). Since `ty` grows monotonically, the
   first skip effectively caps the list — but the loop keeps iterating, and the portrait variant numbers
   items by loop index `i` while the landscape variant uses a separate counter `sr` that only advances on
   drawn items. **The two orientations therefore number a truncated list differently.** With three columns
   and the current box heights, roughly 9 (portrait) / 9 (landscape) items fit; the desktop pipeline caps
   conclusions at 5, so overflow should not occur in practice — but nothing in this file enforces that.

---

## 8. Calibration pulse

```
        ┌───────────┐   ← y − gain (10 mm above baseline)
        │           │
 ───────┘           └───────   ← baseline y
 x    x+2         x+7      x+9
```

Five line segments, 9 mm wide, 10 mm tall — a standard 1 mV / 0.2 s reference. At 25 mm/s the 5 mm top
segment is exactly 200 ms, so the pulse also reads as one large grid box wide.

`drawCalibrationPad(x, …)` simply calls `drawCalibration(x + 4f, …)`; the landscape layouts pass
`leftMargin − 4f`, so the 4 mm pad cancels out and the pulse begins at `leftMargin` exactly. The
indirection is a leftover from the earlier port and could be removed.

---

## 9. Waveform rendering

```kotlin
private fun drawWaveform(canvas, samples, x0Mm, y0Mm, widthMm, gainMmPerMv) {
    if (samples.size < 2) return

    val sorted   = samples.sorted()
    val baseline = sorted[sorted.size / 2]                       // median = robust DC estimate
    val scale    = (1f / ADC_PER_MV) * gainMmPerMv * pxPerMm     // ADC → device units

    var prevX = p(x0Mm)
    var prevY = p(y0Mm) - (samples[0] - baseline) * scale

    for (i in 1 until samples.size) {
        val xMm = x0Mm + i * mmPerSample
        if (xMm > x0Mm + widthMm) break                          // clip, silently
        val x = p(xMm)
        val y = p(y0Mm) - (samples[i] - baseline) * scale
        canvas.drawLine(prevX, prevY, x, y, waveP)
        prevX = x; prevY = y
    }
}
```

**Vertical scale.** `(1/1.8) × 10 × pxPerMm ≈ 5.556 × pxPerMm`. One ADC unit is 1/1.8 mV, and at
10 mm/mV that is 5.556 mm — so a 1.8-ADC deflection draws exactly 10 mm, one large box per millivolt.
Y is subtracted because canvas Y grows downward while ECG amplitude grows upward.

**Baseline removal.** The **median** of the strip is subtracted, not the mean. On an ECG the median sits
on the isoelectric line because most samples are baseline; the mean would be dragged upward by R waves.
This centres each lane independently and removes per-lead DC offset.

Two consequences worth stating plainly:

- Baseline is computed **per strip**, so the same lead can sit at slightly different vertical offsets in
  the 2×6 body (1750 samples) and in the 10 s rhythm strip (5000 samples).
- The method is robust to noise but not to *sustained* pathology: a strip where elevation persists across
  more than half the samples has that elevation partially normalised away. For the 2.5–10 s windows used
  here this is not a practical concern, but it is the reason the desktop pipeline does its own DC removal
  upstream rather than relying on the renderer.

**No decimation.** Every sample becomes one `drawLine` call. A portrait report is
12 × 3500 ≈ **42,000** line calls; a landscape 2×6 report is 12 × 1750 + 5000 = **26,000**. At the 2.5×
oversampled preview size that is many samples per device pixel, so the cost buys no visible detail. The
desktop app solved the equivalent problem by decimating 2× (see the hyperkalemia display notes in the
[README](../README.md)); the same optimisation is available here — reduce the sample stride until roughly
2 points land per device pixel column.

**No NaN guard.** `Float.NaN` in the sample list propagates into `drawLine`, which silently drops the
segment. Unlike the desktop `connect='finite'` path there is no visible break, but there is a gap. The
caller is responsible for sanitising.

---

## 10. Compose layer

### 10.1 `ECGReportScreen(reportData, onNavigateBack)`

A Material 3 `Scaffold`. The top bar carries a back button, a **Share** action and a **Download** action;
the body is `ECGReportCanvas`.

**Share flow** ([:228–263](../ReportScreen.kt#L228)):

1. `shareMutex.tryLock()` — returns immediately if a share is already running.
2. `isSharing = true`, spinner replaces the icon, and both actions disable
   (`enabled = !isSharing && !isExporting`).
3. `generateECGPdf` runs on `Dispatchers.IO`.
4. `FileProvider.getUriForFile(context, "${packageName}.fileprovider", file)` produces a content URI.
5. Every activity resolving the `ACTION_SEND` intent is granted `FLAG_GRANT_READ_URI_PERMISSION`
   explicitly, then `Intent.createChooser` is launched.
6. `finally` clears `isSharing` and unlocks — so a failure cannot wedge the button.

**Download flow** ([:266–308](../ReportScreen.kt#L266)):

1. Generate the PDF on `Dispatchers.IO` into the cache directory.
2. Hold it in `tempPdfFile`, then launch the Storage Access Framework
   (`ActivityResultContracts.CreateDocument("application/pdf")`) with a suggested name
   `RhythmPro_ECG_<yyyyMMdd_HHmmss>.pdf`.
3. On a returned URI, stream the temp file into it and **delete the temp file**, then toast success.
4. On cancellation, delete the temp file and clear state.

Using SAF rather than a hardcoded path is the right choice — it needs no storage permission on any API
level and lets the user pick the destination.

> ⚠️ **`isExporting` is never set to `true`.** Both assignments are commented out
> ([:270](../ReportScreen.kt#L270), [:285](../ReportScreen.kt#L285)). The download spinner therefore never
> appears, and the guard `if (isExporting || isSharing) return@IconButton` cannot block a second tap
> during generation. Rapid double-tapping Download generates two PDFs and opens two file pickers. Share
> is protected by its mutex; Download is not. Uncommenting both lines restores the intended behaviour.

### 10.2 `ECGReportCanvas(reportData, modifier)`

```
BoxWithConstraints
  └─ pxPerMm      = availableWidthPx / pageWidthMm
     displayW/H   = the exact on-screen page rectangle
     bmpW/H       = display × 2.5          ← quality oversample
     renderPxPerMm= pxPerMm × 2.5
```

The report is rasterised **once** into an ARGB_8888 bitmap inside `produceState` keyed on
`(reportData, bmpW, bmpH)`, on `Dispatchers.Default`. Pan and pinch-zoom then transform that finished
bitmap through a `graphicsLayer` (`scale` clamped to 0.5×–8×, origin centred). Redrawing vector content on
every gesture frame would be far too slow; this trades memory for a smooth gesture.

While the bitmap is null a centred spinner and *"Rendering ECG report…"* are shown. A render exception is
caught, logged under tag `ECGReportScreen`, and leaves the state null — the spinner then never resolves,
which is the one failure mode with no user-visible explanation.

> ⚠️ **Bitmap memory.** ARGB_8888 at 2.5× in both axes is 6.25× the pixel count of the displayed page.
> On a 1080 px-wide phone rendering portrait A4: 2700 × 3819 × 4 bytes ≈ **41 MB** for a single bitmap,
> and the previous one is only freed once `produceState` reassigns. On a low-RAM device this is a
> plausible `OutOfMemoryError`. Two mitigations, either sufficient: use `Bitmap.Config.RGB_565` (the
> report has no transparency — halves the cost), or lower `qualityMultiplier` to 1.5–2.0 and let the
> zoom gesture re-render at higher fidelity only when the user actually zooms in.

> ⚠️ **Unbounded pan.** `offsetX`/`offsetY` accumulate raw gesture deltas with no clamping
> ([:396–397](../ReportScreen.kt#L396)), so the report can be flung entirely off-screen with no way back
> except leaving and re-entering the screen. Clamping to the scaled bitmap bounds — or a double-tap
> reset — would fix it.

---

## 11. `generateECGPdf(context, data): File?`

```kotlin
val pageWPt = if (isPortrait) 595 else 842      // A4 in PDF points
val pageHPt = if (isPortrait) 842 else 595
val pxPerMm = pageWPt / pageWMm                  // ≈ 2.833 pt/mm

val document = PdfDocument()
val page     = document.startPage(PdfDocument.PageInfo.Builder(pageWPt, pageHPt, 1).create())
ECGReportRenderer(data, pxPerMm, context).draw(page.canvas)
document.finishPage(page)
```

Output lands in `context.cacheDir/shared_pdfs/RhythmPro_ECG_<yyyyMMdd_HHmmss>.pdf`. The cache directory is
deliberate: it is the standard `FileProvider` root for sharing, and Android may reclaim it under storage
pressure. Returns `null` on any exception, logged at `ERROR` under `ECGReportScreen` — every caller
null-checks and toasts *"PDF generation failed"*.

Because `pxPerMm` is derived from the page size in points, the renderer emits true A4 geometry: a 5 mm
grid box measures 5 mm on paper when printed at 100 % scale.

> **Housekeeping gap.** The Share path writes a fresh PDF into `shared_pdfs/` on **every** tap and never
> deletes it (only the Download path cleans up, and only its own temp file). Repeated sharing grows the
> cache monotonically until Android reclaims it. Pruning files older than a few hours on entry to the
> screen would be a two-line fix.

> **`SimpleDateFormat` + `Locale.getDefault()`.** Under a non-Gregorian default calendar (e.g. a
> Thai-locale device) `yyyy` yields Buddhist-era years, producing filenames like `RhythmPro_ECG_2569…`.
> The desktop doctor-review backend parses device id and timestamp **out of the filename**, so a
> non-Gregorian year would break that matching. `Locale.US` is the correct choice for machine-readable
> timestamps.

---

## 12. Divergence from the sibling copy

[test_cloud_connection/ReportScreen.kt](../test_cloud_connection/ReportScreen.kt) is an **older** copy of
this file. Known differences:

| | Root `ReportScreen.kt` | `test_cloud_connection/` copy |
|---|---|---|
| Margins | 5 mm | 10 mm |
| `ADC_PER_MM` | absent | `2.75f` |
| Grid size constants | present (`GRID_W_*` / `GRID_H_*`) | absent |
| `ALL_LEADS` / `ECGReportRenderData` | resolved implicitly | imported explicitly from `livemonitorecg` |
| HRV imports | present (unused) | absent |

Treat the **root** file as current. The duplicate is a snapshot and should either be deleted or clearly
marked, because a future edit applied to the wrong copy will silently do nothing.

---

## 13. Dead code and unused imports

Confirmed by occurrence count across the file:

| Symbol | Status |
|---|---|
| `adcScaleFactor` ([:466](../ReportScreen.kt#L466)) | Declared and computed, **never read**. `drawWaveform` uses `ADC_PER_MV` directly. Leaving it in place invites someone to "fix" the scale by editing a constant that does nothing. |
| `MARGIN_PDF` | Declared, never referenced |
| `tp6`, `tp20_5B` | Paints declared, never used |
| `tp8B` | Used only in a commented-out logo fallback |
| `HRVAnalysisResult`, `HRVReportRenderer`, `EcgPdfGenerator` | Imported, never used |
| `kotlinx.coroutines.flow.first` | Imported, never used |
| `BitmapFactory`, `Rect`, `RectF`, `R` | Imported for the commented-out logo block only |
| `data.pqrstAxis` | Referenced only inside a comment |
| Commented `draw()` variant ([:568](../ReportScreen.kt#L568)) | Superseded by the grid-constant version |
| Commented `drawCalibration` variant ([:1030](../ReportScreen.kt#L1030)) | "Left bottom line removed" experiment |
| Commented `drawWaveform` variant ([:1063](../ReportScreen.kt#L1063)) | Pre-median-baseline version |

None of it is harmful today. All of it costs a future reader time, and the `adcScaleFactor` case is an
active trap.

---

## 14. Threading

| Work | Dispatcher | Why |
|---|---|---|
| Bitmap rasterisation | `Dispatchers.Default` | CPU-bound drawing; `produceState` cancels it automatically when the composable leaves |
| PDF generation | `Dispatchers.IO` | Renders and writes a file |
| Copying the PDF into the SAF URI | `Dispatchers.IO` | Stream copy |
| Toasts, UI state | Main | Explicit `withContext(Dispatchers.Main)` inside IO blocks |

Nothing renders on the main thread. `shareMutex` prevents concurrent share operations; see §10.1 for the
missing equivalent on Download.

---

## 15. Making changes safely

**Changing paper speed or gain.** `FIXED_WAVE_SPEED` and `FIXED_WAVE_GAIN` are the only inputs to
`mmPerSample` and to the waveform scale, and both are interpolated into the printed spec line — so the
header can never lie about what was drawn. But the sample-count table in §3.3 assumes 25 mm/s: at
50 mm/s, `leadData3500` would need 370 mm and be clipped to just over half. **Changing the speed requires
re-deriving §3.3 and re-checking every lane width.**

**Adding a layout.** Add the branch in `draw()`, write a `drawNxM` following the existing shape
(centre-line per lane → `drawCalibrationPad` → `drawTxt` label at `midY − 10` → `drawWaveform`), pick the
matching `leadDataN` map, and confirm `samples × 0.05 mm ≤ laneWidth`.

**Moving header or footer text.** Remember `drawTxt` takes the **top** of the text, not the baseline
(§5.1). Verify in both orientations — the footers are separate functions with independent coordinates,
and it is easy to fix one and leave the other.

**Before merging any change to this file**, export a PDF and print it at 100 % scale, then measure a
large grid box with a ruler. It must be 5 mm. That one measurement catches almost every unit-conversion
regression this file can suffer.

---

## 16. Verification checklist

Mirrors the format of the desktop [EXE test checklist](EXE_TEST_CHECKLIST.md). Prefix `KRP`.

**Rendering**

- [ ] `KRP-01` All three layouts render: `"1x12"`, `"2x6"`, `"3x4"`.
- [ ] `KRP-02` An unrecognised `layout` string falls back to `"2x6"` rather than crashing.
- [ ] `KRP-03` Grid is 1 mm minor / 5 mm major and spans the full page edge to edge.
- [ ] `KRP-04` Every lane shows its calibration pulse: 9 mm wide, 10 mm tall.
- [ ] `KRP-05` Every lane shows its lead label above the trace.
- [ ] `KRP-06` A missing lead key renders an empty lane — label and calibration intact, no crash.
- [ ] `KRP-07` A single-sample lead list renders nothing and does not throw.
- [ ] `KRP-08` The 2×6 rhythm strip uses lead II from `leadData5000` and spans the full 10 s.

**Geometry — the checks that need a ruler**

- [ ] `KRP-09` Exported PDF printed at 100 % scale: one large grid box measures **5.0 mm**.
- [ ] `KRP-10` The calibration pulse measures **10 mm** tall and **5 mm** across its top segment.
- [ ] `KRP-11` A 1 mV deflection in the trace measures 10 mm — one large box.
- [ ] `KRP-12` One second of signal spans 25 mm — five large boxes.
- [ ] `KRP-13` No strip is clipped: the last sample of each stream is visible within its lane.
- [ ] `KRP-14` Nothing overflows the page in either orientation, including a long organisation name.

**Header and footer**

- [ ] `KRP-15` Blank patient fields render as `-`, not as an empty gap or `null`.
- [ ] `KRP-16` The spec line reads `25.0 mm/s   0.5-25 Hz   AC:<n>Hz   10.0 mm/mV`.
- [ ] `KRP-17` RV5, SV1 and the index print to 3 decimal places.
- [ ] `KRP-18` `RV5 + SV1 ≥ 3.5` appends `*`; below the threshold it does not.
- [ ] `KRP-19` The footer shows the **last 4** characters of `machineSerial` and is horizontally centred.
- [ ] `KRP-20` `hr <= 0` prints exactly **"No ECG data available"** in the conclusion box and nothing else.
- [ ] `KRP-21` Blank conclusions and `"---"` are filtered out.
- [ ] `KRP-22` Conclusions stay inside the box border in both orientations; numbering is sequential from 1.

**Interaction**

- [ ] `KRP-23` The spinner and *"Rendering ECG report…"* appear, then resolve to the report.
- [ ] `KRP-24` Pinch-zoom clamps at 0.5× and 8×.
- [ ] `KRP-25` Pan moves the report smoothly. *(Known issue: pan is unbounded — record the behaviour.)*
- [ ] `KRP-26` Rotating the device re-renders at the new width without crashing.
- [ ] `KRP-27` Back navigation leaves the screen cleanly and cancels any in-flight render.

**Share and download**

- [ ] `KRP-28` Share opens the system chooser; the recipient app can open the PDF.
- [ ] `KRP-29` The share spinner shows during generation and both actions disable.
- [ ] `KRP-30` Rapid double-tap on **Share** starts exactly one operation (mutex).
- [ ] `KRP-31` Rapid double-tap on **Download** starts exactly one operation.
      *(Known issue: currently fails — `isExporting` is never set. See §10.1.)*
- [ ] `KRP-32` Download opens the SAF picker pre-filled with `RhythmPro_ECG_<timestamp>.pdf`.
- [ ] `KRP-33` Saving writes a valid PDF to the chosen location and toasts success.
- [ ] `KRP-34` Cancelling the picker deletes the temp file and leaves no orphan in `cacheDir/shared_pdfs`.
- [ ] `KRP-35` The exported PDF opens in Drive, Adobe Reader and Chrome.
- [ ] `KRP-36` Filenames use Gregorian years on a device with a non-Gregorian default locale.
      *(Known issue — see §11.)*

**Robustness**

- [ ] `KRP-37` Rendering ten reports in sequence causes no `OutOfMemoryError` on a 3 GB device.
- [ ] `KRP-38` `NaN` in a sample list does not crash the renderer.
- [ ] `KRP-39` A render failure is logged under tag `ECGReportScreen` and does not crash the app.
- [ ] `KRP-40` `generateECGPdf` returning `null` surfaces *"PDF generation failed"* to the user.

---

## 17. Open items

Ordered by impact. None is a crash today; the first two are the ones a user would notice.

| # | Item | Section |
|---|---|---|
| 1 | `isExporting` never set — Download has no spinner and no double-tap guard | §10.1 |
| 2 | 2×6 pairs augmented leads as `aVR, aVF, aVL` rather than `aVR, aVL, aVF` — confirm intent | §7.3 |
| 3 | 41 MB ARGB_8888 bitmap on a 1080 px device; OOM risk on low-RAM hardware | §10.2 |
| 4 | Pan is unbounded — the report can be flung off-screen irrecoverably | §10.2 |
| 5 | `shared_pdfs/` grows without cleanup on every Share | §11 |
| 6 | `Locale.getDefault()` in filename timestamps breaks backend filename parsing on non-Gregorian locales | §11 |
| 7 | `isForPdf` inferred from `pxPerMm` — misfires in small/split-screen windows | §5.2 |
| 8 | No decimation: up to 42,000 `drawLine` calls per report | §9 |
| 9 | Conclusion overflow silently truncates, and the two orientations number the survivors differently | §7.5 |
| 10 | `adcScaleFactor` is computed and never used | §13 |
| 11 | Duplicate stale copy at `test_cloud_connection/ReportScreen.kt` | §12 |
| 12 | Margin comments say 10 mm; the constants are 5 mm | §4 |
