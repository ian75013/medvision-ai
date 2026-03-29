# Segmentation UI Guide (Streamlit)

This guide explains how to use segmentation features in the MedVision Streamlit app and how to diagnose empty or weak masks.

## 1. Scope

This document covers segmentation usage in:

- streamlit_app.py (Prediction Studio)
- multitask segmentation tracks:
  - brain_tumor_segmentation
  - chest_xray_segmentation

## 2. Quick start

1. Start the app:

```bash
streamlit run streamlit_app.py
```

2. In the left panel, select a segmentation problem:

- Brain Tumor Segmentation + Classification
- Chest X-ray Segmentation + Abnormality Classification

3. Upload an image and select one or more available models.

4. Use the Mask threshold slider in the sidebar to control binary mask generation.

## 3. What the app shows for segmentation

For each selected model, the app displays:

- Predicted class and confidence
- Binary mask
- Overlay (mask blended over image)
- Probability map (raw segmentation output before thresholding)
- Segmentation stats table with:
  - threshold
  - mask_foreground_ratio
  - prob_min
  - prob_mean
  - prob_max

## 4. Understanding the threshold

The model outputs a probability map in [0, 1].
The binary mask is created as:

- mask = probability_map >= threshold

Interpretation:

- Higher threshold -> stricter mask (smaller regions)
- Lower threshold -> larger mask (more permissive)

Recommended workflow:

1. Start at 0.50
2. If mask is empty, try 0.30
3. If still empty, try 0.20
4. If mask becomes too large/noisy, increase threshold gradually

## 5. Why you may see "nothing"

If your binary mask is black and overlay looks unchanged, usually one of these is happening:

1. The threshold is too high for this image.
2. The model predicts very low probabilities overall.
3. The selected model is not well trained for this data distribution.
4. Preprocessing mismatch (image type/content differs from training data).

Use the Probability map + prob_max to diagnose:

- prob_max < threshold -> binary mask will be empty
- prob_max near 0.5 with localized bright areas -> lower threshold slightly

## 6. Reading segmentation metrics in the UI

Key values in the segmentation stats table:

- mask_foreground_ratio:
  - 0.0 means no positive pixels in binary mask
  - very high values can indicate over-segmentation
- prob_mean:
  - overall confidence level of segmentation branch
- prob_max:
  - highest predicted probability in image
  - useful to tune threshold quickly

## 7. Practical troubleshooting checklist

If segmentation output is empty or poor:

1. Lower threshold to 0.30 or 0.20.
2. Check prob_max and probability map brightness.
3. Try another image from the same dataset distribution.
4. Compare multiple models in parallel.
5. Validate model artifacts and metrics files exist in artifacts/.
6. Re-run segmentation training if all models stay near-zero on valid samples.

## 8. Data and model sanity checks

Before concluding the UI is wrong:

1. Verify segmentation model exists in artifacts/models.
2. Verify metrics JSON exists in artifacts/reports.
3. Verify overlays were generated during training in artifacts/overlays.
4. Ensure selected problem is a segmentation track in registry.

## 9. Best practices for demos

1. Use representative images close to training distribution.
2. Keep threshold visible in screenshots or recordings.
3. Include probability map next to binary mask and overlay.
4. Report mask_foreground_ratio when discussing results.

## 10. Related docs

- docs/FASTAPI_STREAMLIT_ALIGNMENT.md
- docs/DATASETS.md
- docs/MLOPS_GUIDE.md
- docs/KNOWN_GAPS.md

If segmentation behavior changes in the UI (new controls, new metrics, new output format), update this guide in the same PR.
