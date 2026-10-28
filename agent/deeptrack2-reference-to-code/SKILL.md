---
name: deeptrack2-reference-to-code
description: Generate standard, runnable DeepTrack2 Python code from user descriptions or microscopy reference images, possible for training the machine learning model further.
---

# DeepTrack2 Reference-to-Code Skill

## Goal

Generate standard, runnable DeepTrack2 Python code from either:

1. A user task description, or
2. A microscopy reference image plus a desired synthetic-data goal.

The output should approximate the visual style or imaging modality of the reference image while following DeepTrack2 code patterns, ready for machine learning application.

## When to use this skill

Use this skill when the user asks to:

- generate DeepTrack2 code
- simulate microscopy images
- create synthetic microscopy datasets
- match a microscopy modality from a reference image
- produce DeepTrack2 examples from provided documentation or code examples
- refactor or standardize DeepTrack2 code
- prepare the simulation data for machine learning model

## Source priority

Follow this order:

1. Local examples in `examples/`
2. Docstring inside function/class
3. Official DeepTrack2 documentation/example
4. User-provided code or images

Do not invent unsupported DeepTrack2 APIs.

## Important library rules

- Use `import deeptrack as dt`
- Prefer complete runnable Python scripts
- Use type hints where practical
- Use `if __name__ == "__main__":`
- Do not use private or undocumented APIs
- Must wrapper new operators use `dt.Lambda`, read `/examples/Lambda.py` everytime creat scripts
- Chaining as a pipeline using `>>`, `^`, and `&` as much as possible
- DeepTrack2 2.0+ does not support TensorFlow
- Do not generate TensorFlow/Keras-based DeepTrack code unless the user explicitly asks for legacy DeepTrack 1.7 code

## General workflow

1. Identify the user’s goal:
   - particle simulation
   - cell simulation
   - brightfield image generation
   - fluorescence image generation
   - holography-like image generation
   - dataset generation
   - augmentation / noise modeling
   - training-data pipeline

2. Search local examples for the closest pattern.

3. Extract:
   - imports
   - initialization style
   - scatterer/sample construction
   - optics/microscope construction
   - noise and augmentation steps
   - `.resolve()` or documented execution pattern

4. Generate code in the same style as examples.

5. Validate:
   - every `dt.<name>` appears in local examples or docs
   - no fake APIs
   - no TensorFlow unless legacy requested
   - code is complete and runnable

## Reference-image workflow

When the user provides a microscopy example image:

1. Inspect the image visually.

2. Infer likely modality:
   - brightfield
   - fluorescence
   - holography
   - phase contrast / DIC-like
   - unknown

3. Extract visual traits:
   - background brightness
   - object shape
   - object density
   - contrast polarity
   - blur / PSF size
   - noise type
   - illumination gradient
   - field of view
   - artifacts

4. Map traits to DeepTrack2 components:
   - particles/cells -> scatterers
   - optics -> Brightfield / Fluorescence / holography-related examples
   - blur -> optics or postprocessing
   - camera noise -> Poisson or Gaussian noise examples
   - variability -> random parameter distributions
   - normalization -> documented preprocessing or augmentation examples

5. Generate a runnable DeepTrack2 script that creates visually similar synthetic images.

6. Clearly mark inferred parameters as estimates.

## Modality mapping

### Brightfield-like

Visual clues:

- gray or bright background
- darker objects
- halos or soft shadows
- uneven illumination possible

Prefer examples involving:

- particles or cells
- brightfield optics
- background variation
- Gaussian noise
- blur

### Fluorescence-like

Visual clues:

- dark background
- bright objects
- glowing cells / spots
- shot-noise appearance

Prefer examples involving:

- fluorescence optics
- bright scatterers
- Poisson noise
- Gaussian blur
- intensity variation
- normalization

### Holography-like

Visual clues:

- diffraction rings
- interference fringes
- phase-like contrast
- central particle with halo

Prefer examples involving:

- holography examples
- particle scatterers
- coherent-style optical setup
- ring-like artifacts
- background correction

## Output format

Return:

1. Inferred goal or modality
2. Key visual/code assumptions
3. Complete Python code
4. Notes on parameters to calibrate
5. Caveats if the API or modality is uncertain

## Standard simulation code style

Generated simualtion code should follow this structure:

```python
from __future__ import annotations

import deeptrack as dt
import matplotlib.pyplot as plt


def build_pipeline():
    ...


def main() -> None:
    pipeline = build_pipeline()
    image = pipeline.resolve()

    plt.imshow(image.squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()