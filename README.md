# Physically-Informed Domain-Independent Data-Driven Inundation Forecast

This repository contains a domain-independent, physically-informed data-driven model for producing multi-step flood inundation forecasts. The model follows the approach presented in:

> Felix Schmid, Leonie Müller, Jorge Leandro, *A physically informed domain-independent data-driven inundation forecast model*, Water Research, 2025. https://doi.org/10.1016/j.watres.2025.124819
---

Note: We are actively working on improvements and addressing the limitations discussed in the publication.
Future updates will include extended experiments, enhanced physical constraints, and broader benchmarking.
Stay tuned for upcoming releases.

## Repository contents

- `src/`
  - `arcitectures/` — ai model architectures
  - `train/` — config, training loop, loss function, metrics, checkpoint utilities
- `README.md` — this file.
- `LICENSE`
-  the actuall pdf of the published work (the quality was downsampled, for a high quality pdf please refer to the Water Research page via: https://doi.org/10.1016/j.watres.2025.124819) 
- `requirements.txt` — pinned Python dependencies.

---

## Installation
we used a windows operating system at the time of the research

pip install -r requirements.txt
