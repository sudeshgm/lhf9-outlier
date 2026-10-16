# LHF9 Emergent Outlier Rejection — non-VGGT check

CPU side code for the first pass of the kill test:

Does the layer-wise clean/distractor separation curve appear in one non-VGGT feed-forward geometry model (DUSt3R)?

## Model
- `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`
- Loaded via `AsymmetricCroCo3DStereo.from_pretrained`
- Frozen weights only
- Decoder layers (12) are probed

## Protocol
- Clean views vs distractor views (other scenes)
- View-level only
- Separation = mean feature cosine similarity (clean-clean pairs) − mean feature cosine similarity (clean-distractor pairs)
- Recorded at first / middle / final decoder layer

## Run on the GPU machine
```bash
cd lhf9-outlier
git pull
export HF_HOME="$PWD/hf-cache"
export HUGGINGFACE_HUB_CACHE="$PWD/hf-cache/hub"
python scripts/lhf9_dust3r_separation.py
```

Results are written to `results/lhf9_separation.json` and `results/lhf9_separation.txt`.
