"""V2Net capacity presets for the size sweep.

Hardware is no longer ESP32-locked (STM32-class / Cortex-M7 or higher), so we let
the data choose the size on the accuracy-vs-deployability curve rather than
assuming the smallest model wins. SupCon + adversarial heads are ON for all sizes
so the sweep also tests the hypothesis that a larger backbone is needed for the
adversarial domain-alignment head to work (a too-small backbone can bottleneck it).

deploy params (backbone+classifier, what actually ships) are reported separately
from train params (which include the training-only SupCon/adversarial heads).
"""

SIZE_PRESETS = {
    # name: backbone widths + classifier width.  ~train params (supcon+adv, 36 subj)
    "S": dict(widths=[32, 64, 128], fc_units=64),            # ~67K train
    "M": dict(widths=[48, 96, 192, 256], fc_units=128),      # ~232K train
    "L": dict(widths=[64, 128, 256, 512, 512], fc_units=256),  # ~947K train
}


def preset(name: str, *, supcon=True, adversarial=True, adv_lambda_max=1.0, **extra) -> dict:
    mc = dict(SIZE_PRESETS[name])
    mc.update(supcon=supcon, adversarial=adversarial)
    # Larger fine-tune batch: the model is tiny, so 64 starves the GPU. 256 keeps
    # the 4080 fed and cuts the number of Python-loop iterations per epoch.
    mc.setdefault("batch_size", 256)
    # CRITICAL: the adversarial head must be EXERCISED, not just present. A zero
    # adv_lambda_max builds the head but never adds its loss. Default it on so the
    # gradient-reversal domain alignment actually runs and gets tuned in the sweep.
    if adversarial:
        mc["adv_lambda_max"] = adv_lambda_max
    mc.update(extra)
    return mc
