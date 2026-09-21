"""
Generate per-τ ground-truth labels for the 3-class turn-taking classifier from per-participant VAD + transcript JSONLs; emits one json
per τ in the grid. 
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# -----------------------------------------------------------------------------
# Shared helpers via sys.path 
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]                 # repo root (parent of scripts/)
SHARED_SCRIPTS_DIR = THIS_FILE.parent  # download_annotated_interactions.py is a sibling
if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from download_annotated_interactions import (  # noqa: E402
    extract_interaction_id,
    extract_participant_id,
)

# ----- Sampling ------------------------------------------------------------
STRIDE_MS = 500

WINDOW_S = 2.0

# ----- τ grid --------------------------------------------------------------
TRAIN_TAU_MS = 400

TAU_GRID_MS = (100, 200, 400, 500, 800, 1600)

BC_MAX_DURATION_MS = 500
# Upper bound for BC-qualifying clipped-utterance duration.

BC_MAX_WORD_COUNT = 2
# Maximum word count for the "auto-qualify" leg of BC-qualifying.

SUBSTANTIVE_DURATION_MS = BC_MAX_DURATION_MS + 200
# Leg 1 of the substantive-B OR-gate. 

SUBSTANTIVE_WORD_COUNT = BC_MAX_WORD_COUNT + 1
# Leg 2 of the OR-gate. 
# Leg 3 of the OR-gate is implicit: any clipped B-utterance containing a
# non-BC-lexicon word is substantive. 

# ----- YIELD tolerance -----------------------------------------------------
A_YIELD_TOLERANCE_MS = 300
# Max A-continues-past-B-start overlap permitted for a clean YIELD.

# ----- Class integers ------------------------------------------------------
CLASS_HOLD = 0
CLASS_YIELD = 1
CLASS_BACKCHANNEL = 2
CLASS_INTERRUPT = 3 # excluded from training; retained for stats
CLASS_FAILED = 4     # excluded from training; retained for stats
CLASS_LAPSE = 5   # excluded from training; retained for stats

TRAINING_CLASSES = {CLASS_HOLD, CLASS_YIELD, CLASS_BACKCHANNEL}
EXCLUSION_CLASSES = {CLASS_INTERRUPT, CLASS_FAILED, CLASS_LAPSE}

CLASS_NAME = {
    CLASS_HOLD:        "HOLD",
    CLASS_YIELD:       "YIELD",
    CLASS_BACKCHANNEL: "BACKCHANNEL",
    CLASS_INTERRUPT:   "INTERRUPT",
    CLASS_FAILED:      "FAILED",
    CLASS_LAPSE:       "LAPSE",
}

# ----- Class imbalance strategy --------------------------------------------
# NO data-level rebalancing is performed here. All splits (train, val, test)
# are emitted at natural rate. Per-class imbalance is handled at training
# time via inverse-frequency class weights in nn.CrossEntropyLoss.

# ----- Backchannel vocabulary ---------------------------------------------
BC_LEXICON = frozenset({
    "mm", "hm", "hmm", "mhm", "mmhmm",
    "uhhuh", "huh",
    "ah", "aha", "oh",
    # short affirmatives
    #   Gravano & Hirschberg 2011 (Comp. Speech & Lang., Games Corpus)
    #   Ward & Tsukahara 2000
    #   Jurafsky et al. 1997 (SWBD-DAMSL "bf" backchannel tag)
    "yeah", "yep", "yup", "ok", "okay", "sure",
    "right", "true", "exactly", "totally",
    # short assessments / reactives
    #   Ruede, Müller, Stüker & Waibel 2017 (Interspeech, deep BC predictor)
    #   Lala, Inoue & Kawahara 2017
    "wow", "really", "gotcha",
})

AUDIO_DURATIONS_S = {
    "V00_S0691_I00000482": 240.0, "V00_S0743_I00000483": 156.0, "V00_S1005_I00000249": 246.0,
    "V00_S1132_I00000333": 262.0, "V00_S1132_I00000334": 194.0, "V00_S1132_I00000335": 290.0,
    "V00_S1132_I00000336": 354.0, "V00_S1132_I00000337": 122.0, "V00_S1132_I00000338": 238.0,
    "V00_S1132_I00000340": 100.0, "V00_S1132_I00000341": 82.0, "V00_S1132_I00000352": 204.0,
    "V00_S1132_I00000354": 132.0, "V00_S1132_I00000360": 202.0, "V00_S1132_I00000361": 160.0,
    "V00_S1204_I00000198": 198.0, "V00_S1204_I00000200": 96.0, "V00_S1204_I00000201": 458.0,
    "V00_S1204_I00000202": 126.0, "V00_S1204_I00000204": 420.0, "V00_S1204_I00000207": 282.0,
    "V00_S1204_I00000208": 278.0, "V00_S1204_I00000209": 176.0, "V00_S1279_I00000314": 300.0,
    "V00_S1279_I00000315": 254.0, "V00_S1279_I00000316": 322.0, "V00_S1279_I00000319": 402.0,
    "V00_S1279_I00000323": 432.0, "V00_S1279_I00000324": 486.0, "V00_S1279_I00000325": 302.0,
    "V00_S1288_I00000092": 498.0, "V00_S1288_I00000093": 156.0, "V00_S1288_I00000094": 300.0,
    "V00_S1288_I00000095": 386.0, "V00_S1288_I00000097": 240.0, "V00_S1288_I00000098": 250.0,
    "V00_S1288_I00000099": 500.0, "V00_S1288_I00000102": 212.0, "V00_S1288_I00000103": 606.0,
    "V00_S1289_I00000144": 216.0, "V00_S1289_I00000146": 216.0, "V00_S1289_I00000147": 234.0,
    "V00_S1289_I00000149": 258.0, "V00_S1289_I00000150": 188.0, "V00_S1289_I00000152": 160.0,
    "V00_S1289_I00000153": 200.0, "V00_S1289_I00000154": 200.0, "V00_S1289_I00000157": 156.0,
    "V00_S1291_I00000000": 262.0, "V00_S1291_I00000001": 448.0, "V00_S1291_I00000002": 402.0,
    "V00_S1291_I00000003": 376.0, "V00_S1291_I00000006": 246.0, "V00_S1291_I00000007": 324.0,
    "V00_S1291_I00000008": 266.0, "V00_S1291_I00000009": 162.0, "V00_S1295_I00000280": 90.0,
    "V00_S1295_I00000281": 350.0, "V00_S1295_I00000282": 462.0, "V00_S1295_I00000286": 502.0,
    "V00_S1295_I00000287": 348.0, "V00_S1295_I00000297": 396.0, "V00_S1295_I00000298": 228.0,
    "V00_S1297_I00000197": 248.0, "V00_S1297_I00000200": 122.0, "V00_S1297_I00000201": 158.0,
    "V00_S1297_I00000202": 124.0, "V00_S1297_I00000204": 168.0, "V00_S1297_I00000205": 220.0,
    "V00_S1297_I00000226": 280.0, "V01_S0283_I00000266": 128.0, "V01_S0325_I00000314": 198.0,
    "V02_S3160_I00000501": 196.0, "V02_S3227_I00000514": 214.0, "V02_S3447_I00000082": 216.0,
    "V02_S3613_I00000053": 182.0, "V02_S3653_I00000055": 172.0, "V02_S5099_I00000215": 174.0,
    "V02_S5262_I00000293": 164.0, "V03_S0120_I00000131": 118.0, "V03_S0132_I00000125": 190.0,
    "V03_S0132_I00000127": 356.0, "V03_S0132_I00000130": 420.0, "V03_S0132_I00000131": 744.0,
    "V03_S0132_I00000134": 198.0, "V03_S0132_I00000138": 162.0, "V03_S0132_I00000140": 158.0,
    "V03_S0132_I00000307": 222.0, "V03_S0134_I00000128": 250.0, "V03_S0134_I00000130": 190.0,
    "V03_S0134_I00000132": 170.0, "V03_S0134_I00000138": 182.0, "V03_S0139_I00000135": 230.0,
    "V03_S0139_I00000477": 60.0, "V03_S0139_I00000478": 106.0, "V03_S0139_I00000479": 180.0,
    "V03_S0139_I00000500": 106.0, "V03_S0142_I00000126": 212.0, "V03_S0142_I00000128": 314.0,
    "V03_S0145_I00000126": 238.0, "V03_S0145_I00000127": 184.0, "V03_S0145_I00000130": 134.0,
    "V03_S0145_I00000137": 148.0, "V03_S0145_I00000307": 178.0, "V03_S0146_I00000371": 212.0,
    "V03_S0146_I00000373": 190.0, "V03_S0154_I00000127": 270.0, "V03_S0154_I00000128": 218.0,
    "V03_S0154_I00000131": 178.0, "V03_S0154_I00000307": 208.0, "V03_S0155_I00000135": 246.0,
    "V03_S0155_I00000138": 184.0, "V03_S0155_I00000371": 178.0, "V03_S0155_I00000372": 188.0,
    "V03_S0155_I00000375": 176.0, "V03_S0158_I00000125": 192.0, "V03_S0158_I00000138": 218.0,
    "V03_S0162_I00000135": 228.0, "V03_S0162_I00000477": 148.0, "V03_S0162_I00000478": 228.0,
    "V03_S0162_I00000482": 118.0, "V03_S0162_I00000500": 184.0, "V03_S0163_I00000381": 232.0,
    "V03_S0164_I00000371": 178.0, "V03_S0164_I00000375": 590.0, "V03_S0166_I00000375": 368.0,
    "V03_S0176_I00000128": 196.0, "V03_S0176_I00000134": 202.0, "V03_S0176_I00000307": 142.0,
    "V03_S0176_I00000313": 176.0, "V03_S0176_I00000461": 78.0, "V03_S0190_I00000477": 202.0,
    "V03_S0190_I00000479": 350.0, "V03_S0190_I00000482": 528.0, "V03_S0190_I00000483": 122.0,
    "V03_S0191_I00000126": 338.0, "V03_S0191_I00000130": 284.0, "V03_S0191_I00000135": 460.0,
    "V03_S0193_I00000371": 226.0, "V03_S0193_I00000373": 270.0, "V03_S0193_I00000374": 248.0,
    "V03_S0193_I00000377": 272.0, "V03_S0194_I00000477": 132.0, "V03_S0194_I00000478": 188.0,
    "V03_S0194_I00000482": 220.0, "V03_S0194_I00000483": 490.0, "V03_S0199_I00000125": 170.0,
    "V03_S0199_I00000128": 200.0, "V03_S0199_I00000137": 286.0, "V03_S0199_I00000138": 222.0,
    "V03_S0199_I00000307": 160.0, "V03_S0199_I00000309": 230.0, "V03_S0199_I00000461": 166.0,
    "V03_S0201_I00000371": 230.0, "V03_S0201_I00000374": 212.0, "V03_S0201_I00000375": 256.0,
    "V03_S0201_I00000382": 206.0, "V03_S0201_I00000502": 280.0, "V03_S0202_I00000126": 266.0,
    "V03_S0202_I00000130": 244.0, "V03_S0202_I00000138": 192.0, "V03_S0202_I00000307": 184.0,
    "V03_S0202_I00000309": 220.0, "V03_S0205_I00000372": 212.0, "V03_S0205_I00000382": 184.0,
    "V03_S0205_I00000502": 206.0, "V03_S0205_I00000504": 172.0, "V03_S0208_I00000478": 218.0,
    "V03_S0208_I00000480": 190.0, "V03_S0208_I00000482": 108.0, "V03_S0210_I00000377": 196.0,
    "V03_S0210_I00000378": 136.0, "V03_S0210_I00000382": 270.0, "V03_S0210_I00000383": 224.0,
    "V03_S0212_I00000377": 295.17, "V03_S0212_I00000482": 92.0, "V03_S0212_I00000487": 210.0,
    "V03_S0212_I00000488": 182.0, "V03_S0213_I00000125": 380.0, "V03_S0213_I00000130": 250.0,
    "V03_S0213_I00000461": 136.0, "V03_S0214_I00000480": 270.0, "V03_S0217_I00000486": 202.0,
    "V03_S0217_I00000487": 182.0, "V03_S0217_I00000500": 134.0, "V03_S0220_I00000371": 120.0,
    "V03_S0220_I00000374": 168.0, "V03_S0220_I00000498": 128.0, "V03_S0221_I00000135": 204.0,
    "V03_S0221_I00000377": 102.29, "V03_S0221_I00000480": 182.0, "V03_S0221_I00000486": 284.0,
    "V03_S0221_I00000495": 186.0, "V03_S0225_I00000373": 168.0, "V03_S0225_I00000374": 260.0,
    "V03_S0226_I00000139": 222.0, "V03_S0226_I00000140": 224.0, "V03_S0241_I00000140": 136.0,
    "V03_S0241_I00000373": 288.0, "V03_S0242_I00000372": 228.0, "V03_S0242_I00000502": 156.73,
    "V03_S0243_I00000477": 168.0, "V03_S0243_I00000482": 186.0, "V03_S0243_I00000483": 272.0,
    "V03_S0258_I00000137": 146.0, "V03_S0258_I00000140": 196.0, "V03_S0259_I00000495": 304.0,
    "V03_S0260_I00000126": 150.0, "V03_S0260_I00000132": 188.0, "V03_S0260_I00000135": 336.0,
    "V03_S0260_I00000309": 170.0, "V03_S0261_I00000374": 436.0, "V03_S0262_I00000126": 166.0,
    "V03_S0262_I00000131": 216.0, "V03_S0262_I00000309": 176.0, "V03_S0263_I00000125": 150.0,
    "V03_S0263_I00000130": 210.0, "V03_S0263_I00000135": 196.0, "V03_S0275_I00000377": 225.58,
    "V03_S0275_I00000478": 230.0, "V03_S0281_I00000484": 156.0, "V03_S0292_I00000411": 212.0,
    "V03_S0292_I00000414": 192.0, "V03_S0292_I00000415": 196.0, "V03_S0292_I00000416": 204.0,
    "V03_S0292_I00000417": 290.0, "V03_S0292_I00000419": 206.0, "V03_S0292_I00000420": 178.0,
    "V03_S0292_I00000423": 230.0, "V03_S0295_I00000054": 342.0, "V03_S0310_I00000126": 192.0,
    "V03_S0310_I00000132": 192.0, "V03_S0310_I00000309": 192.0, "V03_S0310_I00000313": 164.0,
    "V03_S0310_I00000498": 183.6, "V03_S0328_I00000126": 148.0, "V03_S0328_I00000137": 188.0,
    "V03_S0329_I00000065": 278.0, "V03_S0332_I00000054": 294.0, "V03_S0333_I00000054": 160.0,
    "V03_S0333_I00000055": 144.0, "V03_S0333_I00000061": 220.0, "V03_S0333_I00000063": 106.0,
    "V03_S0333_I00000065": 414.0, "V03_S0340_I00000392": 172.0, "V03_S0368_I00000128": 900.0,
    "V03_S0368_I00000130": 532.0, "V03_S0368_I00000132": 632.0, "V03_S0368_I00000461": 444.0,
    "V03_S0368_I00000462": 448.0, "V03_S0370_I00000170": 282.0, "V03_S0370_I00000175": 160.0,
    "V03_S0370_I00000178": 237.95, "V03_S0379_I00000371": 134.0, "V03_S0379_I00000373": 254.0,
    "V03_S0379_I00000375": 266.0, "V03_S0379_I00000504": 106.0, "V03_S0386_I00000055": 416.0,
    "V03_S0386_I00000057": 240.0, "V03_S0386_I00000060": 210.0, "V03_S0386_I00000063": 260.0,
    "V03_S0386_I00000067": 200.0, "V03_S0389_I00000059": 256.0, "V03_S0389_I00000061": 238.0,
    "V03_S0389_I00000063": 348.0, "V03_S0391_I00000105": 402.0, "V03_S0391_I00000110": 390.0,
    "V03_S0391_I00000111": 570.0, "V03_S0408_I00000025": 206.0, "V03_S0408_I00000026": 298.0,
    "V03_S0408_I00000031": 156.0, "V03_S0412_I00000411": 220.0, "V03_S0425_I00000054": 148.0,
    "V03_S0434_I00000030": 272.0, "V03_S0434_I00000039": 295.11, "V03_S0437_I00000026": 292.0,
    "V03_S0441_I00000195": 168.0, "V03_S0442_I00000026": 262.0, "V03_S0442_I00000032": 216.0,
    "V03_S0442_I00000036": 190.0, "V03_S0446_I00000025": 136.0, "V03_S0446_I00000028": 230.0,
    "V03_S0458_I00000072": 176.0, "V03_S0458_I00000073": 210.0, "V03_S0458_I00000077": 350.0,
    "V03_S0458_I00000078": 336.0, "V03_S0459_I00000391": 230.0, "V03_S0459_I00000392": 234.0,
    "V03_S0465_I00000159": 280.0, "V03_S0465_I00000162": 276.0, "V03_S0480_I00000055": 392.0,
    "V03_S0483_I00000025": 192.0, "V03_S0483_I00000026": 234.0, "V03_S0484_I00000060": 200.0,
    "V03_S0484_I00000061": 324.0, "V03_S0503_I00000073": 172.0, "V03_S0503_I00000074": 170.0,
    "V03_S0503_I00000075": 134.0, "V03_S0503_I00000079": 166.0, "V03_S0503_I00000083": 114.0,
    "V03_S0503_I00000085": 170.0, "V03_S0503_I00000087": 220.0, "V03_S0526_I00000026": 394.0,
    "V03_S0526_I00000030": 332.0, "V03_S0526_I00000031": 162.0, "V03_S0526_I00000039": 334.0,
    "V03_S0553_I00000194": 356.0, "V03_S0553_I00000412": 120.0, "V03_S0553_I00000414": 118.0,
    "V03_S0553_I00000420": 160.0, "V03_S0553_I00000425": 226.0, "V03_S0558_I00000315": 170.0,
    "V03_S0558_I00000316": 206.0, "V03_S0558_I00000320": 226.0, "V03_S0558_I00000324": 172.0,
    "V03_S0581_I00000402": 226.0, "V03_S0607_I00000106": 214.0, "V03_S0607_I00000114": 158.0,
    "V03_S0607_I00000123": 154.0, "V03_S0702_I00000205": 256.0, "V03_S0706_I00000391": 290.0,
    "V03_S0706_I00000393": 232.0, "V03_S0706_I00000400": 158.0, "V03_S0706_I00000438": 274.0,
    "V03_S0710_I00000063": 156.0, "V03_S0712_I00000412": 200.0, "V03_S0712_I00000415": 130.0,
    "V03_S0712_I00000418": 88.0, "V03_S0712_I00000420": 436.0, "V03_S0712_I00000425": 191.38,
    "V03_S0717_I00000180": 358.0, "V03_S0717_I00000185": 560.0, "V03_S0717_I00000188": 398.0,
    "V03_S0774_I00000144": 208.0, "V03_S0774_I00000145": 320.0, "V03_S0774_I00000147": 318.0,
    "V03_S0846_I00000070": 166.0, "V03_S0846_I00000285": 146.0, "V03_S0846_I00000297": 236.0,
    "V03_S0846_I00000298": 128.0, "V03_S0846_I00000303": 168.0, "V03_S0865_I00000162": 128.0,
    "V03_S0865_I00000164": 128.0, "V03_S0865_I00000169": 130.0, "V03_S0870_I00000054": 220.0,
    "V03_S0880_I00000014": 138.0, "V03_S0880_I00000153": 246.0, "V03_S0880_I00000156": 52.0,
    "V03_S0888_I00000144": 560.0, "V03_S0888_I00000153": 110.0, "V03_S0926_I00000072": 250.0,
    "V03_S0926_I00000075": 168.0, "V03_S0926_I00000078": 126.0, "V03_S0926_I00000081": 144.0,
    "V03_S0926_I00000088": 274.0, "V03_S0932_I00000162": 184.0, "V03_S0941_I00000179": 230.0,
    "V03_S0941_I00000180": 286.0, "V03_S0941_I00000181": 358.0, "V03_S0941_I00000182": 318.0,
    "V03_S0941_I00000184": 310.0, "V03_S0941_I00000190": 348.0, "V03_S0966_I00000210": 218.0,
    "V03_S1052_I00000319": 288.0, "V03_S1052_I00000323": 268.0, "V03_S1052_I00000324": 212.0,
    "V03_S1052_I00000327": 246.0, "V03_S1062_I00000057": 480.0, "V03_S1128_I00000087": 192.0,
    "V03_S1131_I00000026": 286.0, "V03_S1131_I00000027": 432.0, "V03_S1131_I00000028": 612.0,
    "V03_S1131_I00000031": 456.0, "V03_S1132_I00000105": 336.0, "V03_S1132_I00000110": 406.0,
    "V03_S1135_I00000055": 216.0, "V03_S1139_I00000026": 146.0, "V03_S1139_I00000027": 206.0,
    "V03_S1139_I00000032": 266.0, "V03_S1147_I00000105": 222.0, "V03_S1147_I00000110": 280.0,
    "V03_S1153_I00000026": 250.0, "V03_S1153_I00000027": 380.0, "V03_S1153_I00000028": 556.0,
    "V03_S1153_I00000031": 666.0, "V03_S1153_I00000032": 462.0, "V03_S1156_I00000027": 642.0,
    "V03_S1159_I00000401": 374.0, "V03_S1159_I00000402": 330.0, "V03_S1176_I00000232": 124.0,
    "V03_S1176_I00000388": 200.0, "V03_S1176_I00000389": 312.0, "V03_S1176_I00000390": 178.0,
    "V03_S1176_I00000391": 314.0, "V03_S1176_I00000397": 252.0, "V03_S1176_I00000398": 126.0,
    "V03_S1176_I00000438": 140.0, "V03_S1209_I00000054": 232.0, "V03_S1209_I00000055": 154.0,
    "V03_S1209_I00000070": 226.0, "V03_S1209_I00000453": 105.98, "V03_S1220_I00000402": 538.0,
    "V03_S1220_I00000403": 402.0, "V03_S1246_I00000056": 302.0, "V03_S1253_I00000314": 242.0,
    "V03_S1253_I00000315": 212.0, "V03_S1253_I00000316": 216.0, "V03_S1253_I00000319": 196.0,
    "V03_S1253_I00000320": 214.0, "V03_S1253_I00000321": 186.0, "V03_S1253_I00000323": 188.0,
    "V03_S1253_I00000325": 216.0, "V03_S1270_I00000059": 170.0, "V03_S1270_I00000068": 220.0,
    "V03_S1279_I00000111": 204.0, "V03_S1279_I00000112": 192.0, "V03_S1279_I00000120": 226.0,
    "V03_S1279_I00000121": 210.0, "V03_S1309_I00000413": 264.0, "V03_S1309_I00000417": 376.0,
    "V03_S1309_I00000419": 290.0, "V03_S1312_I00000159": 364.0, "V03_S1312_I00000166": 512.0,
    "V03_S1313_I00000105": 356.0, "V03_S1313_I00000106": 206.0, "V03_S1313_I00000107": 262.0,
    "V03_S1313_I00000110": 218.0, "V03_S1313_I00000111": 412.0, "V03_S1313_I00000112": 362.0,
    "V03_S1317_I00000194": 194.0, "V03_S1317_I00000415": 236.0, "V03_S1317_I00000420": 154.0,
    "V03_S1322_I00000110": 192.0, "V03_S1322_I00000119": 272.0, "V03_S1322_I00000121": 128.0,
    "V03_S1331_I00000107": 192.0, "V03_S1331_I00000108": 172.0, "V03_S1331_I00000115": 220.0,
    "V03_S1331_I00000118": 210.0, "V03_S1331_I00000121": 94.0, "V03_S1417_I00000320": 382.0,
    "V03_S1631_I00000212": 154.0, "V03_S1825_I00000001": 156.0, "V03_S1848_I00000179": 152.0,
    "V03_S1848_I00000185": 320.0, "V03_S1848_I00000189": 310.0, "V03_S1848_I00000190": 316.0,
    "V03_S1869_I00000303": 210.0, "V03_S1930_I00000105": 256.0, "V03_S2044_I00000334": 406.0,
    "V03_S2044_I00000337": 286.0, "V03_S2044_I00000338": 482.0, "V03_S2044_I00000340": 371.18,
    "V03_S2083_I00000210": 198.0, "V03_S2083_I00000213": 226.0, "V03_S2083_I00000216": 210.0,
    "V03_S2132_I00000216": 142.0,
}

# =============================================================================
# IO paths + loaders
# =============================================================================

DEFAULT_MANIFEST_PATH = str(
    PROJECT_ROOT / "manifests" / "manifest.csv"
)
DEFAULT_OUTPUT_ROOT = str(
    PROJECT_ROOT / "model_input"
)
DEFAULT_VAD_DIR = str(
    PROJECT_ROOT / "subset" / "vad"
)
DEFAULT_TRANSCRIPT_DIR = str(
    PROJECT_ROOT / "subset" / "transcript"
)

def load_manifest(path: str) -> list[dict]:
    """Load manifest.csv into a list of dict rows and validate columns.
    Fails fast on missing file or missing required columns — schema drift
    must surface immediately."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found at {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "split", "interaction_id",
            "file_id_a", "file_id_b",
            "participant_a", "participant_b",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest {path} has no rows")
    return rows

def load_vad_jsonl(path: str) -> list[dict]:
    """Load a Silero VAD JSONL → list of {'start': float, 'end': float}.
    Missing/empty file is fatal — VAD is mandatory per our download-script
    strict-vad policy."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"VAD file missing or empty: {path}")
    segments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segments.append(json.loads(line))
    # Sort by start defensively — Silero output is usually ordered, but
    # enforce it here so downstream binary-search assumptions hold.
    segments.sort(key=lambda s: s["start"])
    return segments

def load_transcript_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    segments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                segments.append(json.loads(line))
            except json.JSONDecodeError:
                # Defensive — skip malformed lines rather than failing.
                continue
    return segments

def extract_words_from_transcript(segments: list[dict]) -> list[dict]:
    """Flatten a list of WhisperX segment dicts into a flat word list.
    Each word: {'word': str, 'start': float, 'end': float, 'score': float}.
    Words without valid start/end are dropped (WhisperX occasionally emits
    unaligned words at the end of a segment)."""
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            if "start" in w and "end" in w and w["start"] is not None:
                words.append(w)
    words.sort(key=lambda w: w["start"])
    return words

def get_audio_duration(interaction_id: str) -> float:
    if interaction_id not in AUDIO_DURATIONS_S:
        raise KeyError(
            f"interaction {interaction_id!r} not in AUDIO_DURATIONS_S; "
            f"the dict is stale relative to manifest.csv. Regenerate it "
            f"per the recipe near the dict definition."
        )
    return AUDIO_DURATIONS_S[interaction_id]

# =============================================================================
# Primitive time helpers
# =============================================================================

def is_voiced_at(t_query: float, vad_segments: list[dict]) -> bool:
    """Return True iff any VAD segment contains t_query (closed interval).
    Linear scan is fine at Silero segment counts (~10^2 per recording)."""
    for seg in vad_segments:
        if seg["start"] <= t_query <= seg["end"]:
            return True
    return False

def current_turn_end(t: float, vad_segments: list[dict]) -> float | None:
    """If t is inside a VAD segment, return that segment's end (seconds).
    Else return None. Used for the A-yield-tolerance check."""
    for seg in vad_segments:
        if seg["start"] <= t <= seg["end"]:
            return seg["end"]
    return None

def clip_segment_to_horizon(
    seg: dict, t: float, tau: float
) -> tuple[float, float] | None:
    """Clip a VAD segment to the horizon [t, t+tau]. Returns (clipped_start,
    clipped_end) in seconds, or None if the segment doesn't intersect
    the horizon."""
    clipped_start = max(seg["start"], t)
    clipped_end = min(seg["end"], t + tau)
    if clipped_start >= clipped_end:
        return None
    return (clipped_start, clipped_end)

# =============================================================================
# Word normalization and BC lexicon lookup
# =============================================================================

_PUNCT_RE = re.compile(r"[^\w\s-]")
_WS_HYPHEN_RE = re.compile(r"[\s\-]+")

def normalize_word(raw: str) -> str:
    """Lowercase → strip leading/trailing punct → collapse internal
    whitespace and hyphens. Result: 'Mm-hmm!' / 'mm hmm' / 'mmhmm' all
    → 'mmhmm'."""
    s = raw.lower().strip()
    s = _PUNCT_RE.sub("", s)       # drop punctuation
    s = _WS_HYPHEN_RE.sub("", s)   # collapse whitespace and hyphens
    return s

def is_bc_token(word: str) -> bool:
    """True iff the normalized word is in BC_LEXICON."""
    return normalize_word(word) in BC_LEXICON

# =============================================================================
# Per-horizon B-utterance construction
# =============================================================================

class BUtterance:
    """One B-VAD segment clipped to a given horizon, with the subset of
    B-transcript words whose start falls inside the clipped interval."""

    __slots__ = ("clipped_start", "clipped_end", "duration_ms",
                 "words", "raw_segment")

    def __init__(self, clipped_start, clipped_end, words, raw_segment):
        self.clipped_start = clipped_start
        self.clipped_end = clipped_end
        self.duration_ms = (clipped_end - clipped_start) * 1000.0
        self.words = words
        self.raw_segment = raw_segment

def build_b_utterances_in_horizon(
    t: float, tau: float,
    b_vad: list[dict],
    b_words: list[dict],
) -> list[BUtterance]:
    """For a given sample point t and horizon length tau, build the list
    of B's clipped-to-horizon utterances. Returns utterances sorted by
    clipped_start."""
    utterances = []
    for seg in b_vad:
        clipped = clip_segment_to_horizon(seg, t, tau)
        if clipped is None:
            continue
        c_start, c_end = clipped
        # Attach words whose start falls in the clipped interval. Using
        # word-start as the anchor (word-end can extend past τ without
        # disqualification — transcription boundaries are noisier than VAD).
        u_words = [
            w for w in b_words
            if c_start <= w["start"] <= c_end
        ]
        utterances.append(BUtterance(c_start, c_end, u_words, seg))
    utterances.sort(key=lambda u: u.clipped_start)
    return utterances

# =============================================================================
# Substantive / BC-qualifying tests
# =============================================================================

def is_substantive(u: BUtterance,
                   substantive_duration_ms: float,
                   substantive_word_count: int) -> bool:
    if u.duration_ms >= substantive_duration_ms:
        return True
    if len(u.words) >= substantive_word_count:
        return True
    if u.words and any(not is_bc_token(w["word"]) for w in u.words):
        return True
    return False

def is_bc_qualifying(u: BUtterance,
                     bc_max_duration_ms: float,
                     bc_max_word_count: int) -> bool:
    """BACKCHANNEL qualification.
        Duration ≤ BC_MAX_DURATION_MS
        AND (word-count ≤ BC_MAX_WORD_COUNT OR all words in BC_LEXICON)
    """
    if u.duration_ms > bc_max_duration_ms:
        return False
    if len(u.words) <= bc_max_word_count:
        return True
    # ≥ 3 words case: require every word to be BC-lexicon.
    return all(is_bc_token(w["word"]) for w in u.words)

# =============================================================================
# Per-(t, τ) classification
# =============================================================================

def classify_sample(
    t: float,
    tau: float,
    a_vad: list[dict],
    b_vad: list[dict],
    b_words: list[dict],
    *,
    substantive_duration_ms: float = SUBSTANTIVE_DURATION_MS,
    substantive_word_count: int = SUBSTANTIVE_WORD_COUNT,
    bc_max_duration_ms: float = BC_MAX_DURATION_MS,
    bc_max_word_count: int = BC_MAX_WORD_COUNT,
    a_yield_tolerance_s: float = A_YIELD_TOLERANCE_MS / 1000.0,
) -> int:
    """Implement the precedence-ordered decision tree. Returns one of the
    6 class integers. Prerequisite (A voiced at t) is the caller's
    responsibility — this function assumes it."""
    # Step 1: build B's clipped utterances in horizon.
    b_utts = build_b_utterances_in_horizon(t, tau, b_vad, b_words)

    # Step 2: find first substantive-B utterance (precedence: earliest start).
    first_sub = next(
        (u for u in b_utts
         if is_substantive(u, substantive_duration_ms, substantive_word_count)),
        None,
    )

    # Step 3: substantive-B branch — YIELD / INTERRUPT / FAILED.
    if first_sub is not None:
        a_turn_end = current_turn_end(t, a_vad)
        assert a_turn_end is not None, \
            "classify_sample called when A not voiced at t (prerequisite)"

        if a_turn_end <= first_sub.clipped_start + a_yield_tolerance_s:
            return CLASS_YIELD

        # A keeps going past tolerance → floor contest.
        # Distinguish INTERRUPT vs FAILED using B's state at horizon end.
        b_voiced_at_tau = is_voiced_at(t + tau, b_vad)
        a_voiced_at_tau = is_voiced_at(t + tau, a_vad)
        if b_voiced_at_tau:
            return CLASS_INTERRUPT
        if a_voiced_at_tau:
            return CLASS_FAILED
        # Both silent at horizon end despite prior contest — degenerate.
        # Treat as INTERRUPT (unresolved state, not a clean FAILED).
        return CLASS_INTERRUPT

    # Step 4: no substantive-B. BACKCHANNEL if qualifying AND A still voiced.
    a_voiced_at_tau = is_voiced_at(t + tau, a_vad)
    if a_voiced_at_tau and any(
        is_bc_qualifying(u, bc_max_duration_ms, bc_max_word_count)
        for u in b_utts
    ):
        return CLASS_BACKCHANNEL

    # Step 5: HOLD if A still voiced.
    if a_voiced_at_tau:
        return CLASS_HOLD

    # Step 6: fall-through. A stopped, no substantive-B. LAPSE.
    return CLASS_LAPSE

# =============================================================================
# Per-interaction, per-participant pipeline
# =============================================================================

class SampleOutput:
    """One emitted sample. Labels are accumulated in memory and written out
    per-τ at the end of the run."""

    __slots__ = ("basename", "split", "file_id", "t_ms", "labels_by_tau")

    def __init__(self, basename, split, file_id, t_ms):
        self.basename = basename
        self.split = split
        self.file_id = file_id
        self.t_ms = t_ms
        # {tau_ms: class_int} for each tau in TAU_GRID_MS
        self.labels_by_tau: dict[int, int] = {}

def enumerate_sample_times(
    a_vad: list[dict],
    file_duration: float,
    stride_ms: int,
    tau_grid_ms: tuple,
) -> list[float]:
    t_min = WINDOW_S
    t_max = file_duration - max(tau_grid_ms) / 1000.0
    if t_max <= t_min:
        return []
    step = stride_ms / 1000.0
    candidates = []
    t = t_min
    while t <= t_max:
        if is_voiced_at(t, a_vad):
            candidates.append(t)
        t += step
    return candidates

def process_one_participant(
    *,
    split: str,
    speaker_file_id: str,
    speaker_vad: list[dict],
    listener_vad: list[dict],
    listener_words: list[dict],
    file_duration: float,
    stride_ms: int,
    tau_grid_ms: tuple,
    substantive_duration_ms: float,
    substantive_word_count: int,
    bc_max_duration_ms: float,
    bc_max_word_count: int,
    a_yield_tolerance_s: float,
) -> list[SampleOutput]:
    sample_times = enumerate_sample_times(
        a_vad=speaker_vad,
        file_duration=file_duration,
        stride_ms=stride_ms,
        tau_grid_ms=tau_grid_ms,
    )

    outputs: list[SampleOutput] = []
    for t in sample_times:
        t_ms = int(round(t * 1000))
        basename = f"{speaker_file_id}_t_{t_ms:07d}"

        so = SampleOutput(
            basename=basename, split=split,
            file_id=speaker_file_id, t_ms=t_ms,
        )
        for tau_ms in tau_grid_ms:
            tau = tau_ms / 1000.0
            cls = classify_sample(
                t=t, tau=tau,
                a_vad=speaker_vad, b_vad=listener_vad,
                b_words=listener_words,
                substantive_duration_ms=substantive_duration_ms,
                substantive_word_count=substantive_word_count,
                bc_max_duration_ms=bc_max_duration_ms,
                bc_max_word_count=bc_max_word_count,
                a_yield_tolerance_s=a_yield_tolerance_s,
            )
            so.labels_by_tau[tau_ms] = cls
        outputs.append(so)

    return outputs

def process_interaction(
    row: dict,
    *,
    vad_dir: str,
    transcript_dir: str,
    stride_ms: int,
    tau_grid_ms: tuple,
    substantive_duration_ms: float,
    substantive_word_count: int,
    bc_max_duration_ms: float,
    bc_max_word_count: int,
    a_yield_tolerance_s: float,
) -> dict:
    """Process one manifest row — both perspective participants. Returns a
    per-dyad result dict for the global manifest."""
    dyad_id = row["interaction_id"]
    split = row["split"]
    file_id_a = row["file_id_a"]
    file_id_b = row["file_id_b"]

    # VAD + transcript for both participants.
    vad_a = load_vad_jsonl(os.path.join(vad_dir, f"{file_id_a}.jsonl"))
    vad_b = load_vad_jsonl(os.path.join(vad_dir, f"{file_id_b}.jsonl"))

    words_a = extract_words_from_transcript(
        load_transcript_jsonl(os.path.join(transcript_dir, f"{file_id_a}.jsonl"))
    )
    words_b = extract_words_from_transcript(
        load_transcript_jsonl(os.path.join(transcript_dir, f"{file_id_b}.jsonl"))
    )

    # File duration: dict lookup against the inlined AUDIO_DURATIONS_S
    # table. Both participants in a Seamless dyad share an identical wav
    # duration (verified across the active manifest), so one lookup
    # serves both.
    file_duration = get_audio_duration(dyad_id)

    transcript_present_a = len(words_a) > 0
    transcript_present_b = len(words_b) > 0

    all_outputs: list[SampleOutput] = []

    # A-as-speaker (B is listener).
    all_outputs += process_one_participant(
        split=split,
        speaker_file_id=file_id_a,
        speaker_vad=vad_a, listener_vad=vad_b, listener_words=words_b,
        file_duration=file_duration,
        stride_ms=stride_ms, tau_grid_ms=tau_grid_ms,
        substantive_duration_ms=substantive_duration_ms,
        substantive_word_count=substantive_word_count,
        bc_max_duration_ms=bc_max_duration_ms,
        bc_max_word_count=bc_max_word_count,
        a_yield_tolerance_s=a_yield_tolerance_s,
    )

    # B-as-speaker (A is listener).
    all_outputs += process_one_participant(
        split=split,
        speaker_file_id=file_id_b,
        speaker_vad=vad_b, listener_vad=vad_a, listener_words=words_a,
        file_duration=file_duration,
        stride_ms=stride_ms, tau_grid_ms=tau_grid_ms,
        substantive_duration_ms=substantive_duration_ms,
        substantive_word_count=substantive_word_count,
        bc_max_duration_ms=bc_max_duration_ms,
        bc_max_word_count=bc_max_word_count,
        a_yield_tolerance_s=a_yield_tolerance_s,
    )

    return {
        "dyad_id": dyad_id,
        "split": split,
        "n_samples": len(all_outputs),
        "transcript_present_a": transcript_present_a,
        "transcript_present_b": transcript_present_b,
        "file_duration_s": file_duration,
        "outputs": all_outputs,
    }

# =============================================================================
# Orchestration
# =============================================================================

def build_per_tau_label_files(
    results: list[dict],
    output_root: str,
    tau_grid_ms: tuple,
    dry_run: bool,
) -> dict:
    """Write labels/labels_tau_XXXX.json files, one per τ. All splits are
    emitted at natural rate — class imbalance is handled at training time
    via inverse-frequency weights in the loss. Returns per-(tau, split)
    class-count dicts for the manifest."""
    labels_dir = os.path.join(output_root, "labels")
    if not dry_run:
        os.makedirs(labels_dir, exist_ok=True)

    # Collect all outputs grouped by split.
    all_outputs_by_split: dict[str, list[SampleOutput]] = defaultdict(list)
    for r in results:
        for o in r["outputs"]:
            all_outputs_by_split[o.split].append(o)

    counts_summary: dict[int, dict[str, dict[str, int]]] = {}
    for tau_ms in tau_grid_ms:
        counts_summary[tau_ms] = {}
        labels_for_tau: dict[str, int] = {}

        for split, outputs in all_outputs_by_split.items():
            counts = Counter(o.labels_by_tau[tau_ms] for o in outputs)
            counts_dict = {CLASS_NAME[k]: v for k, v in counts.items()}

            # Every sample flows through to labels.json; no subsampling.
            # EXCLUSION classes (INTERRUPT / FAILED / LAPSE) are included
            # here and filtered out by the loader at training time.
            for o in outputs:
                labels_for_tau[o.basename] = o.labels_by_tau[tau_ms]

            counts_summary[tau_ms][split] = {
                "counts": counts_dict,
                "n_samples": len(outputs),
            }

        if not dry_run:
            path = os.path.join(labels_dir, f"labels_tau_{tau_ms:04d}.json")
            with open(path, "w") as f:
                json.dump(labels_for_tau, f, indent=2, sort_keys=True)

    return counts_summary

def write_manifest(
    results: list[dict],
    counts_summary: dict,
    output_root: str,
    config: dict,
    dry_run: bool,
) -> None:
    """Write manifest.json capturing full run provenance — config + per-
    (tau, split) natural-rate class counts + per-dyad flags."""
    if dry_run:
        return
    manifest = {
        "config": config,
        "classes": {
            str(k): CLASS_NAME[k] for k in sorted(CLASS_NAME.keys())
        },
        "training_classes": sorted(TRAINING_CLASSES),
        "exclusion_classes": sorted(EXCLUSION_CLASSES),
        "bc_lexicon": sorted(BC_LEXICON),
        "per_tau_per_split_counts": {
            str(tau): v for tau, v in counts_summary.items()
        },
        "per_dyad": [
            {
                "dyad_id": r["dyad_id"],
                "split": r["split"],
                "n_samples": r["n_samples"],
                "transcript_present_a": r["transcript_present_a"],
                "transcript_present_b": r["transcript_present_b"],
                "file_duration_s": r["file_duration_s"],
            }
            for r in results
        ],
    }
    path = os.path.join(output_root, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-τ ground-truth labels for the 3-class turn-taking "
            "classifier. Emits labels JSONs only — feature slicing is the "
            "job of per-modality extraction scripts. See module docstring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O paths.
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH,
                        help="Path to manifest.csv. Default: %(default)s")
    parser.add_argument("--vad-dir", default=DEFAULT_VAD_DIR,
                        help="Per-participant Silero VAD JSONLs. "
                             "Default: %(default)s")
    parser.add_argument("--transcript-dir", default=DEFAULT_TRANSCRIPT_DIR,
                        help="Per-participant WhisperX transcript JSONLs. "
                             "Default: %(default)s")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Parent of the emitted labels/ subdir and "
                             "manifest.json. Default: %(default)s")

    # Threshold overrides (all default to the CAPS constants above).
    parser.add_argument("--stride-ms", type=int, default=STRIDE_MS)
    parser.add_argument("--substantive-duration-ms", type=int,
                        default=SUBSTANTIVE_DURATION_MS)
    parser.add_argument("--substantive-word-count", type=int,
                        default=SUBSTANTIVE_WORD_COUNT)
    parser.add_argument("--bc-max-duration-ms", type=int,
                        default=BC_MAX_DURATION_MS)
    parser.add_argument("--bc-max-word-count", type=int,
                        default=BC_MAX_WORD_COUNT)
    parser.add_argument("--a-yield-tolerance-ms", type=int,
                        default=A_YIELD_TOLERANCE_MS)
    parser.add_argument("--tau-grid-ms", type=str, default=None,
                        help="Comma-separated override, e.g. "
                             "'100,200,400,800,1600'. Default: %(default)s")

    # Runtime plumbing.
    parser.add_argument("--filter-split", choices=["train", "val", "test"],
                        default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute labels + counts but don't write files.")

    args = parser.parse_args()

    # Parse τ-grid override.
    if args.tau_grid_ms is not None:
        try:
            tau_grid = tuple(int(x) for x in args.tau_grid_ms.split(","))
        except ValueError:
            print(f"ERROR: bad --tau-grid-ms: {args.tau_grid_ms}",
                  file=sys.stderr)
            return 1
    else:
        tau_grid = TAU_GRID_MS

    # Load manifest.
    try:
        rows = load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.filter_split:
        rows = [r for r in rows if r["split"] == args.filter_split]
    if not rows:
        print("No rows to process.", file=sys.stderr)
        return 0

    print(f"Building labels for {len(rows)} dyad(s)")
    print(f"  output_root:              {args.output_root}")
    print(f"  tau_grid_ms:              {tau_grid}")
    print(f"  stride_ms:                {args.stride_ms}")
    print(f"  substantive_duration_ms:  {args.substantive_duration_ms}")
    print(f"  substantive_word_count:   {args.substantive_word_count}")
    print(f"  bc_max_duration_ms:       {args.bc_max_duration_ms}")
    print(f"  bc_max_word_count:        {args.bc_max_word_count}")
    print(f"  a_yield_tolerance_ms:     {args.a_yield_tolerance_ms}")
    print(f"  class balance:            natural-rate (no subsampling); "
          f"use class weights in loss")
    print(f"  dry_run:                  {args.dry_run}")

    a_yield_tolerance_s = args.a_yield_tolerance_ms / 1000.0

    # Ensure the output root exists up front — creating it here gives
    # --dry-run a visible skeleton and prevents an all-dyads-fail first
    # run from leaving write_manifest with no parent directory.
    if not args.dry_run:
        os.makedirs(args.output_root, exist_ok=True)

    start = time.time()
    results = []
    failed = 0
    for i, row in enumerate(rows, 1):
        dyad = row["interaction_id"]
        try:
            result = process_interaction(
                row=row,
                vad_dir=args.vad_dir,
                transcript_dir=args.transcript_dir,
                stride_ms=args.stride_ms,
                tau_grid_ms=tau_grid,
                substantive_duration_ms=args.substantive_duration_ms,
                substantive_word_count=args.substantive_word_count,
                bc_max_duration_ms=args.bc_max_duration_ms,
                bc_max_word_count=args.bc_max_word_count,
                a_yield_tolerance_s=a_yield_tolerance_s,
            )
            results.append(result)
            print(f"  [{i:3d}/{len(rows)}] {row['split']:<5} {dyad:<30} "
                  f"{result['n_samples']:>5} samples")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [{i:3d}/{len(rows)}] {row['split']:<5} {dyad:<30} "
                  f"FAILED: {e}", file=sys.stderr)

    # Build per-τ labels and manifest.
    if results:
        counts_summary = build_per_tau_label_files(
            results=results,
            output_root=args.output_root,
            tau_grid_ms=tau_grid,
            dry_run=args.dry_run,
        )

        config = {
            "stride_ms": args.stride_ms,
            "window_s": WINDOW_S,
            "train_tau_ms": TRAIN_TAU_MS,
            "tau_grid_ms": list(tau_grid),
            "substantive_duration_ms": args.substantive_duration_ms,
            "substantive_word_count": args.substantive_word_count,
            "bc_max_duration_ms": args.bc_max_duration_ms,
            "bc_max_word_count": args.bc_max_word_count,
            "a_yield_tolerance_ms": args.a_yield_tolerance_ms,
            "class_balance_strategy": "natural-rate; class weights at train time",
        }
        write_manifest(
            results=results,
            counts_summary=counts_summary,
            output_root=args.output_root,
            config=config,
            dry_run=args.dry_run,
        )

        # Print a summary table per τ for quick sanity.
        print("\n" + "=" * 78)
        print("PER-τ PER-SPLIT CLASS COUNTS (natural rate)")
        print("=" * 78)
        for tau_ms in tau_grid:
            print(f"\nτ = {tau_ms} ms:")
            for split in sorted(counts_summary[tau_ms].keys()):
                c = counts_summary[tau_ms][split]
                print(f"  {split:<6} n_samples={c['n_samples']:>5}")
                print(f"         counts: {c['counts']}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. {failed}/{len(rows)} dyad(s) failed.")
    return 2 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
