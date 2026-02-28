import sys
import numpy as np

try:
    print("Importing app and loading models...")
    from app import haze_to_clean, clean_to_haze_to_clean
    dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    print("Testing haze_to_clean...")
    out1, out2, out3, text = haze_to_clean(dummy_img, dummy_img)
    print("haze_to_clean metrics:\n", text)
    
    print("Testing clean_to_haze_to_clean...")
    hazy, out1, out2, out3, out4, text = clean_to_haze_to_clean(dummy_img, 2.0)
    print("clean_to_haze_to_clean metrics:\n", text)
    
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
