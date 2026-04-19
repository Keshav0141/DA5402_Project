import requests
import json
import time

def test_colors():
    base_url = "http://localhost:8000/predict"
    
    # Paths to the 4 test images
    images = {
        "glioma": r"C:\Users\kille\Downloads\DA5402_Project\DA5402_Project\CerebroNet\data\raw\Testing\glioma\Te-gl_1.jpg",
        "meningioma": r"C:\Users\kille\Downloads\DA5402_Project\DA5402_Project\CerebroNet\data\raw\Testing\meningioma\Te-me_1.jpg",
        "notumor": r"C:\Users\kille\Downloads\DA5402_Project\DA5402_Project\CerebroNet\data\raw\Testing\notumor\Te-no_1.jpg",
        "pituitary": r"C:\Users\kille\Downloads\DA5402_Project\DA5402_Project\CerebroNet\data\raw\Testing\pituitary\Te-pi_1.jpg"
    }

    # The mapping from frontend/app.py
    hue_map = {
        "glioma":     "155deg",    # → Red
        "meningioma": "180deg",    # → Yellow/Orange
        "notumor":    "240deg",    # → Green
        "pituitary":  "0deg"       # → Blue
    }

    success = True
    print("\nStarting End-to-End Dynamic Color Verification...")
    print("-" * 50)
    for expected_class, img_path in images.items():
        try:
            with open(img_path, 'rb') as f:
                response = requests.post(base_url, files={"file": f})
            
            if response.status_code == 200:
                data = response.json()
                predicted_class = data.get("prediction", "Unknown").lower()
                hue = hue_map.get(predicted_class, "Unknown")
                
                print(f"Testing {expected_class.upper()} image:")
                print(f"  -> Predicted API Class: {predicted_class}")
                print(f"  -> Frontend UI Injection: hue-rotate({hue})")
                
                if predicted_class == expected_class:
                    print("  [PASS] Model predicted correctly and mapped to correct UI color.")
                else:
                    print(f"  [FAIL] Expected {expected_class}, got {predicted_class}")
                    success = False
            else:
                print(f"  [FAIL] API Error: {response.status_code} {response.text}")
                success = False
                
        except Exception as e:
            print(f"  [Exception] testing {expected_class}: {e}")
            success = False
        print("-" * 50)
        time.sleep(0.5)
        
    if success:
        print("\n[ALL 4 CLASSES PASSED E2E UI VERIFICATION]")
    else:
        print("\n[SOME TESTS FAILED]")

if __name__ == "__main__":
    test_colors()
