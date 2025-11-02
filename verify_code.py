"""
Quick verification script to check if the code has any obvious issues
"""
import sys

def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    errors = []
    
    try:
        import cv2
        print("✓ opencv-python")
    except ImportError as e:
        errors.append(f"✗ opencv-python: {e}")
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        errors.append(f"✗ numpy: {e}")
    
    try:
        import tqdm
        print("✓ tqdm")
    except ImportError as e:
        errors.append(f"✗ tqdm: {e}")
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        errors.append(f"✗ scikit-learn: {e}")
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        errors.append(f"✗ matplotlib: {e}")
    
    return errors

def check_syntax():
    """Check if Python files have valid syntax"""
    print("\nChecking syntax...")
    import py_compile
    errors = []
    
    files = [
        'reconstruct_video.py',
        'enhanced_reconstructor.py',
        'test_reconstruction.py'
    ]
    
    for file in files:
        try:
            py_compile.compile(file, doraise=True)
            print(f"✓ {file}")
        except py_compile.PyCompileError as e:
            errors.append(f"✗ {file}: {e}")
    
    return errors

def check_class_instantiation():
    """Check if classes can be instantiated"""
    print("\nChecking class instantiation...")
    errors = []
    
    try:
        from reconstruct_video import VideoReconstructor
        # Don't actually instantiate with a file, just check import
        print("✓ VideoReconstructor class")
    except Exception as e:
        errors.append(f"✗ VideoReconstructor: {e}")
    
    try:
        from enhanced_reconstructor import EnhancedVideoReconstructor
        print("✓ EnhancedVideoReconstructor class")
    except Exception as e:
        errors.append(f"✗ EnhancedVideoReconstructor: {e}")
    
    try:
        from test_reconstruction import ReconstructionEvaluator
        print("✓ ReconstructionEvaluator class")
    except Exception as e:
        errors.append(f"✗ ReconstructionEvaluator: {e}")
    
    return errors

def main():
    print("="*60)
    print("CODE VERIFICATION")
    print("="*60)
    
    all_errors = []
    
    all_errors.extend(check_imports())
    all_errors.extend(check_syntax())
    all_errors.extend(check_class_instantiation())
    
    print("\n" + "="*60)
    if all_errors:
        print("ISSUES FOUND:")
        for error in all_errors:
            print(f"  {error}")
        print("="*60)
        return 1
    else:
        print("✓ ALL CHECKS PASSED!")
        print("="*60)
        return 0

if __name__ == '__main__':
    sys.exit(main())
