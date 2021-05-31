# D3Net (Music Source Separation)

This is inference code for D3Net based music source separation.
- Example
 ```
 # Assuming test.wav file in the current directory.
 python ./separate.py -i ./test.wav -o output/
```
- Options  
-i : Input files (Multiple wave files can be specified.)  
-o : Output directory. (Output folder path to save separated instruments)  
-m : Model file. (Pre-trained model)  
-c : Context. (Extension modules : `cpu` or `cudnn`)
