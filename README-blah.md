# Just code.... to be removed soon

1. Download onnx-community/Llama-3.2-1B-Instruct-ONNX
2. Set up a virtual environment and install dependencies for python 3.10
3. Run model-qa.py
```
python ./model-qa.py  -m ./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 -k 40 -p 0.95 -t 0.8 -r 1.0
```
4. Build model-qa.cpp
```
cmake -B build -H.
cmake --build build
```
5. Run model-qa
```
./build/model-qa -m ./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 -k 40 -p 0.95 -t 0.8 -r 1.0
```


# Notes
model-qa.py works fine.
model-qa.cpp causes an Segmentation fault.

