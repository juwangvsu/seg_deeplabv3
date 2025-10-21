ws3: deeplabseg:

---------------10/21/25 ---------------------------
pick one output token mlp layer. original paper use the token for cls. it actually does not matter.
I pick the last output token. still trainable.
 
python3 vit_test.py

EPOCH 32
Training predictions:    [7 2 1 7 4 6 9 1 6 5 1 6 2 6 0 3 6 7 7 5 5 3 9 8 6 8 1 5 0 6]
Training labels:         [2 2 1 7 4 6 9 1 6 5 1 6 0 6 0 3 4 7 7 5 5 3 9 8 6 8 1 5 0 6]
Validation predictions:  [1 7 1 3 9 3 8 7 2 7 0 4 1 4 1 9 8 3 0 0 2 4 9 1 0 2 3 5 7 7]
Validation labels:       [1 7 1 3 9 3 8 7 2 7 0 4 1 4 1 9 8 3 0 0 2 4 9 1 0 2 3 5 7 7]
------------------------------
Train Loss: 0.2648
Valid Loss: 0.2035
Train Accuracy: 0.9256

