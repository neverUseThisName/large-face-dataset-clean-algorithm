# large-face-dataset-clean-algorithm
Python scripts for cleanning large scale face dataset. This algorithm cleans only inner class imposter faces. 
## How it works
For each folder, extract 512-D face features (embeddings) for each image under that folder using pretrained arcface. Compute pair-wise cosine similarities. Define a threshold t. Construct a graph such that each edge connects a pair of images whose similarity exceeds t. Finally, find the maximal connected component as the set of genuine faces.
## How to use
`python3.6 clean_by_connected_cpnt.py path/to/your/dataset threshold`

Your dataset structure should look like `path/to/your/dataset/class0/img0.jpg ... path/to/your/dataset/class1000/img0.jpg`.
`threshold` is in (0, 1). I personally used 0.7 for a satisfactory result.

Imposter faces will be deleted.
