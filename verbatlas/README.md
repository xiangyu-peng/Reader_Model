# Semantic Role Labeling Using VerbAtlas


In this work, we took a pretrained model on Propbank and mapped the propbank ids to the VerbAtlas Ids, 
then we changed the roles from Propank to Verbatlas.   

## Resources
VerbAtlas: a Novel Large-Scale Verbal Semantic Resource and Its Application to Semantic Role Labeling
link: https://www.aclweb.org/anthology/D19-1058/

Transformer-srl: a fine-tuned BERT SRL model on propbank. 
https://github.com/Riccorl/transformer-srl

## Installation 
1. In your terminal cd to the folder containing the srl-requirements.txt.
2. Create a python >= 3.6 environment (I used python 3.8).
2. Run the following command to install the dependencies: 
`pip install -r srl-requirements.txt`
   
## Usage 

1.  Modify the sentence in line 140 to your desired sentence [TODO: To what should change this?]
2.  In your terminal run the following line
````python verbatlas_srl.py````
3. You don't need to modify the parameters to run the model, the default parameters have relative paths which should work without any modifications. 
   Alternatively, if you want to use other models or verb sources, You can pass your arguments to the file during runtime by utilizing the following paramters: 
    * model_path: The pretrained SRL  model path.
    * verb_source: verbatlas or verbnet (experimental).
    * pb_va: PropBank VerbAtlas Mapping file path.
    * pb_vn: PropBank VerbNet Mapping file path.
    * va_fi: VerbAtlass class names to ids file path.
   

Happy coding! :)