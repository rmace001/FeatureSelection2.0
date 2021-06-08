# FeatureSelection2.0
UCR CS205 Artificial Intelligence

## Usage
`$: python3 FeatureSelect2.0.py --filesize small --mode 1 --debug`
### Arguments
    --filesize: ['small' | 'large' | 'None']
    --mode: [1 | 2 | 3]
    [--debug]: stores True if present

## Bonus Dataset
To perform forward or backward selection on this dataset, toggle-comment line 168 and line 169.

In addition, to select between multi-class classification and binary classification, toggle-comment the desired 
mapping on lines 162 and 163.

Then simply run: 

`$: python3 FeatureSelect2.0.py --filesize None --mode 3`

