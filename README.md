# Attacks on Federated Learning 

## Usage 
Run training, using 

```python
main.py 
```
See training settings in utils/oprions.py


## Data distribution 

Three data distributions are supported: 
 
 1. `iid` - the data is equally distributed among parties.
 2. `non_iid (n_labels)` - each party have the data with only few labels. Two parties can have the same label.
 3. `non_iid_exclusive (n_labels)` - each party have the data with only few labels. Two parties can NOT have the same label.

Additional argument `sub_labels` limits the distribution to a subset of labels.

##Agregation methods

Only Federated Averaging is implemented for now. 

