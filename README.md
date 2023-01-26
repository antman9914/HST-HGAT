# HST-HGAT

To run the HST-HGAT code, you need to run `data_preprocess.py` first, and meanwhile change the neighbor limitation, dataset and running mode in the parameter list of `precompute_egonet(dataset, n_layer, max_neighbor, mode)` if necessary. The data preprocessing step will take several hours or even longer.

After data preprocessing, you can train the model with `run.sh`. For the needed experiments, you only need to change dataset, max_len and checkpoint directory. The explanations of needed hyper-parameters are listed in `train.sh`. 
