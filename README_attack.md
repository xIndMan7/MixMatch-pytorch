## ReadMe for Attack
This is the ReadMe for the modified Mixmatch to do a model extraction attack on model zoo.

Below is the full list of the arguments, however we do not need this many:
```train.py [-h] [--direct | --no-direct] [--zoo | --no-zoo] [--clockwork | --no-clockwork] [--server | --no-server]
                [--zoo_victim which model in zoo] [--fingerprinting | --no-fingerprinting] [--epochs N] [--start-epoch N]
                [--batch-size N] [--lr LR] [--om oracle_model] [--om_r resnet type] [--am_r resnet type] [--resume PATH]
                [--manualSeed MANUALSEED] [--gpu GPU] [--n-labeled N_LABELED] [--train-iteration TRAIN_ITERATION]
                [--out OUT] [--alpha ALPHA] [--lambda-u LAMBDA_U] [--T T] [--ema-decay EMA_DECAY]```

### To Run The Server with Model Zoo:
```train.py --zoo --server --zoo_victim [which model in the list] --am_r [which resnet number] --gpu [gpu number] --epochs [N] ```

To load the model zoo currently, you have to manually do it after lines 216 where there is High zoo, mid zoo, spread out zoo. Since there is the torchvision arch, mixmatch arch, custom arch, currently not generalized. 

```--direct``` or ```--no-direct``` flag tells whether to target or not

--fingerprinting or --no-fingerprinting tells whether to fingerprint or not

Can run without the server if we remove the server flag.


### To Run Clockwork with model zoo
```train.py --zoo --clockwork --zoo_victim [which model in the list] --am_r [which resnet number] --gpu [gpu number] --epochs [N] ```

To specify the names of the models in clockwork, edit line 199 and manually edit the list. 
Line 145, 146 stores the latency and accuracy of the list.
Working on making this so it reads from a text file.

```--direct``` or ```--no-direct``` flag tells whether to target or not

```--fingerprinting``` or ```--no-fingerprinting``` tells whether to fingerprint or not