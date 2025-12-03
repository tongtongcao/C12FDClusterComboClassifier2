The package is to train classifier to predict if a 6-cluster combo belongs to a track candidate by Multi-Layer Perceptron (MLP).\
The AI model is designed by pytorch and pytorch-lightning.

It is recommended to run the package in python virtual enviroment.
  - To install python virtual enviromen for cup-only, source setup_pytorch_lightning_venv.csh
  - To install python virtual enviromen for GPU, cuda needs to be installed. In ifarm, cuda 12.4 has been installed. Change "set use_cuda = 0" into "set use_cuda = 1", then source setup_pytorch_lightning_venv.csh. Python virtual enviromen for GPU will be installed.

Then, enter python virtual enviroment,
  - source venvs/torch-cpu/bin/activate.csh for cpu-only
  - source venvs/torch-cu124/bin/activate.csh for GPU at ifarm
  
Finally, run the package,
- python3 train.py ...

Arguments:
  - positional arguments:
    - inputs      &nbsp;&nbsp;&nbsp;          One or more CSV files (default: avgWiresSlopesLabel.csv)

  - options:
    - -h, --help     &nbsp;&nbsp;&nbsp;       show this help message and exit
    - --device {cpu,gpu,auto} &nbsp;&nbsp;&nbsp; Choose device: cpu, gpu, or auto (default: auto)
    - --max_epochs MAX_EPOCHS &nbsp;&nbsp;&nbsp; Number of training epochs
    - --batch_size BATCH_SIZE &nbsp;&nbsp;&nbsp; Batch size for DataLoader
    - --outdir OUTDIR   &nbsp;&nbsp;&nbsp;    Directory to save models and plots
    - --end_name END_NAME &nbsp;&nbsp;&nbsp;  Optional suffix to append to output files (default: none)
    - --hidden_dim HIDDEN_DIM   &nbsp;&nbsp;&nbsp;      Number of neurons in hidden layers (default: 16)
    - --num_layers NUM_LAYERS &nbsp;&nbsp;&nbsp; Number of hidden layers (default: 2)
    - --lr LR        &nbsp;&nbsp;&nbsp;       Learning rate for optimizer (default: 1e-3)
    - --dropout DROPOUT &nbsp;&nbsp;&nbsp; Dropout rate for the model (default: 0.2)
    - --no_train      &nbsp;&nbsp;&nbsp;      Skip training and only run inference using a saved model
    - --enable_progress_bar &nbsp;&nbsp;&nbsp; Enable progress bar during training (default: disabled)

For the estimator, inputs are average wires and slopes of 6 clusters, where average wires are divided by 112 for normalization.
     
and output is probability.

To apply the estimator in coatjava, an example for application of the estimator by ai.djl is developed.
- Install: mvn clean install
- Run: java -cp "target/TestDJL-1.0-SNAPSHOT.jar:target/lib/*" org.example.Main
