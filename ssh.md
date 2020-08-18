
# Passwordless setup with ssh keys

# Setup DAS4 
ssh-copy-id bhkam@fs3.das4.tudelft.nl
# Also place your public key on the actual server (run this command on bhkam@fs3.das4.tudelft.nl)
scp authorized_keys team_epoch@10.141.3.188:~/.ssh/

# Setup WS6 robotlab UvA
ssh-copy-id teamepoch@146.50.28.207

# ssh config file


# All changes on ws7 and das4 can be discarded
# Run this on laptop
rsync -avzP --exclude "baseline.tf/" --exclude "preprocessed/"   * teamepoch@146.50.28.207:~/Cornell_birdcall_identification_Epoch
rsync -avzP --exclude "baseline.tf/" --exclude "preprocessed/"   * apollo:~/Cornell_birdcall_identification_Epoch


# CUDA_VISIBLE_DEVICES=0


# Test data evaluation
CUDA_VISIBLE_DEVICES=1 python eval_nn.py models/resnet_run0.val_f1.0.154.h5

# detatch 
Ctrl-B D
# Scroll
Ctrol-B [

# tmux
tmux a

# top
htop
nvtop


# SSH TRICK
enter
~.
 
-L 6006:localhost:6006