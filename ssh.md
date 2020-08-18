# Passwordless setup with ssh keys

#### Setup DAS4 

`ssh-copy-id bhkam@fs3.das4.tudelft.nl`

Also place your public key on the actual server (run this command on bhkam@fs3.das4.tudelft.nl)

`scp authorized_keys team_epoch@10.141.3.188:~/.ssh/`

####  Setup WS6 robotlab UvA
ssh-copy-id teamepoch@146.50.28.207

####  Create an ssh config file
Create the file `~/.ssh/config` with the following content.
You can then ssh into the servers using `ssh apollo` and `ssh epoch`.

```
## DAS4 
Host das4          
  Hostname fs3.das4.tudelft.nl    
  User bhkam        

Host apollo    
  ProxyCommand ssh -q das4 nc 10.141.3.188  22
  User team_epoch
  LocalForward 6006 localhost:6006


## Robotlab
Host epoch 
  Hostname 146.50.28.207
  User teamepoch
  LocalForward 6006 localhost:6006
```

# Using the servers
I never write any code on the remote servers, so all changes can be discarded at any time.
To push changes to the remote servers, either rsync can be used to copy all local files to the servers (excluding some directories)
or you can discard everything on the server (`git stash`) and pull changes. 

**Important: Expect changes on the workstation to be overwritten. Develop locally.**

#### Sync with rsync

Run this on your local machine to copy files from your laptop to the servers. Customise as needed, such ash which folders to exclude.

`rsync -avzP --exclude "models/" --exclude "preprocessed/"   * epoch:~/Cornell_birdcall_identification_Epoch`

`rsync -avzP --exclude "models/" --exclude "preprocessed/"   * apollo:~/Cornell_birdcall_identification_Epoch`

# Running training

Always start programs in tmux, forgetting to do so is annoying because you'll have to restart in a tmux if you want to keep 
your program running.

You can manage which gpu's are used with, adding `CUDA_VISIBLE_DEVICES=` before the command. This is usefull to keep one
GPU free for evaluation.

By default both are used on apollo, so nothing is equal to
`CUDA_VISIBLE_DEVICES=0,1`

Select a specific gpu like so: `CUDA_VISIBLE_DEVICES=0 python train.py`


#### using tmux
https://tmuxcheatsheet.com/

In short:

`tmux a -t NAME` or `tmux a` for last used terminal

`tmux new -s NAME`

detatch:

`Ctrl-B D`

scroll

`Ctrl-B [`

exit scroll with `q`






# Test data evaluation
Example test data evaluation command:

`CUDA_VISIBLE_DEVICES=1 python eval_nn.py models/resnet_run0.val_f1.0.154.h5`

# Tips
Monitor the server with `htop` and `nvtop`.


Is your ssh connnection stuck? type the following keystrokes:
`[enter]`
`~`
`.`
`[enter]`
 
Forward ports like this: `ssh apollo -L 6007:localhost:6007`
