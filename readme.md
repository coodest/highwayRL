# RL with graph memory

## make a env of conda called rl

    conda create -n rl python=3.9 -y

## install components (note: install pytorch depends on your system)

    conda install pytorch cudatoolkit=11.6 -c pytorch -c conda-forge -y
    
## install the dependencies

    pip install -r requirements.txt

## install roms for Atari games

    AutoROM -y


## set GPU to EXCLUSIVE_PROCESS compute mode (optional)

    sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS


## run the code locally

    ./run.sh

## set GPU to DEFAULT compute mode (optional)

    sudo nvidia-smi -i 0 -c DEFAULT

## log explains:

    210627103607 6995 9987 evl_actor R: 300.00 Fps: 2071.9 H:  0.0% L: 0/937                                    
    210627103609 6795 9987 evl_actor R: 600.00 Fps: 2049.2 H:  0.0% L: 0/1071                                   
    210627103613 6496 9787 learner frames:  1.6M fps: 28838.1 G: 327716 V: /                                    
    210627103614 5296 9987 evl_actor R: 400.00 Fps:  900.9 H:  3.7% L: 0/1057       # the episode is not over when sync optimal graph, rest of the episode causes 3.7% hut rate                            
    210627103616 6796 9987 evl_actor R: 1100.00 Fps: 2099.3 H: 100.0% L: 1296/1296
    210627103619 6896 9987 evl_actor R: 1100.00 Fps: 2126.5 H: 100.0% L: 1296/1296
    210627103621 6896 9987 evl_actor R: 1100.00 Fps: 2061.7 H: 100.0% L: 1296/1296

each result of the testing actor just after the graph update should be ignored since the graph used for action generation is altered during this episode.

## To remove `__pycache__`

    find . -type d -name __pycache__ -exec rm -r {} \+
