You can find/customize the install command here: https://pytorch.org/

Generally though, this should work:

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

Then, ensure that CUDA compatibility is enabled by running in the Spyder iPython terminal:

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

You should see the output printed as "cuda:0".

Note, if it does not print "cuda:0" that means it is not linked to your GPU and we need to troubleshoot. I've successfully installed on windows machines without much problem so hopefully this will also be smooth. If it doesn't show up, try reopening your Spyder terminal.

Otherwise, everything else should be the same! Potentially it will ask for other packages to be installed but I'm sure you know the drill by now. Just try:

conda install <package name>
or
pip install <package name>

