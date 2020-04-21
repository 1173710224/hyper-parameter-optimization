import os
for i in range(8):
    os.system('nohup python CNNComp.py ' + str(i) + ' f &')
