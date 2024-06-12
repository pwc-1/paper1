# USDL
python -u main.py --type USDL --lr 5e-6 --weight_decay 1e-3 --use_i3d_bb 1 --use_swin_bb 0
python -u main.py --type USDL --lr 5e-6 --weight_decay 1e-3 --use_i3d_bb 0 --use_swin_bb 1
python -u main.py --type USDL --lr 5e-6 --weight_decay 1e-4 --use_i3d_bb 1 --use_swin_bb 0
python -u main.py --type USDL --lr 5e-6 --weight_decay 1e-5 --use_i3d_bb 0 --use_swin_bb 1

python -u main.py --type USDL --lr 1e-6 --weight_decay 1e-3 --use_i3d_bb 1 --use_swin_bb 0
python -u main.py --type USDL --lr 1e-6 --weight_decay 1e-3 --use_i3d_bb 0 --use_swin_bb 1
python -u main.py --type USDL --lr 1e-6 --weight_decay 1e-4 --use_i3d_bb 1 --use_swin_bb 0
python -u main.py --type USDL --lr 1e-6 --weight_decay 1e-4 --use_i3d_bb 0 --use_swin_bb 1


## MUSDL
#python -u main.py --type MUSDL --lr 1e-4 --weight_decay 1e-5 --gpu 0,1
