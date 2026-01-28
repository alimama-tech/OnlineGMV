#reader
python method_main.py --model='reader' --pretrain_remarks='sharedtwotower' --mode='stream' --gpu_id='4' --lr=0.01  --boost_weight=0.1 --ga_loss_weight=0.5 --train_start_day=57 --train_end_day=81.875
#oracle
python main.py --model='baselinemlp_oracle' --pretrain_remarks='all_pay' --mode='stream' --gpu_id='4'  --train_start_day=57 --train_end_day=81.875  --lr=0.01
python main.py --model='sharebottom_twotower_oracle' --pretrain_remarks='sharedtwotower' --gpu_id='4' --mode='stream' --train_start_day=57 --train_end_day=81.875 --lr=0.001
#online
python main.py --model='baselinemlp_vanilla' --pretrain_remarks='all_pay' --mode='stream' --gpu_id='4'  --train_start_day=57 --train_end_day=81.875 --lr=0.01
python method_main.py --model='sharedbottom_twotower_vanilla' --pretrain_remarks='sharedtwotower' --mode='stream'  --gpu_id='4' --lr=0.01 --train_start_day=57 --train_end_day=81.875
#offline
python method_main.py --model='baselinemlp_bdl' --pretrain_remarks='all_pay' --mode='stream' --gpu_id='4' --lr=0.0001 --train_start_day=57 --train_end_day=81.875
python method_main.py --model='shredbottom_twotower_bdl' --pretrain_remarks='sharedtwotower' --gpu_id='4' --mode='stream' --lr=0.001 --train_start_day=57 --train_end_day=81.875

#pretrain
python main.py --model='baselinemlp'  --gpu_id='4' --pretrain_remarks='all_pay' --mode='pretrain' --train_start_day=0 --train_end_day=50 --lr=0.001
python main.py --model='sharebottom_twotower'  --gpu_id='4' --pretrain_remarks='sharedtwotower' --mode='pretrain' --train_start_day=0 --train_end_day=50 --lr=0.001
python method_main.py --model='classifier' --mode='pretrain' --gpu_id='4' --pretrain_remarks='classifier' --lr=0.002
python method_main.py --model='calibrator' --mode='pretrain' --gpu_id='4' --pretrain_remarks='calibrator' --lr=0.001


