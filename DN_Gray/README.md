# Train
python train.py --save_path=exp --dir_data=DIV2K --lr=2e-4 --mode=E --noiseL=25
python train.py --save_path=exp --dir_data=DIV2K --lr=1e-3 --mode=B --noiseL=25

# Test
python test.py --mode=E --ensemble --logdir=checkpoints/res_cola_v2_6_3_25_l4/model/model_best.pt --test_noiseL=25. --test_data=testsets/Set12
python test.py --mode=B --logdir=checkpoints/res_cola_v1_6_3_25_l4/model/model_best.pt --test_noiseL=25. --test_data=testsets/Set12
