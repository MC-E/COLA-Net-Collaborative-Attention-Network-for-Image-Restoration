# Train
python train.py --dir_data=DIV2K --quality=10 --save_path=exp --lr=2e-4 --mode=E

python train.py --dir_data=DIV2K --quality=10 --save_path=exp --lr=1e-3 --mode=B

Download datasets from the provided links and place them in this directory. Your directory structure should look something like this：
      
`DIV2K` <br/>
  `├──`DIV2K_HQ <br/>
  `└──`DIV2K_LQ <br/>
      `└──Quality factor`

# Test
python test.py --logdir=checkpoints/res_cola_v2_6_3_10_d300/model/model_best.pt --quality=10 --test_data=testsets/Classic5 --ensemble --mode=E

python test.py --logdir=checkpoints/res_cola_v1_6_3_10_d300/model/model_best.pt --quality=10 --test_data=testsets/Classic5 --mode=B
