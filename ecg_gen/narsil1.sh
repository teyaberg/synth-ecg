export CUDA_VISIBLE_DEVICES=0
python train.py --num_workers 32 --name "HRbs512emb256" --data_path "./ecg_gen/ecg_large/val/" --train_dir "./ecg_gen/ecg_large/train/" --embedding_dims 256 --batch_size 512 --epochs 30 --downstream_task "HR" --eval_model_name "linr"
python train.py --num_workers 32 --name "HRbs512emb512" --data_path "./ecg_gen/ecg_large/val/" --train_dir "./ecg_gen/ecg_large/train/" --embedding_dims 512 --batch_size 512 --epochs 30 --downstream_task "HR" --eval_model_name "linr"
python train.py --num_workers 32 --name "HRbs512emb1024" --data_path "./ecg_gen/ecg_large/val/" --train_dir "./ecg_gen/ecg_large/train/" --embedding_dims 1024 --batch_size 512 --epochs 30 --downstream_task "HR" --eval_model_name "linr"
python train.py --num_workers 32 --name "HRbs512emb2048" --data_path "./ecg_gen/ecg_large/val/" --train_dir "./ecg_gen/ecg_large/train/" --embedding_dims 2048 --batch_size 512 --epochs 30 --downstream_task "HR" --eval_model_name "linr"
python train.py --num_workers 32 --name "HRbs512emb128" --data_path "./ecg_gen/ecg_large/val/" --train_dir "./ecg_gen/ecg_large/train/" --embedding_dims 128 --batch_size 512 --epochs 30 --downstream_task "HR" --eval_model_name "linr"
