python3 run.py \
--data_path traffic.csv \
--model_id traffic_336_96 \
--model VarFormer \
--data custom \
--features M \
--seq_len 336 \
--label_len 24 \
--pred_len 96 \
--d_model 16 \
--d_ff 64 \
--inter_dim 256 \
--latent_dim 256 \
--n_heads 4 \
--learning_rate 0.0001 \
--weight_decay 0.01 \
--e_layers 2 \
--d_layers 2 \
--enc_in 862 \
--c_in 862 \
--c_out 862 \
--nvars 862 \
--input_dim 862 \
--freq h \
--des Exp \
--itr 1 \
--train_epochs 20 \
--batch_size 32 \
--patience 7 \

python3 run.py \
--data_path traffic.csv \
--model_id traffic_336_192 \
--model VarFormer \
--data custom \
--features M \
--seq_len 336 \
--label_len 24 \
--pred_len 192 \
--d_model 16 \
--d_ff 64 \
--inter_dim 256 \
--latent_dim 256 \
--n_heads 4 \
--learning_rate 0.0001 \
--weight_decay 0.01 \
--e_layers 2 \
--d_layers 2 \
--enc_in 862 \
--c_in 862 \
--c_out 862 \
--nvars 862 \
--input_dim 862 \
--freq h \
--des Exp \
--itr 1 \
--train_epochs 20 \
--batch_size 32 \
--patience 7 \

python3 run.py \
--data_path traffic.csv \
--model_id traffic_336_336 \
--model VarFormer \
--data custom \
--features M \
--seq_len 336 \
--label_len 24 \
--pred_len 336 \
--d_model 16 \
--d_ff 64 \
--inter_dim 256 \
--latent_dim 256 \
--n_heads 4 \
--learning_rate 0.0001 \
--weight_decay 0.01 \
--e_layers 2 \
--d_layers 2 \
--enc_in 862 \
--c_in 862 \
--c_out 862 \
--nvars 862 \
--input_dim 862 \
--freq h \
--des Exp \
--itr 1 \
--train_epochs 20 \
--batch_size 32 \
--patience 7 \

python3 run.py \
--data_path traffic.csv \
--model_id traffic_336_720 \
--model VarFormer \
--data custom \
--features M \
--seq_len 336 \
--label_len 24 \
--pred_len 720 \
--d_model 16 \
--d_ff 64 \
--inter_dim 256 \
--latent_dim 256 \
--n_heads 4 \
--learning_rate 0.0001 \
--weight_decay 0.01 \
--e_layers 2 \
--d_layers 2 \
--enc_in 862 \
--c_in 862 \
--c_out 862 \
--nvars 862 \
--input_dim 862 \
--freq h \
--des Exp \
--itr 1 \
--train_epochs 20 \
--batch_size 32 \
--patience 7 \
