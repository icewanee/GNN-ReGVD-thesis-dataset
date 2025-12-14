python3.9 -u run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/juliet/train_CWE78.jsonl --eval_data_file=../dataset/juliet/val_CWE78.jsonl --test_data_file=../dataset/juliet/test_CWE78.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee training_log_juliet_CWE78.txt

python3.9 -u run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/juliet/train_CWE134.jsonl --eval_data_file=../dataset/juliet/val_CWE134.jsonl --test_data_file=../dataset/juliet/test_CWE134.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee training_log_juliet_CWE134.txt

python3.9 -u run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/juliet/train_CWE190.jsonl --eval_data_file=../dataset/juliet/val_CWE190.jsonl --test_data_file=../dataset/juliet/test_CWE190.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee training_log_juliet_CWE190.txt

python3.9 -u run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/juliet/train_CWE401.jsonl --eval_data_file=../dataset/juliet/val_CWE401.jsonl --test_data_file=../dataset/juliet/test_CWE401.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee training_log_juliet_CWE401.txt

python3.9 -u run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/juliet/train_CWE415.jsonl --eval_data_file=../dataset/juliet/val_CWE415.jsonl --test_data_file=../dataset/juliet/test_CWE415.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee training_log_juliet_CWE415.txt

python3.9 -u run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/juliet/train_CWE690.jsonl --eval_data_file=../dataset/juliet/val_CWE690.jsonl --test_data_file=../dataset/juliet/test_CWE690.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee training_log_juliet_CWE690.txt

python3.9 -u run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/juliet/train_CWE762.jsonl --eval_data_file=../dataset/juliet/val_CWE762.jsonl --test_data_file=../dataset/juliet/test_CWE762.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee training_log_juliet_CWE762.txt

python3.9 -u run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/juliet/train_CWE789.jsonl --eval_data_file=../dataset/juliet/val_CWE789.jsonl --test_data_file=../dataset/juliet/test_CWE789.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee training_log_juliet_CWE789.txt