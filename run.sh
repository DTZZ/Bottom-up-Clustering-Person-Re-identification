dataset=market1501
#dataset=duke
#dataset=mars
#dataset=DukeMTMC-VideoReID

batchSize=16
size_penalty=0.003
merge_percent=0.05

logs=logs/$dataset


python run.py --dataset $dataset --logs_dir $logs \
              -b $batchSize --size_penalty $size_penalty -mp $merge_percent

# python run.py --dataset market1501 --logs_dir logs\market1501 -b 16 --size_penalty 0.003 -mp 0.05