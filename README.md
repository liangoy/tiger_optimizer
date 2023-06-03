## Tiger
tiger optimizer for torch

## Feature
When do trainning both on float16 parameter and float16 gradient,round-off error is a serious risk.
This implementation method of tiger optimizer can help you to avoid round-off error if max(parameter)<1 so that you can do trainning both on float16 parameter and float16 gradient.

## Notice

* max(parameter) must be smaller then 1 if you want to train on float16
* lr * weight_decay must be bigger then 1e-4 if you want to train on float16 

## Citation
@misc{su2023tiger,
  title     = {Tiger: A Tight-fisted Optimizer},
  author    = {Jianlin Su},
  year      = {2023},
  howpublished = {\url{https://github.com/bojone/tiger}}
}
