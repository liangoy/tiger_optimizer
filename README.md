## Tiger
tiger optimizer for tiger

## feature
When do trainning both on float16 parameter and float16 gradient,round-off error is a serious risk.
This implementation method of tiger optimizer can help you to avoid round-off error if abs(parameter)/1000 > learning_rate * abs(gradient) so that you can do trainning both on float16 parameter and float16 gradient.


## Citation
@misc{su2023tiger,
  title     = {Tiger: A Tight-fisted Optimizer},
  author    = {Jianlin Su},
  year      = {2023},
  howpublished = {\url{https://github.com/bojone/tiger}}
}
