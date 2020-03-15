# psosearch

psosearch is a Python library selecting machine learning algorithms parameters.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install psosearch.

```bash
pip install psosearch
```

## Usage

```python
import psosearch

params = {"optimizer":['adam','nadam','sgd','adadelta'],
    "learning_rate":  [0.01,0.2,2],
    "hiddenactivation": ['sigmoid','tanh','relu'],
    "activation": ['sigmoid','tanh','relu']}
task='binary classification'
score='auc'
number_of_particles=10
number_of_iterations=15
p=psosearch('mlp',params,task,score)
p.fitpsosearch(x_train,y_train,x_val,y_val,number_of_particles=number_of_particles,
               number_of_iterations=number_of_iterations)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)