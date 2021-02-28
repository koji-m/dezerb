# Dezerb

Deep learning frame work for Ruby based on [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3).

## Installation

```shell
$ git clone https://github.com/koji-m/dezerb.git
$ cd dezerb
$ rake install
```

## Usage

```ruby
require 'dezerb'

# forward calculation
x0 = Dezerb::Variable.new(Numo::NArray.cast(1.0))
x1 = Dezerb::Variable.new(Numo::NArray.cast(1.0))
x0.name = 'x0'
x1.name = 'x1'
y = x0 + x1

# backward calculation (differentiation)
y.backward

# print each gradient
p x0.grad, x1.grad

# write calculation graph to image file (add.png)
Dezerb::Utils.plot_dot_graph(y, verbose: false, to_file: 'add.png')
```

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

