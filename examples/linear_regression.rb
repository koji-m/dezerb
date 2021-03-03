# need to install numo-gnuplot gem to run
require 'dezerb'
require 'numo/gnuplot'

include Dezerb
include Utils
F = Functions

x = Numo::DFloat.new(100, 1).rand
y = 5 + 2 * x + Numo::DFloat.new(100, 1).rand

w = Variable.new(Numo::DFloat.zeros 1, 1)
b = Variable.new(Numo::DFloat.zeros 1)

predict =  Proc.new do |x|
  F.matmul(x, w) + b
end

lr = 0.1
iters = 100
y_pred = nil

iters.times do
  y_pred = predict.(x)
  loss = F.mean_squared_error y, y_pred

  w.cleargrad
  b.cleargrad
  loss.backward

  w.data -= lr * w.grad.data
  b.data -= lr * b.grad.data
  puts "w:#{w}, b:#{b}, loss#{loss}"
end

Numo::gnuplot do
  set title: "linear regression"
  sorted = x.to_a.zip(y_pred.data.to_a).sort_by {|v| v[0][0]}
  x_nar = Numo::NArray.cast(sorted.map {|v| v[0]})
  y_nar = Numo::NArray.cast(sorted.map {|v| v[1]})
  plot [x_nar, y_nar, w: "lines", t: "prediction"],
    [x, y, w: "points", t: "traindata"]
  pause mouse: %w[close]
end

