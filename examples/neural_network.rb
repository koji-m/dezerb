# need to install numo-gnuplot gem to run
require 'dezerb'
require 'numo/gnuplot'

include Dezerb
include Utils
F = Functions

x = Numo::DFloat.new(100, 1).rand
y = Numo::NMath.sin(2 * Math::PI * x) + Numo::DFloat.new(100, 1).rand

I, H, O = 1, 10, 1
w1 = Variable.new(0.01 * Numo::DFloat.new(I, H).rand)
b1 = Variable.new(Numo::DFloat.zeros H)
w2 = Variable.new(0.01 * Numo::DFloat.new(H, O).rand)
b2 = Variable.new Numo::DFloat.zeros O

predict = Proc.new do |x|
  _y = F.linear x, w1, b1
  _y = F.sigmoid _y
  F.linear _y, w2, b2
end

lr = 0.2
iters = 10000

y_pred = nil
iters.times do |i|
  y_pred = predict.(x)
  loss = F.mean_squared_error y, y_pred

  w1.cleargrad
  b1.cleargrad
  w2.cleargrad
  b2.cleargrad
  loss.backward

  w1.data -= lr * w1.grad.data
  b1.data -= lr * b1.grad.data
  w2.data -= lr * w2.grad.data
  b2.data -= lr * b2.grad.data

  p loss if i % 1000 == 0
end

Numo::gnuplot do
  set title: "neural network"
  sorted = x.to_a.zip(y_pred.data.to_a).sort_by {|v| v[0][0]}
  x_nar = Numo::NArray.cast(sorted.map {|v| v[0]})
  y_nar = Numo::NArray.cast(sorted.map {|v| v[1]})
  plot [x_nar, y_nar, w: "lines", t: "prediction"],
    [x, y, w: "points", t: "traindata"]
  pause mouse: %w[close]
end

